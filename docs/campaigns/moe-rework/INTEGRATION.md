# sm_70 MoE Q2_0 down GEMV kernel 接入 llama.cpp fork — 集成与验证报告（2026-10-04）

## 结论（TL;DR）

- **接入完成（默认关，opt-in），parity 门全过，decode 实测 +1.16%（每 512-token 窗省 69ms），prefill 持平。**
- 生产 preset（island-h-fifo-single，8192 输入 / 512 输出贪心，seed 1234）同二进制 A/B（4 轮，on→off→on→off）：
  - TG（decode）：开关开 **93.15 / 92.98 tok/s**，开关关 **91.98 / 91.83 tok/s**
    → 每臂均值：93.06 vs 91.90 = **+1.27%**（按 predicted_ms：5485.8/5496.0 vs 5555.4/5564.9，每窗 −69.2ms）
    臂内波动 ~0.2%，信号为噪声的 ~6×。与裁决时按流量勘误重估的 MoE 剩余空间（+0.9-1.4%）一致。
  - PP（prefill）：622.1/622.7 vs 621.5/623.7 tps → 持平（预期：prefill 不走 nc>1 moe matvec 路径）。
- **parity：4/4 轮 512 token ID 逐位一致**（且与 speed-bundle-single 参考一致，matched:true ×4）；
  两臂 MTP draft 接受率均 408/408（生成路径完全一致）。
- 开关 `GGML_SM70_MOE_REWORK` **默认 0（原路径）**；=1 启用；=2 启用+stderr 记录命中形状；运行中可切。
- 微基准：Q2_0 down **3.0×**（K=256 nc5, 38.5→12.9µs）/ **3.3×**（K=192）；生产 per-token 路由口径 **1.9×、57% copy**。
- 端到端只有 +1.16% 的原因（诚实记录）：生产 launch 是 distinct 路由（~50 专家/launch、9.2MB 真实流量），
  4.2µs 级 kernel 收益（每 launch ~19µs）摊到含 draft/attention/dense 的整步上，且 nc∈{2,4,5} 混窗收益递减；
  微基准的 3.0× 是 uniform-ids（10 专家）口径——两者都是真数，口径不同。

## 改动文件清单（相对接入前快照，其余文件零改动）

| 文件（服务器 sources/llama-v100-v3/） | 改动 |
|---|---|
| `ggml/src/ggml-cuda/kernel_sm70_moe.cuh` | **新增**（235 行，namespace moe_sm70，自包含；逐设计决策注释含实测依据） |
| `ggml/src/ggml-cuda/mmvq.cu` | +1 行 include；+32 行分发分支（`mul_mat_vec_q_moe_launch()` 顶部） |

完整 patch：`integration-artifacts/mmvq.cu.integration.patch`（46 行）。
回退基线：`experiment/moe-rework-integration/mmvq.cu.pre-integration` + `pre_moe_rework_baseline.diff`（16813 行，接入前全树既有改动）。

### 挂点与守卫（照 bf16-gemv 的 DQ_MMV 纪律）

- 位置：`mul_mat_vec_q_moe_launch()`——MUL_MAT_ID 多 token（ncols_dst>1）专用 moe 路径的唯一分发点；nc=1 与 dense MUL_MAT 不经过此函数，零影响。
- 守卫（全部满足才走新核，否则原路径）：`type==Q2_0`、Volta（cc==700 且最高编译 arch==700）、无 fusion（gate/x_bias/gate_bias/x_scale/gate_scale 全 null）、K∈{192,256}（`moe_sm70_q2_0_valid` 内建白名单+`stride_row_x==K/64` 行连续性校验）、2≤nc≤8。
- 环境开关：`GGML_SM70_MOE_REWORK`（static-lambda-getenv 模式，照 `GGML_SM70_BF16_GEMV`）；默认 **0**。
- CUDA Graph 兼容：kernel 无动态分配/无 setAttribute（smem ~10KB < 48KB 静态限额）、launch 走 fork 的 `ggml_cuda_kernel_launch`，preset graphs-on 下捕获期分发一次即生效。

## 微基准证据（阶段一，2026-10-04，详见阶段一报告）

bench：`/home/debian/moe-rework/bench_moe_rework.cu`（fork kernel 逐字节提取同台对比 + 双精度 CPU 参考，参考的元素映射逐行转录自 fork `dequantize_row_q2_0`）。

| 形状（每卡 10 专家） | fork µs | 新核 µs | 加速 |
|---|---|---|---|
| K=256 nc5（uniform ids，审计口径） | 38.54 | **12.89** | **3.00×** |
| K=192 nc5 | 38.48 | **11.73** | **3.28×** |
| K=256 nc1* | 20.74 | 6.71 | 3.09× |
| K=256 nc5 **distinct ids（生产路由语义）** | 39.53 | **20.74** | 1.91×（57% copy） |

*nc1 为 bench 直测；生产 nc=1 走 dense 路径不经本挂点。
正确性：50 项（形状×nc×ids 模式×配置）全 PASS，max rel err 2.8e-5（门槛 2e-4）；fork 基线在同一数据上过同一门槛。

## 流量勘误（对审计报告的重要修正，供战役记录）

MUL_MAT_ID 的 ids 语义是 `ids[slot + token*ids_stride]`——**每 token 独立路由专家**。nc5 时一次 launch 实际读 ~50 个不同专家的整矩阵（Q2_0 down ≈ 9.2MB/launch，非审计假设的 10 专家 1.84MB，差 ~5×）。推论：
- fork 在 R17 的 41.4µs 实际带宽 ~220GB/s（**25% copy**），审计"4.2%"判低了 ~6×；MoE 家族的"大病"程度被高估。
- distinct 口径物理下限 = 9.2MB/870GB/s + 3.5µs 调度地板 ≈ **14µs**——"4×"目标（<10.4µs）在生产路由语义下不可达；本核 20.7µs 已到 57% copy。
- 审计 §3 的 MoE 池子收益（950ms）基数同步折半以上：三方（Q2_0+IQ4_NL+IQ3_S）现实合计 ~125-200ms（估算），全 MoE 剩余空间 <2% 步时——与裁决一致，IQ4_NL/IQ3_S 不追加投入。

## 未采纳/已知边界

- **IQ4_NL down（K=224）**：微基准 nc5 1.11×、distinct 0.92× 倒退——不接入（本 kernel 也仅实例化 Q2_0）。后续若做：解码 ALU 是瓶颈（非带宽），需专用解码器。
- **IQ3_S gu fused**：解码 ALU 瓶颈（分相实测 44.5µs/48µs 是纯解码）；试过用原生 AND+dp4a 分裂替代 Volta 模拟的 `__vcmpne4/__vsub4`，实测更慢（掩码构建 24 标量指令 > ~5 条模拟），fork 指令组合已近该 ISA 最优。上 1.5×+ 需 iq2s_volta 式解码器重构（池子折算 ~+0.6% 步时，性价比不足，裁决不投入）。
- **staging 窗口越界读**：对齐窗口最多越界 15B 落入相邻张量数据/缓冲余量，从不解码；down_exps 处于权重缓冲中部（后随 norm/output 张量），无 OOB 风险。若未来 down 权重独占缓冲尾部需复核。
- **float 归约顺序**与 fork warp-reduce 不同（确定性）；512 贪心 token parity 见下。

## 验证记录（服务器）

1. **构建**：build-ac922 增量 16 目标，无新增警告。
2. **冒烟+形状核实**：6 卡 llama-cli（island 3,3 / MTP d=4 / graphs 默认）+ `GGML_SM70_MOE_REWORK=2`：24 token 贪心正常，命中 450 次零 assert/CUDA error；直方图 `K=192 N=2560 nc=5×120 / nc=4×120`、`K=256 N=2560 nc=5×58 / nc=4×58 / nc=2×86`——**全部在白名单内**，K=224 零出现（IQ4_NL 层）。
3. **parity + A/B**（与 F 线错峰，6 卡窗口内完成）：
   - 4 轮 `matrix_runner --cases 1:8192`（on/off 各 2，`scripts/run_moerework_ab_all.sh`，
     启动器 `scripts/launch_moerework_arm.sh` 把 `GGML_SM70_BF16_GEMV` 钉在 1 保证单变量）：
     - parity：`manifests/moerework-{on,off}{,-r2}/c1-p8192.json.parity.json` 全部 `matched: true`（512 token 逐位，参考 speed-bundle-single）
     - A/B：predicted_ms 5485.8/5496.0（on）vs 5555.4/5564.9（off）；decode_tps 93.15/92.98 vs 91.98/91.83；draft 接受 408/408 四轮一致
   - 窗口结束 server 已停、6 卡 0 MiB，无残留进程。

## 回退方法

- **运行时（首选）**：不设 `GGML_SM70_MOE_REWORK`（默认 0 即原路径）；运行中一键 `=0`。
- **文件级**：`cp experiment/moe-rework-integration/mmvq.cu.pre-integration sources/llama-v100-v3/ggml/src/ggml-cuda/mmvq.cu && rm sources/llama-v100-v3/ggml/src/ggml-cuda/kernel_sm70_moe.cuh && ninja -C build-ac922 llama-server llama-cli`。

## 工件路径

- 服务器：`/home/debian/ac922-llama-next/experiment/moe-rework-integration/`（patch、pre-integration、kernel 副本、基线 diff）
- 服务器：`/home/debian/ac922-llama-next/manifests/moerework-{on,off}{,-r2}/`（4 组基准+parity JSON）、`logs/moerework-arm{0,1}.log`、`logs/moerework_ab_driver.log`
- 服务器：`/home/debian/moe-rework/`（阶段一 bench + run_final.log + mini*.cu 诊断）
- 服务器：`scripts/launch_moerework_arm.sh`、`run_moerework_ab.sh`、`run_moerework_ab_all.sh`（A/B 启动器，参数 0/1 控制开关，BF16 开关钉在 1 保证单变量）
- 本地：`moe-rework/`（kernel_sm70_moe.cuh、moe_matvec_sm70.cuh/bench_moe_rework.cu 阶段一版、INTEGRATION.md、integration-artifacts/）

## 残留事项 / 边界

- 默认关：生产 preset 想吃到 +1.16% 需显式设 `GGML_SM70_MOE_REWORK=1`（launcher 脚本已支持）；后续跑稳后可讨论改默认。
- 本轮只覆盖 Q2_0 down（白名单 K∈{192,256}）；IQ4_NL down（K=224）与 IQ3_S gu（含 fused）走原路径——微基准显示前者 distinct 倒退、后者解码 ALU 瓶颈，均为后续可选战役（IQ3_S 解码器重构池子折算 ~+0.6%，已裁决不投入）。
- A/B 为单请求两臂各 2 次（bounded 单波）；+1.16% 为臂内波动（~0.2%）的 ~6×，方向明确。
- 512-ID parity 是贪心 token 级一致；若把 kernel 用到白名单外形状（未来扩 K 或 nc>8）需重跑 parity 门。
