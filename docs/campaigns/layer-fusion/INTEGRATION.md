# sm_70 整层融合 launch（fused per-wave BF16 projection）接入 llama.cpp fork — 集成与验证报告（2026-10-03）

## 结论（TL;DR）

- **接入完成，parity 门 4/4 全过（512-ID 逐位一致 + greedy 不漂移），端到端性能中性（差异在噪声内），开关默认开。**
- 生产 preset（island-h-fifo-single，8192 输入 / 512 输出贪心，seed 1234，`GGML_SM70_BF16_GEMV=1` 固定）同二进制 A/B（`GGML_SM70_FUSED_PROJ` 0/1）：

| 臂 | PP tok/s | TG tok/s | predicted_ms | prompt_ms | parity |
|---|---|---|---|---|---|
| fusedproj-on | 955.1 | **111.24** | 4593.5 | 8577.5 | matched ✓ |
| fusedproj-on-r2 | 955.8 | **110.94** | 4606.3 | 8570.9 | matched ✓ |
| fusedproj-off | 956.7 | **111.19** | 4595.8 | 8562.5 | matched ✓ |
| fusedproj-off-r2 | 949.5 | **111.10** | 4599.4 | 8620.0 | matched ✓ |

  - TG 均值 111.09 vs 111.14（-0.05%）、predicted_ms 4599.9 vs 4597.6（+0.05%）——**均在臂间噪声（±1.5 tok/s）内，诚实结论：端到端中性**。
  - **组合数**：on 臂即组合（launcher 固定 `GGML_SM70_BF16_GEMV=1` + `GGML_SM70_FUSED_PROJ=1`），与已落袋的单 kernel 优化叠加无冲突、无回退。
- parity：4 次运行 512/512 token ID **全部逐位一致**（vs speed-bundle-single 参考及 bf16gemv-on manifest 交叉一致）；llama-cli 默认（不设 env）贪心文本与 off 臂逐字一致。
- **为什么中性（重要，修正设计侧假设）**：DESIGN §2 的 BF16 清单与运行时不符。GGUF 实测（blk.1 全量张量类型）：attn_qkv=IQ4_XS、attn_gate=Q6_K、ssm_out=IQ3_S、shexp up/gate/down=IQ3_S/IQ4_XS/IQ4_NL——**wqkv/wqkv_gate/wq/wk/wv/ssm_out/shexp 全部量化，不是 BF16**。真实 BF16 人口 = hc 六件套（down/inject/up ×attn/ffn）+ router + sh_router + ple_key/value + beta/alpha（后两者被 fork 现有 mmvf epilogue 占用）。可融合且分配器布局允许的组每 eval 仅 ~6 个（见 §覆盖率），远小于 DESIGN §9 假设的 ~780 launch/token，收益被稀释到噪声以下。
- 开关默认开（`GGML_SM70_FUSED_PROJ`，缺省=1）；`=0` 一键回退原路径（无需重编译）。

## 改动文件清单（相对接入前工作树，基线 = 服务器 `/tmp/pre_fused_baseline.diff`，已另存 `experiment/fused-proj-integration/pre_fused_baseline.diff`）

| 文件（服务器 /home/debian/ac922-llama-next/sources/llama-v100-v3/） | 改动 |
|---|---|
| `ggml/src/ggml-cuda/fused-projection-sm70.cuh` | **新增**（拷贝自 layer-fusion/fused_projection_sm70.cuh，namespace fusedproj_sm70，自包含，零新依赖） |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +1 行 include；+~250 行（识别器 `ggml_cuda_is_fused_proj_candidate` / `ggml_cuda_fused_proj_feeds_glu` + 图遍历钩子 `ggml_cuda_try_fused_proj` + try_fuse 内 4 行调用） |

增量 patch：`experiment/fused-proj-integration/ggml-cuda.cu.fusedproj.incremental.patch`（301 行 unified diff）。
BF16 GEMV 已落袋改动（kernel_sm70_bf16.cuh + mul_mat 分支）**零改动**；工作树其余 129 个文件零改动。

### 接线结构（照 DESIGN §8，mmvf_multi 同构 + 三处运行时必要的扩展）

1. 挂点：`ggml_cuda_try_fuse` 内、`ggml_cuda_try_mmvf_multi` 块之后。返回 skip span；组内后续成员塞 `cuda_ctx->precomputed`（现成机制）→ 成员不再进 `ggml_cuda_mul_mat`，**不会与 GGML_SM70_BF16_GEMV 单 kernel 双发**。
2. 成员识别：BF16 leaf 权重（src0->op==NONE）× F32 激活、2D、ncols 1..8、同 y（指针/shape/stride，同 mmvf_multi 判据）、窗口 64 节点、≤8 任务（static 表 NT=8 上限）、成员间 overlap 安全校验逐条复刻 mmvf_multi（between 节点不得读写成员输出/y）。
3. **排除守卫**（设计清单与运行时图结构的三个碰撞点，全部实测发现）：
   - `ggml_cuda_match_gemv_chain().post_op != 0` 的候选不收（beta/alpha 的 sigmoid/softplus epilogue 留给 fork 现有融合，一次 kernel 连链算完更优）；
   - 喂 GLU 的候选不收（`ggml_cuda_fused_proj_feeds_glu`：8 节点窗内 GLU 的 src 直接/经一层 wrapper 指向该 mm）——fork 的 mm+mm+GLU 融合本来一次 launch 算俩+GLU；
   - `live>=2` 才融合（单矩阵 wave 只有认领开销，微基准 W2/4/6/8 微亏 7-13%，DESIGN §10 决策照办）。
4. **anchor 尾随 scale 内联计算**（运行时发现的最大障碍）：`build_lora_mm(w,x,w_s)` 给每个投影 mm 后面挂 `MUL(mm, w_s)`，hc_down 后挂 `GGML_OP_SCALE(mm, 1/hc)`——第一个 live 后继节点挡死 skip span（span<1 会让 anchor 落回逐矩阵路径被重算）。解法照 fork 自己的融合契约（ggml-cuda.cu:4979 有先例）：该 scale 节点由钩子内 `ggml_cuda_compute_forward` **内联计算**并计入 span，至多一个。scale 输出与 anchor 输出**原地别名**（分配器复用，anchor 唯一消费者就是 scale）是合法读后写，anchor 自身豁免 overlap 检查；其余 between 节点照查。
5. 表形态：**static**（`fused_projection_sm70_launch_static<8>`，任务表按值进 kernel 参数/constant bank）——无 cudaMalloc/无 H2D/无表缓存，CUDA Graph 捕获直接嵌入参数，重放零额外节点；图变体驱逐/重建无悬挂面。DESIGN §8.2 的 fused_proj_table 缓存仅 dyn 形态需要，static 形态天然免除。
6. 事后动作：成员输出 `q8_1_cache_invalidate_range`（早写缓冲的缓存量化失效；span 内节点主循环也会查，此为补齐）。
7. 开关：`GGML_SM70_FUSED_PROJ`（缺省=1；0=off；2=+每组日志前 96 条；3=2+拒绝原因诊断）。防回退守卫：仅 `cc==VOLTA`。

## 实际命中的融合组（运行时证据）

- llama-cli 6 卡冒烟（nc 1/2/4，30 组）：**每 eval 每卡 1 组 {hc_attn_down(320×10240) + hc_attn_inject(4×10240)}**（6 卡全命中，span=1，scale 内联）+ **dev0 {ple_value(2560×2560) + ple_key(10240×2560)}**。
- 服务端（island-h-fifo + MTP，arm-2 日志 28 组）：**nc=5（MTP verify）组命中**（dev1/dev2 各 4 组 + dev0/dev2 各 1）+ nc=2/nc=1 组。CUDA Graph 捕获后按图重放，无重复日志。
- DESIGN §2 清单对照：W1/W5 = {down,inject} 2 成员 ✓（实测命中）；W3（wqkv 族）/W3'（wq/wk/wv）/**非 BF16，不存在于本模型**；beta/alpha（W3 成员）被 stock epilogue 占用 ✓ 按设计放行；W7 的 up/gate_shexp 喂 GLU + 量化；router↔sh_router 之间隔着整个 MoE 簇（>64 窗口）且 8 字节小输出被 between 节点复用（overlap 正确拒绝）。
- **覆盖率硬上限（分配器别名）**：同一层 hc_down 输出被 scale 消费后即死，分配器把 inject 输出放回同一 range → 组员输出与 anchor 输出重叠 → 安全拒绝（单 launch 内任务块无序，v1 kernel 无任务间顺序保证，融合即数据竞争）。首个 per-device 层的布局恰好不相交所以命中。这是 ggml 分配器复用语义下的结构性限制，不是 bug——拒绝逻辑在冒烟中拦下的全部是该类真竞争。

## 验证记录（全部在服务器 build-ac922，增量编译，无新增警告——3 条 cast-qual 为基线既有 stage_if_overlap 代码，行号平移）

1. **冒烟**：llama-cli 6 卡 24 token 贪心正常，`GGML_SM70_FUSED_PROJ=3` 拒绝原因诊断 + `=2` 组清单 + `GGML_FT_GRAPH_CENSUS=1` 交叉核对（hc/ple 族 skip 计数变化）。
2. **parity**：preset 服务 `scripts/launch_fusedproj_arm.sh <0|1|2>`（systemd-run，env 同 island-h-fifo-single configuration.json，`GGML_SM70_BF16_GEMV=1` 固定），`scripts/matrix_runner.py --cases 1:8192 --reference-directory manifests/speed-bundle-single` × 4：两臂输出 token ID 与参考**逐位一致**（`manifests/fusedproj-{on,off}{,-r2}/c1-p8192.json.parity.json` matched:true ×4），且与 bf16gemv-on manifest 完全相同（跨战役传递一致）。DESIGN §8.6 点名的 ROWS 归约序风险（hc_inject 从 R1 分块并入 R4 波配置）：**512 贪心 token 未翻转**——实测结论，非逐位等价保证（与 bf16-gemv 战役同级别声明）。
3. **greedy 文本**：llama-cli 默认（不设 env）vs off，24 token 逐字一致；服务端 512 token ID 一致即文本一致。
4. **性能**：见 TL;DR 表。注：on-r1 带 mode=2 日志（首次 96 组 ~28 行 stderr，热路径无感）；PP 臂间波动（949-957）与 bf16-gemv 战役观测一致（页缓存主导）。

## 回退方法

- **运行时（首选）**：设 `GGML_SM70_FUSED_PROJ=0`（launcher 脚本已支持，0/1/2 参数），无需重编译；缺省即开。
- **文件级（彻底移除）**：
  1. `cp experiment/fused-proj-integration/ggml-cuda.cu.pre-fusedproj sources/llama-v100-v3/ggml/src/ggml-cuda/ggml-cuda.cu`
  2. `rm sources/llama-v100-v3/ggml/src/ggml-cuda/fused-projection-sm70.cuh`
  3. build-ac922 内重跑 `ninja llama-server`
  pre-fusedproj 文件由 HEAD + `pre_fused_baseline.diff` 的 ggml-cuda.cu 段净重建（含 BF16 GEMV 已落袋改动），回退后 `git diff` 与 `/tmp/pre_fused_baseline.diff` 等价。

## 工件路径

- 服务器：`/home/debian/ac922-llama-next/experiment/fused-proj-integration/`（incremental.patch、pre-fusedproj 文件、kernel 副本、pre_fused_baseline.diff、server-arm2-grouplist.log）
- 服务器：`/home/debian/ac922-llama-next/manifests/fusedproj-{on,off}{,-r2}/`（4 组基准+parity JSON）、`logs/fusedproj-arm{0,1,2}.log`
- 服务器：`/home/debian/ac922-llama-next/scripts/launch_fusedproj_arm.sh`（A/B 启动器，0/1/2 参数）
- 本地镜像：`sources/llama-v100-v3` 已同步（含 fused-projection-sm70.cuh 与 kernel_sm70_bf16.cuh 补拉）；`layer-fusion/integration-artifacts/` 为上述工件镜像
- 接入前基线快照：服务器 `/tmp/pre_fused_baseline.diff`（16533 行，含 BF16 GEMV 已落袋改动）

## 残留事项 / 边界 / 后续

- 测试后 systemd 单元已停、6 卡显存 0 MiB、无残留进程；llama-cli 冒烟 OOM 一次为后台残留测试服务占卡，非本改动问题，清理后复跑通过。
- 端到端中性的根因是**模型量化方案**（关键投影非 BF16）而非 kernel 或接线失败；hc/ple 组的单组收益在微基准中成立（W1 -43%、PLE 组 65MB 权重流单 launch），但每 token 命中组数太少（~6 组 vs 数百 launch），被 23ms 步长淹没。
- 若后续换 BF16 权重的 GGUF 变体（或 wqkv 反量化），本接线无需改动即自动扩大覆盖；coverage 由分配器布局决定的部分（输出别名拒绝）需要 kernel v2 的任务间顺序（两段式或按依赖序多 launch）才能解锁——DESIGN §8.7 二期方向。
- mode=3 诊断日志保留（env 门控，mode 0/1/2 零开销路径），供后续覆盖分析复用。
- 本轮未动：MUL_MAT_ID 专家组、mmvf_multi、GLU/epilogue 现有融合（全部原样，优先级在其后）。
