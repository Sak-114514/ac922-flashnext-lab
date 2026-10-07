# ik_llama.cpp fused_mul_mat_vec_q（MoE gu+gate 结构）移植接入 — 集成与验证报告（2026-10-04）

## 结论（TL;DR）

- **移植完成、接线完成、parity 全过（512-ID greedy 逐位 matched:true），但性能判决为负：保持默认关（GGML_IK_FUSED_MOE=0），不投产。**
- 正确性层面无可挑剔：standalone 对拍 fork 现核 **40/40 组全部逐位相同（0 bit diff，含 IQ2 家族全格式、uniform/distinct 双 ids 口径）**；全模型 512 token greedy parity 与 speed-bundle-single 参考逐位一致。
- 性能层面（同 harness 同数据头对头）：**按本 fork 生产语义（每 (slot, token) 独立路由专家）正确实现的移植核，在全部 8 组形状 × 2 种 ids 口径下比 fork 现核慢 0.59×–0.95×**；全模型 e2e 单轮验证 **decode 88.76 vs 基线 91.98/91.83 tok/s = −3.6%**。
- **根因定位（实验证实，非推测）**：阶段一报告的 1.4–2.3× 收益**全部来自 x 解码在 j 循环上的 CSE 共享**——这要求一个 channel（slot）内全部 token 用同一个专家（ik 原核只读 `ids[channel]` 一次，是其 nc==1 专用 fast path 的语义假设）。本 fork 的 MUL_MAT_ID 路由是 `ids[slot + token*ids_stride]` 每 token 独立（moe-rework/INTEGRATION.md 流量勘误已证：nc5 一次 launch 实际读 ~50 个不同专家）。把 x 基址改成 per-token 后编译器无法 CSE，收益归零，只剩 1-warp 串行 5 token 的低占用率惩罚。
- **阶段一报告的数字本身可复现**（faithful 形态 60.7–67.2µs vs 报告 69.3µs，同数量级）——但该形态在 nc>1 下语义错误（token 1..4 会用 token 0 的专家权重算自己的激活），不可投产。两条 FAIL 线索就此闭环：报告 §2.5 的 IQ2 家族 3e-4~8e-3 relerr 是对浮点 CPU 参考的恒等式截断差，与 fork 现核逐位相同，无需开关挡 IQ2 系。
- 附带发现（对未来 kernel 工作有价值）：vec_dot 调用外层的**运行时 `if (use_gate)` 分支本身使核慢 ~12%**（190.6→168.4µs，IQ3_S N640 nc5）；编译期 gate 化是免费午餐，已记录未实施（本核默认关，不值得再优化）。
- 原计划的 4 臂 A/B 与三开关组合轮**取消**：冒烟门（任务书规定的先决步骤）已在 kernel 级判决为负，e2e 单轮复证 −3.6%（信号为臂内噪声 ~0.2% 的 ~18 倍），继续烧 6 卡窗口只会复述已知回归。IQ2 家族真实 GGUF 对拍由"逐位相同 + 512-ID parity 过（模型含 IQ2 家族 gu+gate 层，命中日志确认）"覆盖，无需单独跑。

## 改动文件清单（相对接入前快照，其余文件零改动）

| 文件（服务器 sources/llama-v100-v3/ggml/src/ggml-cuda/） | 改动 |
|---|---|
| `kernel_ik_fused_moe.cuh` | **新增**（283 行，namespace ik_fused_moe，自包含；头部注释含实测判决，防止未来误开） |
| `mmvq.cu` | +1 行 include；+42 行分发分支（`mul_mat_vec_q_moe_launch()` 内，Q2_0 rework 分支之后、iq2s_volta 分支之前） |

- 集成 patch：`integration-artifacts/mmvq.cu.ikfused.patch`（55 行）。
- 回退基线：服务器 `experiment/ikfused-integration/pre_ikfused_baseline.diff`（16850 行，接入前全树既有改动）+ `mmvq.cu.pre-ikfused`（接入前文件副本）。

### 挂点与守卫（照 bf16-gemv / moe-rework 纪律）

- 位置：`mul_mat_vec_q_moe_launch()`——MUL_MAT_ID 多 token（ncols_dst>1）专用 moe 路径的唯一分发点；nc=1 与 dense MUL_MAT 不经过此函数，零影响。
- 守卫（全部满足才走新核，否则原路径）：`ik_fused_moe::supported<type>()`（仅 IQ2_XXS/XS/S、IQ3_XXS/S——R17 gu+gate 池；Q2_0 与 IQ4_NL 不实例化，与 SM70_MOE_REWORK down 路径类型天然分离）、Volta、2≤nc≤8、无 NVFP4 scale fusion（x_scale/gate_scale 全 null）、**IQ2_S 在 `mmvq_moe_q8_out` 武装时让位给 iq2s_volta**（保住 down 链的 q8_1 cache 命中，摘除风险为零）。
- 环境开关：`GGML_IK_FUSED_MOE`（static-lambda-getenv，照 `GGML_SM70_BF16_GEMV` 模式）；默认 **0**；=2 启用+stderr 记录命中形状。
- CUDA Graph 兼容：无动态分配/无 setAttribute、smem=0、launch 走 fork 的 `ggml_cuda_kernel_launch`，preset graphs-on 下捕获期分发一次即生效（e2e 验证轮即 graphs-on 配置）。

## 移植形态（为什么不是逐字节拷贝）

ik 原核（`fused_mul_mat_vec_q`，mmvq-templates.cuh:305-330）`i02 = ids_data[i2*ids_nb0]` 每 channel 只读一次专家 id——对 nc>1 的 per-token 路由**语义错误**。本移植保持 ik 结构（grid=(ceil(N/2), slots)、block=(32,1)、`__launch_bounds__(32,1)`、tmp/tmp_gate[nc][2] 累加组、SwiGLU 写回融合），但：

- x 基址 per-token：`expert[j] = ids[channel_dst + j*ids_stride]`，j 循环内分别寻址；
- 接 fork fusion 通道（`ggml_cuda_mm_fusion_args_device`：gate/x_bias/gate_bias/glu_op/glu_limit，per-expert bias 索引）；
- kbx/lane 划分、`warp_reduce_sum`、GLU helper 与 fork 现核完全一致 → **输出逐位相同**（这是 parity 全过的结构性原因，不是巧合）。

## 微基准证据（standalone，2026-10-04，GPU0 空载，best-of-3 stream 循环，≥24MB 轮转）

harness：`bench_ikfused_smoke.cu`（fork 现核 mmvq_part1/2.inc 逐字节提取 + fork 树上 kernel_ik_fused_moe.cuh 直连 include，同数据同 ids 头对头）。
形状：K=2560、10 专家槽、gate+up 融合（SWIGLU 无 bias）。fork 侧数字与 G 阶段一 bench-fork 复测一致（±10% 时钟波动内）。

### 判决表 A — 正确语义移植核（树上的 kernel_ik_fused_moe.cuh）vs fork 现核

| 形状 | 格式 | fork µs（uniform/distinct） | 移植核 µs（uniform/distinct） | 加速比 uniform / distinct |
|---|---|---|---|---|
| N640 nc5 | IQ3_S | 156.6 / 147.5 | 190.6 / 223.0 | **0.82× / 0.66×** |
| N640 nc5 | IQ2_XS | 120.2 / 126.4 | 155.6 / 183.6 | **0.77× / 0.69×** |
| N640 nc5 | IQ2_XXS | 99.8 / 103.8 | 143.1 / 163.2 | **0.70× / 0.64×** |
| N640 nc5 | IQ3_XXS | 111.6 / 119.2 | 167.9 / 200.3 | **0.66× / 0.60×** |
| N640 nc5 | IQ2_S | 126.1 / 133.5 | 166.9 / 193.1 | **0.76× / 0.69×** |
| N224 nc5 | IQ3_S | 52.1 / 53.7 | 68.5 / 79.9 | **0.76× / 0.67×** |
| N224 nc5 | IQ2_XS | 46.3 / 52.9 | 60.1 / 76.7 | **0.77× / 0.69×** |
| N224 nc2 | IQ3_S | 26.3 / 30.5 | 33.9 / 43.0 | **0.78× / 0.71×** |
| N224 nc2 | IQ2_XXS | 20.8 / 23.3 | 23.1 / 27.1 | **0.90× / 0.86×** |

（完整 40 组见 `run_smoke.log`；全部 bitdiff=0。nc2 最好情形也只有 0.86–0.91×。）

### 判决表 B — 归因实验（同 harness 四形态对照，IQ3_S N640 nc5）

| 形态 | ids 语义 | µs | vs fork（141–157µs） |
|---|---|---|---|
| fork 现核 mul_mat_vec_q_moe | per-token（正确） | 141.2–156.7 | 1.00× |
| 移植核（树上版，运行时 `if(use_gate)`） | per-token（正确） | 190.6 | 0.74–0.82× |
| 移植核 branchless（gate 编译期化） | per-token（正确） | 168.4 | 0.84–0.86× |
| **faithful 形态（ids 每 channel 读一次，ik 原语义）** | **channel-uniform（nc>1 语义错误）** | **60.7–67.2** | **2.33×** |

faithful 形态复跑了 G 的 bench_ik 原二进制（69.18µs 当场复现）与我的 branchless faithful（60.7/67.2µs 两次）——阶段一数字真实，但仅在该语义下成立。**去掉 use_gate 分支后 faithful 与 G 数字对齐**（此前 122µs 的差异来自我 faithful 变体里的运行时分支，这同时证明分支代价 ~12–2×（faithful 口径下差 2×：122 vs 61µs——分支阻碍了 CSE 后的指令重排））。

### 判决表 C — e2e wiring 验证（单轮，island preset，graphs-on）

| 臂 | 开关 | decode tok/s | predicted_ms（512 窗） |
|---|---|---|---|
| 基线（moerework-off/off-r2，同 pin：BF16=1, MOE_REWORK=0） | IK_FUSED=0 | 91.98 / 91.83 | 5555.4 / 5564.9 |
| **ikfused-arm2**（BF16=1, MOE_REWORK=0） | **IK_FUSED=2** | **88.76** | **5756.9（+201.5ms）** |

- parity：`manifests/ikfused-arm2/c1-p8192.json.parity.json` → **matched: true**（512 token 逐位，参考 speed-bundle-single）。
- 命中形状直方图（`logs/ikfused-arm2.log`，567 次）：`K=2560 N=192 nc5×208 / N=224 nc5×103 / N=256 nc5×82`、nc2 系列 ×97——与 N224 口径的表 A 数字对应（0.59–0.77×），e2e −3.6% 与 kernel 级回归 × gu+gate 池占比（~23%）自洽。
- **生产 gu 形状实证：N=192/224/256（非阶段一的 N640）**，阶段一收益池折算所用的形状本身就偏离了部署形状。

## 与阶段一报告（REPORT.md）的出入（勘误，供战役记录）

1. **收益归因**：报告 §2.2 将加速归因于"单 warp 一次解码向全部 token 的 y 投影"的结构——实测该"一次解码"是编译器对 j 不变 x 基址的 CSE，**不是显式代码**；x 基址 per-token 化（生产语义所迫）后 CSE 消失，结构本身反而负收益（1-warp 块在 sm_70 上 32 blocks/SM 上限 → 32 warps/SM，只有 fork (32,nc) 块 64 warps 的一半占用）。
2. **ik 生产路径事实的推论未向下传递**：报告 §1.2/§6 已记录"ik 生产 host 只在 nc==1 用此核"，但未把"nc==1 专用 = channel-uniform 语义假设"与 moe-rework 流量勘误（per-token distinct）连起来做 distinct 口径微基准。阶段二若加一组 distinct ids 用例，负判决当天即可得出，省去本轮移植成本。
3. **收益池折算**（§3 的 265ms/+4-5%）建立在上述 uniform 口径上，**作废**。down_exps 同构收益的"附带机会"同理不成立（down 也是 per-token 路由；Q2_0 down 的胜出核是 moe-rework 的 staging 重构，与本结构无关）。
4. 报告 §4 估的"1-2 天"实际半天完成（结构清晰、依赖面小，报告这一点是准的）。

## 未采纳 / 已知边界

- **不删除树上代码**：kernel + 接线保留、默认 0、头部注释写明判决。理由：已验证逐位正确 + 一键回退，若未来模型结构变为 channel-uniform 路由（如 soft-MoE 全专家加权、或 nc==1 主路径想借用该结构——注意 nc==1 现行走 dense 路径不经此挂点），开关即用。代价仅 70 个模板实例化的编译时间。
- **IQ2_S 与 iq2s_volta 的关系维持现状**：q8_out 武装时让位逻辑保证 down 链 cache 不受影响；未做报告 §4 建议的"chooser 择优摘除"（无收益可摘）。
- **ksigns64 sign 表顺带移植（报告 §4 可选项）未做**：本核判决为负，不值得在 vecdotq.cuh 上加共享风险面。若未来单独做，独立可测，与本开关无耦合。
- **branchless 化未回灌树**：12% 的分支代价发现记录在案；默认关的核不再投入。
- **A/B 4 臂 + 三开关组合轮取消**（理由见 TL;DR）。若协调者仍要求正式 A/B 记录，脚本已就绪：`scripts/launch_ikfused_arm.sh`（BF16=1/MOE_REWORK=0 钉死）+ `scripts/run_ikfused_check.sh`（改 cases 即可）。

## 回退方法

- **运行时（首选）**：不设 `GGML_IK_FUSED_MOE`（默认 0 即原路径）；运行中一键 `=0`。
- **文件级**：`cp experiment/ikfused-integration/mmvq.cu.pre-ikfused sources/llama-v100-v3/ggml/src/ggml-cuda/mmvq.cu && rm sources/llama-v100-v3/ggml/src/ggml-cuda/kernel_ik_fused_moe.cuh && ninja -C build-ac922 llama-server llama-cli`。
- 全树回退：`experiment/ikfused-integration/pre_ikfused_baseline.diff`。

## 工件路径

- 服务器：`/home/debian/ac922-llama-next/experiment/ikfused-integration/`（patch、pre-ikfused、kernel 副本、基线 diff、4 份 bench 源、run_smoke.log、run_faithful2_excerpt.log、ikfused-arm2.log（8.5MB 含 NCCL 噪声）、driver log）
- 服务器：`manifests/ikfused-arm2/`（c1-p8192.json + parity.json + telemetry）、`logs/ikfused-arm2.log`
- 服务器：`/home/debian/ik-eval/bench-fork/bench_ikfused_{smoke,faithful,faithful2,ported2}.cu` + 二进制（编译命令在各 .cu 头部注释）
- 服务器：`scripts/launch_ikfused_arm.sh`、`scripts/run_ikfused_check.sh`
- 本地：`ik-eval/integration-artifacts/`（上述核心工件副本：kernel、patch、pre-ikfused、4 bench 源、3 份 log）
- 本地：`ik-eval/REPORT.md`（阶段一）、`ik-eval/INTEGRATION.md`（本文件）

## GPU 状态

验证轮结束即停服；最终确认 6×0 MiB、无 llama-server 残留进程、systemd unit inactive。

## 残留事项

- 若后续有人想复活该方向，唯一可行路径是**把 per-token distinct 路由变成可共享的形式**（如 host 侧按专家重排 token 分桶后多 launch，或 kernel 内 warp 按 (slot, expert) 分组）——前者 launch 数 ×10 不现实，后者等于重写 moe-rework 式专用核，均超出"结构移植"范畴。
- 阶段一报告的 §3 收益池与 §4 建议已由本报告勘误；后续战役引用 ik-eval 数据时以本文件为准。
