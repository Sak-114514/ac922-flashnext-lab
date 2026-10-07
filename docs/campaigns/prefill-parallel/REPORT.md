# Prefill 通信税攻坚：并行模式实测报告（方向 6a / 6a' / 6b / 6c）

日期：2026-10-06 | 主机：ac922 (6×V100-SXM2-16GB, NVLink1 mesh, CUDA 12.4 / NCCL 2.27.7) | fork：`/home/debian/ac922-llama-next/sources/llama-v100-v3` @ 生产分支 ac922-flashnext-v100（冻结）
口径：1×8K 与 1×64K 单流、512 token greedy decode、MTP d4、生产配方基线；每臂背靠背同会话执行；8K 512-ID parity 全臂通过（红线保持）

## TL;DR

1. **6a（split-mode row）：判死**——fork 的 CUDA 后端未实现 split-buffer，`--split-mode row` 加载即抛 `device CUDA0 does not support split buffers`（静态+实证）。fork 的 `tensor` 模式（岛内 meta TP）本身就是更强的"行/列混合切+逐层归约"。
2. **6b（P2P AR / NCCL 协议 / ubatch）：64K 主战场全军持平或负**——p2par +0.03%、nccl-simple -0.8%、ub1024 -5.2%、ll128 -19.3%。**64K prefill 的通信税不是协议/发射/压缩问题，是岛内失步（skew）+ 岛间串行结构问题**，零代码旋钮拧不动。
3. **6a'（六段层流水）：stock `--split-mode layer` 是全场唯一大幅赢家——8K PP +14.6%（743.77）且 TG +5.5%，64K PP 持平（-1.3%）而 TG 仍 +2.7%，parity 全匹配、MTP 正常**。但 64K 门槛（≥+5%）未过，且 meta 路径的等价变体（6 单卡岛）双负——路经实现差异极大。
4. **判决**：无任何臂过 64K ≥+5% 门。**记录 smlayer 为强候选（8K 场景）**；64K 的真正出路是 6c 的**跨 ubatch 流水调度**（层流水 + ubatch n+1 提前入段，理论 ~2×，见 §5 修订版设计）——smlayer 的数据第一次实证了这条路的 decode 侧可行性（TG 不降反升）。

---

## 1. 病根量化（qsa-longctx/REPORT.md §1.3 的 64K nsys + 本轮 sqlite 复挖）

prefill 12s 切片（6 卡合计）：

| 证据 | 数字 | 含义 |
|---|---|---|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 4674 实例 / 15138ms | profile 时 prefill 主力 AR：bf16+Ring+LL |
| `ncclDevKernel_AllReduce_Sum_f32_RING_LL` | 906 实例 / 621ms | 小张量 f32 AR |
| bf16 LL AR 单实例时长 | 主体 2.2–3.2ms，尾部至 7.2ms | — |
| 线时间估算 | ~0.07–0.2ms | 512 tok × 2560 hidden × bf16 ≈ 2.6MB；Ring 3-rank 每卡流量 ≈ 3.5MB；2 通道 × 25GB/s |
| **AR 时长中等待占比** | **≈90%+** | NCCL LL/SIMPLE kernel 在 peer 到齐前自旋；等待=岛内三卡失步（MoE 专家按 rank 切分 → 每层每 ubatch 负载天然不均） |
| 今日调优表（NCCL_DEBUG=TUNING） | 2.6MB → RING/**SIMPLE**，仅 24-120B 走 LL | 10-04 profile 时的 LL 选择未复现（调优器决策漂移）；今日口径下协议已近最优，与 6b 实测一致 |

**结论：通信税的本体不是带宽/协议，是同步暴露（skew）与岛间串行结构（单流任一时刻仅 3 卡活跃）。**

## 2. 方向 6a：split-mode row 的语义结论与兼容性（零代码，判死）

### 2.1 本 fork 的 split-mode 语义（读码结论）

| 模式 | 机制 | 状态 |
|---|---|---|
| `layer` | stock 层流水（经典 pipeline，无 meta） | **可跑且实测强势**（见 §2bis） |
| `row` | stock llama.cpp 的 **split-buffer 机制**：权重按行切（2D 张量 ne1 维）、逐层归约；实现在后端 proc address `ggml_backend_split_buffer_type`（src/llama-model.cpp:1291 `make_gpu_buft_list`） | **CUDA 后端未实现**（grep 证实仅 ggml-sycl / ggml-hexagon 导出该 proc address；`ggml_backend_cuda_reg_get_proc_address` 无此项）→ 加载即 throw |
| `tensor`（生产） | **meta 后端岛内 TP**：按张量名 regex 决定切轴（src/llama-model.cpp:425 `llama_meta_device_get_split_state`）——attn_q/kv 头切、attn_output/ffn_down 输入维切（部分和）、ffn_up/gate/共享专家 输出维切、PLE/SSM 头切；逐层 AllReduce 收部分和 | 生产路径 |

vLLM 语义对照（任务书问）：vLLM row-parallel=权重按输出行切+AR；本 fork meta TP 的 attn_output/ffn_down 正是"输入维切+部分和+AR"（row-parallel 的伴生半边），q/kv/up/gate 是 column-parallel 半边。**fork 的 tensor 模式已覆盖 stock row 模式想做的事，且带 PLE/MTP/岛结构适配。**

### 2.2 实证（2026-10-06，单卡即复现，未占生产卡）

```
$ llama-server --split-mode row --tensor-split 1,1,1,1,1,1 ...
E llama_model_load: error loading model: device CUDA0 does not support split buffers
```

（`--fit` 自动重试一次同样失败后退出。）

### 2.3 判决

- row vs tensor 的 A/B：**不可能跑**（加载即失败，非配置可绕）。恢复 stock row 语义需补 CUDA split-buffer（源码工作，价值为负）。
- 6a-3（批维数据并行：每卡吃 1/3 prompt token、只在输出处归约）→ 本质是 micro-batch 流水，归入 6c（§5，且被 6b/6a' 数据重新定向）。

## 2bis. 方向 6a'：六段层流水（架构级探针）

### 2bis.1 静态验证

用户设想"总参分存、活跃参数+单路 KV 由 3 卡处理、并发 3+3 各服务一路"：纯形态显存否决（46.4GB/3卡=15.5GB/卡无 KV 空间）。衍生形态验证：

| 路径 | 支持 | 依据/实测 |
|---|---|---|
| `LLAMA_ISLAND_SPLIT=1,1,1,1,1,1`（meta 路径） | **需 2 个小补丁** | 解析器（src/llama.cpp:181）只要求等尺寸+总和=卡数，6×1 合法；但 `load_tensors` 岛间层分配是 **2 岛硬编码**（`splits[0]=中点、splits[i>0]=1.0`）→ 6 岛时岛 2-5 必空、被 owns-no-blocks 不变式拒绝（实测报错）；且岛成员数=1 时 split-state 回调仍返回切轴 → meta 断言 `src_ss[0].axis != AXIS_0` 失败。已在分支 `prefill-6island` 打 2 个小补丁（>2 岛均匀层边界 + 单成员岛全 MIRRORED），存档 `experiment/prefill-parallel/6island-layer-split.patch`（含两层），独立构建 `build-ac922-6isl/bin/llama-server`；**源码树已还原**（分支 ac922-flashnext-v100、134 脏文件与会话开始一致） |
| `--split-mode layer`（stock 管线，绕过 meta） | **直接可跑** | 生产 env 下 PLE_SHARD/DETP 本就关闭；MTP/kv-unified/QSA 全部正常工作（parity + 正常吞吐证实） |
| per-phase 混合（prefill 层流水 / decode 岛 TP） | **不支持** | split_mode 在 load_tensors 固定权重放置与 KV 布局，运行期无切换路径（读码确认） |

### 2bis.2 理论与实测

理论：岛 3,3 单流任一时刻仅 3 卡活跃（两岛串行交替）；六段流水稳态可 6 卡全开 → **若调度器跨 ubatch 流水，吞吐上界 ~2×**；实际 llama.cpp 调度器（stock 与 meta 两路径）都不做跨 ubatch 段间流水 → 实测大打折扣（见下表）。decode 意外不塌：层流水去掉了全部 AR，MTP draft 正常穿 6 段，TG 反而 +2.7~5.5%。

### 2bis.3 实测（含 §3 全臂总表）

见 §3.2 总表（smlayer / layerpipe6 两行）。

## 3. 方向 6b + 全臂实测总表

### 3.1 机制回顾（读码）

- `GGML_CUDA_P2P_AR_DISABLE`（存在即禁用）；`AC922_TEST_P2P_AR=1` → unset。开启后 meta 走**整岛图合并**：每子图边界插 `GGML_OP_ALLREDUCE`（图内 P2P 归约：peer 显存 staging+flag 自旋，f32 精确、无 bf16 压缩、无 host NCCL 调用），~512 节点分块、3 线程发射、可 CUDA-graph 重放；首见形状 defer。R17 判"噪声"是 1×8K decode 口径；64K prefill 本轮首测。
- NCCL 路径：大张量 to_bf16→AR→to_fp32（有损压缩），host ncclAllReduce group；小张量 f32。

### 3.2 全臂总表（同会话背靠背，基线=本会话 base 臂）

| 臂 | 增量 | 8K PP | Δ8K | 8K TG | 64K PP | Δ64K | 64K TG | 8K parity |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **base**（生产配方） | — | 648.84 | — | 92.93 | 746.45 | — | 83.07 | 参考 |
| **smlayer** | `--split-mode layer` | **743.77** | **+14.6%** | **98.25** | 737.01 | -1.3% | **85.29** | ✅ |
| p2par | `AC922_TEST_P2P_AR=1` | 664.75 | +2.5% | 93.10 | 746.71 | +0.0% | 83.33 | ✅ |
| nccl-simple | `NCCL_PROTO=SIMPLE` | 659.11 | +1.6% | 93.53 | 740.46 | -0.8% | 82.94 | ✅ |
| ub1024 | `AC922_TEST_UBATCH=1024` | 615.74 | -5.1% | 93.78 | 707.86 | -5.2% | 83.07 | ✅ |
| layerpipe6 | 6 单卡岛（分支补丁二进制） | 591.50 | -8.8% | 90.45 | 598.67 | -19.8% | 77.57 | ✅ |
| nccl-ll128 | `NCCL_PROTO=LL128` | 526.65 | -18.8% | 89.08 | 602.36 | -19.3% | 80.27 | ✅ |

判读：

1. **64K 主战场：所有通信旋钮持平或负**。p2par 746.71 vs base 746.45——P2P AR 消灭了 host 发射与 bf16 压缩，64K 却一分不涨 → 印证 AR 时长的 ~90% 是失步等待，不是发射/协议/压缩。8K 的 +2.5%/+1.6% 在 revival 实验的噪声带（同开关历史 +0.25~1.15%）边缘。
2. **smlayer 是唯一大幅信号**：8K PP +14.6%（消灭 AR 在注意力轻的短上下文全额兑现）、64K 持平（层流水失去注意力头并行 3×，长文注意力串行化恰好抵消 AR 收益——因调度器不做跨 ubatch 段间流水，见 §5）。**TG 两个口径都涨**（+5.5%/+2.7%）——decode 侧零 AR + MTP 穿段正常。
3. layerpipe6 双负：同样的层流水概念走 meta 单成员岛路径，meta 的 split-state 追踪/事件栅栏/异步桥开销 > 收益。**路径实现决定成败：stock 直通路径完胜 meta 包装路径。**
4. ll128 在 NVLink1 mesh 上灾难性（-19%），ub1024 计算侧损失大于通信侧收益。均判死。

### 3.3 判决（对照门禁：64K PP ≥+5% 记候选）

- **无臂过 64K 门**。p2par/nccl-simple 维持生产默认关闭/开启现状（p2par 与 base 差异在噪声内，与 revival 结论一致）。
- **smlayer 记录为强候选（未过 64K 门）**：8K +14.6% PP、双口径 TG 改善、parity 红线通过。适用面：短上下文（≤8-16K）单流场景。未见多流/QSA 长文/PLE CPU-offload 交互验证（后续若上生产需补 2×64K fair-share 口径）。

## 4. 方向 6c：通信/计算重叠（源码，未动手）

见 §5 设计文档（已按 6b/6a' 实测数据修订优先级）。

## 5. 6c 设计文档：prefill 专用重叠（按实测数据修订版）

### 5.0 前提与触发

触发条件原为"6a/6b 都不够 +10%"——**已确认触发**（6b 最好 +2.5%/8K，64K 全持平）。纪律：`git stash push -m prod-kernels` → `checkout -b prefill-overlap`；门禁 64K PP ≥+10%、8K parity 红线、TG 退化 ≤3%。

### 5.1 实测数据重排的优先级

| 原候选 | 实测后的新位置 |
|---|---|
| ~~C1 AR 桶化双流重叠~~ | 降级：p2par 持平证明发射/协议侧无油水；桶化只藏 wire 不藏 skew |
| C2 micro-batch token 流水 | **升级为 C2'：跨 ubatch 段间流水（新首选，见 5.2）**——smlayer 已证明层流水可跑、decode 不塌、parity 保持；缺的只是调度器让 ubatch n+1 提前入段 |
| C3 专家热度重排 | 保留为辅助（skew 仍在：64K AR 等待 ~90%；但层流水+跨 ubatch 流水直接绕开 AR，收益上界更大） |

### 5.2 C2' 设计：跨 ubatch 段间流水（prefill 专用调度）

- **洞察**：prefill ubatch 间残差流不跨 ubatch（ubatch n+1 的层 0 输入不依赖 ubatch n 的层 48 输出），只有 KV 因果依赖，而同段串行处理同层自动保序 → **ubatch n+1 可在 ubatch n 尚在第 k 段时进入第 0 段**，合法且零数值改动。
- 形态：`--split-mode layer`（或岛 1×6+补丁）+ 调度器允许 ≥2 个 ubatch 同时在飞（每段一个 in-flight ubatch，段内串行）。落点：`src/llama-context.cpp` 的 prefill ubatch 提交循环 + `ggml-backend.cpp`/meta 的段间事件改为"每段独立 fence"而非全图 fence。
- 上界（§1 数据）：岛 3,3 单流吞吐 = 49×((A+M)/3+r) 每 ubatch（3 卡活跃）；六段全流水 = (49/6)×(A+M)（6 卡活跃）→ **~2×**（r 为每层 AR+失步 ≈0.32(A+M)）。8K 口径 smlayer 已拿到的 +14.6% 是它的"无流水子集"。
- 风险：QSA 块选择跨 ubatch 的因果序（同段串行已保）；KV-unified 写序；MTP draft 验证步与 trunk prefill 的交错（draft 用独立图，段 fence 不共享则无冲突，需 fixture 验证 R17 的 257/15360/65536 模式）。
- 与现结构冲突：与 `LLAMA_PREFILL_FAIR_SHARE`（多流交错）共用 prefill 调度点，需联测；与岛 3,3 正交（层流水形态下岛概念退化为段）。

### 5.3 混合调度（prefill 层流水 / decode 岛 TP）的实测基础

静态不支持（§2bis.1）。但 smlayer 的 TG 数据（+2.7~5.5%）表明** decode 侧也许根本不需要岛 TP**——若 C2' 落地且 decode 保持层流水，混合切换的必要性存疑，可先测"层流水全相态"（简单）再考虑混合（复杂）。

## 6. 生产建议

1. **生产配方（64K 长文主力）维持现状不动**：岛 3,3 + tensor + NCCL auto。所有零代码臂在 64K 上无正收益；p2par 与 base 打平（且历史两轮也在噪声内），不值得引入图内自旋路径的常驻风险。
2. **短上下文场景候选（记录待批）**：`--split-mode layer`（stock）在 8K 单流 +14.6% PP / +5.5% TG、parity 通过。若业务有明确 ≤8-16K 单流负载，值得补测多流与真实语料后按场景启用。**与岛/tensor 模式互斥**（权重放置不同，同模型只能选一）。
3. **下一步立项（源码，走 6c C2'）**：跨 ubatch 段间流水是 64K prefill 唯一有 ~2× 量级理论上限的路径，且 decode 兼容性已被 smlayer 实测背书。分支 `prefill-6island` 的 2 个小补丁（6 段均匀层分配 + 单成员岛 MIRRORED）是它的第一块垫脚石——但注意 meta 路径当前比 stock 慢 8-20%，C2' 实现建议**直接在 stock layer 路径**做调度器改动，meta 单岛路径仅作对照。
4. 判死清单：NCCL_PROTO=LL128（-19%）、UBATCH=1024（-5%）、恢复 stock row split-buffer（无意义）、NCCL_MIN_NCHANNELS 调通道（调优表显示大 AR 已 2 通道 SIMPLE，且 proto 臂证明非瓶颈——不再立项）。

## 7. 复现材料（远程 ac922）

- 编排器：`~/ac922-llama-next/experiment/prefill-parallel/run_arms.py`（生产配方 env 逐行照抄 + 臂增量；独立 unit `ac922-flashnext-pp6`/port 8003；启动门=六卡显存全放；绝不触碰 K8V4 代理的 `ac922-flashnext-test` 单元）；`run_server_pp6.sh` / `run_server_pp6_layer.sh`
- 结果：`~/ac922-llama-next/manifests/prefill-parallel/{base,smlayer,p2par,nccl-simple,ub1024,nccl-ll128,layerpipe6}/`（state.json + c1-p8192/c1-p65536.json + parity）
- 分支与补丁：分支 `prefill-6island`（2 补丁），diff 存档 `~/ac922-llama-next/experiment/prefill-parallel/6island-layer-split.patch`，二进制 `sources/llama-v100-v3/build-ac922-6isl/bin/llama-server`；**源码树已还原生产态（134 脏文件、分支 ac922-flashnext-v100、stash 无新增）**
- row 实证：`/tmp/row_load_test.sh` 输出（§2.2）；nsys 复挖：`/tmp/q1.py /tmp/q2.py`
- 服务器日志：`~/ac922-llama-next/logs/prefill-parallel-<arm>.log`（含 NCCL_DEBUG=TUNING 调优表证据）

## 8. 会话收尾核查

- 源码树：分支 ac922-flashnext-v100，`git status --porcelain | wc -l` = 134（与会话开始一致）；llama-model.cpp 从备份还原；stash list 无新增
- 进程/单元：`ac922-flashnext-pp6` inactive；无 run_arms 残留进程；六卡显存全放（16150 MiB free/卡）
- K8V4 代理的 `ac922-flashnext-test` 单元未被本会话触碰（礼让期间其 kvtest 多臂正常交错运行）
