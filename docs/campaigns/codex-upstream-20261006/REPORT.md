# AC922：上游回移与高潜力组合

日期：2026-10-06。状态：本轮组合测试完成；层流水未通过整模型正确性门，已归档并回退。生产未切换。

## 结果摘要

本轮最强已验证组合：**明确 UUID 的真实 3+3 岛 + 岛内图中 P2P AllReduce + 四头 fused/tiled indexer + NextN reserve 修复 + 已修复的 R19 GPU MTP 链、resident embedding、FIFO、稳定 top-k、紧凑图缓存**。

- 8K 重复文本回归：prefill **920.56**、decode **125.06 tok/s**。
- 64K 重复文本回归：prefill **1003.40**、decode **108.81 tok/s**。
- 两档均为 512 个输出 token，与本轮冻结基准逐 token 一致。
- 固定真实代码/文字 8K：冷运行 **183.83 / 71.29**，同一输入热运行 **949.21 / 98.48**；两次 512-token 输出一致，接受率均 **367/573 = 64.05%**。
- 此组合采样到的最小单卡空余 **2281 MiB**，最高 GPU 和主机温度传感器均 **46°C**。显存/温度每 5 秒采样，不代表捕捉了所有瞬时峰值。

真实语料与重复文本必须分开看；重复文本 408/408 的接受率不能代表日常使用。真实语料冷/热请求均重做了 prefill（cache_n=0）；热运行仍受 CPU PLE 文件页缓存和图缓存影响。尚未测试多并发、128K/262K 实际长输入，也没有部署生产。

## 同轮组合比较

数值均为单流、512 输出 token，格式 PP/TG（tok/s）。对照为本轮冻结的原二进制、原配置，不能把它与历史 749.79/100.04 或 955–1055 的其他缓存状态直接混算。

| 配置 | 8K PP/TG | 64K PP/TG | 512-token 回归 | 判定 |
|---|---:|---:|---|---|
| 冻结原配置 | 648.86 / 93.67 | 738.80 / 83.02 | 对照 | 保留回滚运行时 |
| 新 indexer + 真实岛/P2P，UB512 | 964.69 / 113.73 | 未跑 | 一致 | 短 prefill 候选 |
| 新 indexer + stock layer，UB1024 | 863.51 / 101.04 | 880.88 / 87.28 | 两档一致 | 可用备选；不优于岛组合 |
| 新 indexer + 真实岛/P2P，UB1024 | 913.12 / 114.21 | 1006.00 / 99.27 | 两档一致 | 被完整 R19 组合替代 |
| 上项 + R19 快路径 | **920.56 / 125.06** | **1003.40 / 108.81** | 两档一致 | 当前速度候选 |
| 262K 配置、Q8 K/Q4 V、UB512 | 881.43 / 113.92 | 987.49 / 99.20 | 两档一致 | 容量候选，非实际 262K 输入验收 |
| 262K、UB1024 | 无有效结果 | 无有效结果 | 未完成 | 首次 prefill 工作区 OOM，已退出 |
| 层流水 + R19 + 新 indexer，UB512 | 912.40 / 102.70 | 1176.37 / 88.46 | 两档一致 | **真实语料重复失败，已回退** |

完整 R19 组合相对同轮基准：8K PP **+41.87%**、TG **+33.51%**；64K PP **+35.82%**、TG **+31.06%**。这是组合收益，未逐项归因。

基准首档期间有 3 个 CPU 构建任务；未进行重复统计。上述增量是本轮这些有界请求的比较，不能作为独立部件的收益或所有负载的保证。

### 直接埋点的 graph_compute 调用时间

以下取该请求中 `rebuilt=0`、decode token 数不超过 6 的 step-timing 窗口中位数；窗口及原始日志在 results.json。**这是 host 调用计时，包含部分等待，不是 GPU 完成时间或完整 MTP 步墙钟。** 不同执行拓扑会把等待移动到调用之外，尤其 stock layer；因此没有用 tok/s 换算步时，也没有把这些数当作 GPU 内核时间。

| 配置 | 8K target 调用 ms | 64K target 调用 ms |
|---|---:|---:|
| 冻结原配置 | 22.686 | 24.032 |
| 岛/P2P + indexer，UB512 | 13.434 | 未跑 |
| 岛/P2P + indexer，UB1024 | 14.554 | 14.398 |
| 上项 + R19 | 14.603 | 14.371 |
| 262K、UB512 | 13.440 | 14.177 |
| stock layer、UB1024 | 2.978 | 2.975 |

## 关键纠正：旧 GPU 顺序把岛混在一起

在 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 下，旧可见顺序 `0,4,2,1,3,5` 的逻辑前三卡为 `0004:04 / 0035:04 / 0004:06`，后三卡也跨岛。Meta 后端按可见设备顺序每三张分组，不会修正它。

正确顺序用 GPU UUID 固定，对应物理 PCI：

| 逻辑岛 | PCI 总线 |
|---|---|
| A | 0004:04:00.0、0004:05:00.0、0004:06:00.0 |
| B | 0035:03:00.0、0035:04:00.0、0035:05:00.0 |

CUDA runtime peer 查询确认同岛成员可互访、跨岛不可互访。新日志实际出现两岛的 `in-graph P2P AllReduce on 3 devices` / `using in-graph P2P AllReduce`。因此此前在混合分组上“P2P 环境变量没有收益”的结论不适用于真实岛分组。未重新做 nsys；不能给出新的 NCCL 时间占比或宣称所有同步消失。

## 上游吸收与验证

基底为 `7fe450e19305b828c199d602c23a8337aaa1f03b` 加备份时的 79 个 tracked 修改；没有整树 merge、切分支、stash 或 reset。查询到的上游 HEAD 为 `6c73b3e12dc501de35fe5f6979960d06921a2f6c`。

| 上游项 | 本机处理 | 验证 |
|---|---|---|
| [#30020](https://github.com/ggml-org/llama.cpp/pull/30020)，`1a3011cc` | NextN flag 真正变化时设 sched_need_reserve | 真实 tiny CPU context 的 setter/reserve 通过 |
| [#29825](https://github.com/ggml-org/llama.cpp/pull/29825)，`889edf43` | 选择最终融合 scorer/CUDA 四头支持，适配本 fork 原 block/cell bias；默认 legacy，env opt-in fused | 保留原 unit head 权重，未套用不同的缩放语义 |
| [#29901](https://github.com/ggml-org/llama.cpp/pull/29901)，`1b43d311` | SM70 可编译的 64-key × 8-token tile，短 batch vector 回退 | 9/9 GPU/CPU 数值 fixture：4 头、kv257、batch7/8/9、F32/F16/Q8_0；实跑 tile 激活日志 |
| [#29994](https://github.com/ggml-org/llama.cpp/pull/29994)，`43fe9c64` | 不强行移植不同版本的标准 k-pool；核对现有自定义分序列 pooled 布局 | 实际 CPU shared-row、stale invalidation、partial seq_cp fixture 通过 |

#29825 的 PR 初始说明描述逐头计算，但最终合并 diff 使用融合 LIGHTNING_INDEXER；本轮按最终代码适配，没有机械套用最初说明。也没有升级 ggml 0.26 或吸收 Metal/Vulkan 路径。

已经使用 SM70 路径：Volta GDN X4、SM70 BF16/GEMV/MoE 分支、QSA sparse，以及新的 indexer CUDA tile。`Onecat FATTN`、`PLE_ONECAT`、MonoMoE 仍关闭，未把旧失败项混入本组合。

## 真实语料的正确性边界

旧预设未启用 `GGML_CUDA_TOPK_STABLE`。原二进制的首次真实语料与另一台新实例上的同输入也产生不同输出；那两次请求历史不同，不能仅凭此证明所有情况下随机不确定，但它们确实不是稳定的逐 token oracle。

补齐 R19 的稳定 top-k 与 GPU MTP 链后，本组合的两次真实 8K 输出 512/512 相同，接受数也相同。新 scorer 改变了浮点计算顺序，GPU fixture 验证的是数值容差；**没有证明任意真实输入与旧 runtime bit-exact 等价**。未经稳定 top-k 的早期真实样本不参与严格逐 token 选优。

## 容量与失败记录

262144 是服务配置的 KV 上限；完成输入为 8K 和 64K。Q4 V 降低 KV 占用，但并未进行质量或满窗验收。该容量组合真实 8K 冷样本仅 183.49/51.04、356/617 接受，未达到速度组合的真实语料表现。

262K + UB1024 能加载并启动，首次请求扩展工作区时 device3 的 4824.31 MiB 分配 OOM；不能把 READY 当容量通过。UB512 两档完成，采样最小单卡空余约 1783 MiB；运行结束释放所有 GPU。

另有一次新 binary 启动 SIGSEGV：其 bin 中缺少后端库，动态加载把新的 libggml-base 与旧 `/usr/local/lib` CPU/CUDA 后端混用。通过实验目录内配套库链接修复，并保存实际运行映射和哈希；没有改系统库。

## 文件、复现与回滚

远端隔离根目录：`/home/debian/ac922-llama-next/experiment/codex-upstream-20261006`。原源码和原 runtime 未被本轮覆盖；原 model GGUF 未修改。

主要修改：

- `source/src/llama-context.cpp`：NextN reserve；层流水部分已回退。
- `source/src/models/qwen4exp.cpp`：融合 indexer 适配。
- `source/ggml/src/ggml-cuda/lightning-indexer.cu`：四头 vector/tile、一次性激活日志。
- `source/tests/test-backend-ops.cpp`：四头/tile 边界覆盖。
- `fixtures/pr29994-qsa-multi.cpp`：真实 CPU cache/reserve 检查。
- `scripts/run_arm.py`、`scripts/replay.py`：独占 transient user unit、GPU 空闲检查、58°C 与 512MiB 余量保护、超时、退出释放；新标签新日志。
- `backups/`、`artifacts/`、`logs/`、`CURRENT-STATE.md`、本报告及 results.json：冻结来源、运行配置、完整输出和诊断。

本轮增量构建只重编 indexer CUDA TU，并重用冻结的其余 CUDA 对象；C++ 通过私有 GGML prefix 编译。没有安装新依赖。原运行时存于 `baseline-runtime/`，获胜组合的配套二进制/库存于 `artifacts/bundle-v1-runtime/`，其 symlink 都指向目录内文件。

复跑速度候选（一次有界测试，完成后自动释放）：

```bash
ssh ac922 'python3 /home/debian/ac922-llama-next/experiment/codex-upstream-20261006/scripts/replay.py fast132'
```

容量候选把末尾 `fast132` 改为 `capacity262`。这不是生产服务部署命令。

所有测试仅使用自有 `ac922-codex-upstream-*` user unit 和 `127.0.0.1:18092`，finally 自动停止；也有 RuntimeMaxSec=1800 后备限时。回滚实验使用冻结 `baseline-runtime/llama-server` 及基准 configuration.json，或关闭各 opt-in；生产本轮未变，无需恢复生产文件。不要广泛停止其他 `ac922-*` 服务。

## 层流水工程与回退结论

实现了 host INPUT pinned snapshots、四个 scheduler copy 槽轮换、CUDA VIEW 的 `src/view_src/data` 原子重绑定、目的 copy 槽消费事件保护、512MiB pinned 上限及失败同步回退。保留零输入 split fence；跨 SYS 的复制仍可能走同步 fallback。这不是 per-phase 权重布局切换。

实际 GPU 小模型由父代理修正为 layer split `1:3`，使两个目标层分别位于 CUDA0/CUDA3；等分会把两个目标层都放在 CUDA0，不能当跨阶段验证。固定非重复 1024/1023-token 前缀、清缓存、127-token 尾批、无需手动 prefill 同步后转入输出；8×128 logits 的最大绝对误差 **1.164e-10**、argmax 一致、copy0/1/2/3 多次轮换，CUDA Graph 也实际启用。

整模型组合 8K/64K 重复文本均 512-token parity 通过，64K PP 达 1176.37。可是固定真实 8K 冷/热重复从 **第37个输出 token** 开始分岔。因此这个 1176 数字不能作为合格候选。

诊断只跑一次同样 layer/R19/fused/UB512、关闭新流水的真实 8K 两波：两次输出 **512/512 一致**，接受数均 350/644；冷 180.17/46.39，热 843.09/68.75。此诊断用新实例直接跑真实语料，开启流水时则先跑了 8K/64K 回归，请求历史不同；它支持优先检查流水与请求重置/缓存的交互，尚不足以唯一定位具体 kernel/缓冲依赖。不能用小模型通过替代真实模型门。

三处流水源码已恢复到修改前，bin 恢复到经过完整测试的 bundle-v1；hash 全部匹配。原始失败源码、补丁、配套 runtime 和当时的 harness 存于 `artifacts/rejected-stage-pipeline/`，不进入可用配置。当前 harness 已移除启动该路径的参数。保留失败记录用于下一轮修复；没有继续堆配置掩盖问题。

构建目录中间对象可能仍来自失败实验；恢复的 source/header 已更新时间，未来正常构建会重新编译。当前 bin 和 `artifacts/bundle-v1-runtime/` 一致，可直接使用经过测试的冻结运行时。

未进行：负载长压测、外网 API 验收、并发验收、满 262K 输入、生产推广。未操作 BMC、风扇、PSU、驱动、内核或电源。

## 多流源码核对（2026-10-06，未启动新 GPU 测试）

本轮所有性能请求都是单流。当前最快链不能仅靠增加 slot 直接推广：`common/speculative.cpp:1512` 的 MTP fold 要求 n_seq=1；`src/llama-context.cpp:1147` 的 GPU carry 配置拒绝 n_seq_max!=1；prefill/MTP FIFO 同样限定单序列。其余优化尚未完成新版多流数值与性能验收。

`common/speculative.cpp:1528` 的 GPU chain 初始化还要求 mtp_adaptive=false。直接开启现有 adaptive 设置会停用这条快链；动态深度应与多序列 token/hidden-state ownership、固定 verify padding、GPU argmax/feedback 一并实现，保留 CUDA Graph。

下一步优先：有界 2×8K/4×8K 实际请求，记录每路/合计速度和 TTFT；将 GPU MTP 链与固定图适配多序列；继续检查失败流水的缓冲及请求重置/缓存交互。CPU PLE 定向预取/有限热行缓存可用于改善冷 prefill；K8V4 与新版快链、并发状态的组合尚未验收。岛内 DWDP 式专家布局/预取是更大的独立工程。
