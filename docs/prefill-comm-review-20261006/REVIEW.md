# AC922 prefill 通信优化研究与实现建议

日期：2026-10-06。对象：六张 V100 的 Flash-Next llama.cpp fork。当前源码 HEAD 为 `7fe450e19305b828c199d602c23a8337aaa1f03b`，含正在变化的未提交改动。

最值得继续的是两条路线：近期完善现有 3+3 的双岛微批流水；长期实现逐层临时聚合专家权重，减少 prefill 的逐层归约。直接保留两份完整 GPU 权重布局会挤掉长上下文空间，六卡跨岛拉取专家权重也不适合这台机器的互联。以下收益与工期判断均为工程推断，不是新 benchmark 成绩。

## 原始 trace 的通信时间

对现有 SQLite 使用只读连接，按 kernel 时间区间并集重算。分母为切片首个 kernel 开始至末个 kernel 结束，不是完整请求时间。不同类别可以重叠，不能把 GPU 通信、计算和 CPU synchronize 时间直接相加。

| 指标 | 64K prefill 切片 | R17 8K decode 切片 |
|---|---:|---:|
| 时间窗口 | 12,030.355 ms | 5,171.827 ms |
| 全部 kernel 事件 | 263,523 | 777,762 |
| NCCL kernel 事件 | 5,580 | 606 |
| NCCL 时间并集 | 5,875.501 ms | 7.677 ms |
| NCCL 覆盖窗口比例 | 48.839% | 0.1484% |
| NCCL 各卡 kernel 时长之和 | 15,759.835 ms | 15.782 ms |
| 图内 P2P AR kernel 事件 | 0 | 32,028 |
| P2P AR 时间并集 | 0 | 663.467 ms，12.8285% |
| NCCL 时间簇，间隔阈值 20 ms | 26 | 4 |

因此，prefill 的约 49% 通信覆盖得到原始数据支持。R17 的 606 个残余 NCCL 核集中于四个窗口，稳态归约已由 P2P 核承担；不能用残余 NCCL 数量推断当前 prefill 的通信组成。四个窗口的跨度包含中间计算和空隙，不能当成 NCCL 执行时长。事件数量也是跨卡 kernel 实例数量，不是逻辑 collective 次数。

prefill 的 4,674 个 BF16 NCCL 核累计 15,138.497 ms，是主项；另有 906 个 F32 核。两岛按相同核名与时间序号配对后，collective 的 rank 开始时间平均相差 1.192 / 1.056 ms，P95 相差 3.649 / 3.590 ms。这个差值提示发射或上游计算失步，但不是纯链路传输时间，尚不能归因到专家路由。

同一 prefill 切片中，两岛都有 kernel 活跃的交集为 2,591.162 ms，占窗口 21.54%。这里包含 draft、通信和 target，因此不能据此宣布 target 流水已充分重叠，也不能把“单个 ubatch 依次过两岛”直接推成整个 prefill 只有半数 GPU 工作。

复算数据：[trace-metrics.json](trace-metrics.json)。脚本：[analyze_trace.py](analyze_trace.py)。

## 类似思想与可借用的实现

| 工作 | 可以借用的机制 | 对 AC922 的限制 |
|---|---|---|
| [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal)，[源码](https://github.com/microsoft/sarathi-serve) | 固定大小 chunked prefill、平衡微批、减小流水气泡 | 借调度方式；其现成运行时并未验证本机 GGUF、GDN、PLE 与 MTP 组合 |
| [VPP](https://arxiv.org/html/2608.26523v1)，[论文给出的源码分支](https://github.com/RookieCoder-Camera/vllm-ascend/tree/vpp-dev) | 固定 chunk、虚拟 stage、异步 send/recv、双队列在途 batch | 实现平台是 Ascend/HCCL；复用状态机和事件依赖设计，不直接搬算子。分支 README 可读，具体调度源码未逐项审计 |
| [DistServe](https://github.com/LLMServe/DistServe/blob/main/distserve/config.py) | prefill 与 decode 独立 TP/PP 配置，KV 迁移 | `DisaggParallelConfig` 明确保存两套配置，GPU 数为两阶段 world size 之和；它不能替现有 llama_model 自动共享不同物理布局 |
| [MoE-Prefill](https://arxiv.org/html/2605.02960v2) | 稠密权重副本、当前层完整专家集、下一层权重异步 AllGather、层入口 event wait | 原名 ZeRO-Prefill，v2 改名。需要计算窗口覆盖传输；论文侧重 prefill-only，本机还须保留 decode 所需状态，不能采用其丢弃 KV 的模式 |
| [EPS-MoE](https://arxiv.org/html/2410.12247v2) | 专家分组计算与传输重叠，根据负载选择 GEMM 路径并控制资源争用 | 算法调度可借；本机七类 GGUF 专家格式及 SM70 需自己的实现，不能移植 H800 性能数字 |
| [PipeInfer](https://github.com/AutonomicPerfectionist/PipeInfer)，[作者说明](https://github.com/ggml-org/llama.cpp/discussions/10344) | 基于 llama.cpp 的异步投机微批、事务式 KV 提交与取消 | 老 MPI/backend 接口；作者注明 GPU 分支存在首尾层回 CPU 的缺陷，适合借事务设计，不能直接当替代引擎 |
| [Speculative Pipeline Decoding](https://github.com/yuyijiong/speculative_pipeline_decoding) | 跨深度隐藏状态驱动的流水 draft，多个 token 位于不同 stage | 使用需训练的专用 Pipeline Draft Module；不是现有 MTP4 的免费开关，单进程理论计时也不等于多卡墙钟性能 |
| [CPU GPU hybrid MoE 研究](https://arxiv.org/html/2606.10493v1) | 流加载 prefill、SmallEP、共享权重、双 batch 的 attention/MoE 重叠 | 是布局和执行方式的参考；AVX-512、现代消费 GPU 路径不直接适用 POWER9/V100 |

论文或外部引擎的加速倍数均不作为这台 AC922 的预期成绩。

## 方案 A 的当前状态

16:48 的源码快照显示，解析器允许六个 singleton，但 `load_tensors()` 仍只有两段中点分配，后四岛会触发零 block 保护。调查期间并行实验方已修改此处；16:51 复读确认当前工作树出现 `n_devices()>2` 分支，按 `(i+1)/n_devices()` 的累计边界均分 GPU blocks。旧文件 SHA256 为 `b2764ef4...`，新文件为 `b52a8676...`。这项加载障碍应标为“源码已修，加载及性能待实验结果”。

六段层放置与六段在途流水仍是两个实现问题。内建 scheduler 的 `pipeline_parallel` 要求 `split_mode==LAYER`；singleton meta islands 仍是 TENSOR 模式，不自动走该通用 graph-copy 路径。工作树的 FIFO 特例只覆盖两个 meta devices，3->3 async bridge 也只覆盖两岛。六段探针即使加载成功，也可能仍逐 ubatch 串行经过六卡。

应分别验收各卡拥有非零 blocks、无逐层 AR，以及不同 ubatch 的 stage 真正重叠。仅 PP tok/s 变化不能证明三项同时完成。尾部 MTP 与输出权重、各层 QSA/GDN 工作量会影响 stage 平衡；MTP4 是投机深度，不能据此假定有四个 draft 层。

源码位置：`src/llama.cpp:181-224`；当前 `src/llama-model.cpp:1747-1765`；`src/llama-context.cpp:564-569`。最初快照见 [source-snapshot.json](source-snapshot.json)，修复后快照见 [loader-update.json](loader-update.json)。

## 方案 B 的可行性与较小替代方案

现有 `llama_model` 的 `dev_layer`、`dev_output` 和 tensor buffer 在加载时固定。两个 context 可以共享同一模型的权重，但不能各自把同一个权重改成不同物理分片。第二个 context 还会增加 KV、recurrent state、工作区和图实例；它本身不提供阶段切换。

若让两份完整 GPU 布局同时常驻，按原稿 46.4 GiB 的口径仅权重即 92.8 GiB，接近六卡 ECC 后总量 92.82 GiB。原稿的 GB/GiB 未完全统一，但即便按十进制 GB 计算，也不能据此保证长上下文与工作区能够容纳。全模型重载则把请求级权重传输加入 TTFT，可能还会中断已有 decode。

更小的路线是复用现有 3+3 放置，仅让 prefill 的 stage0(frame i+1) 与 stage1(frame i) 重叠：

```text
时间片          0       1       2       3
岛0            A0      A1      A2      A3
岛1                    B0      B1      B2
同一微批顺序    A0 -> B0，A1 -> B1，A2 -> B2
```

工作树已有双帧输入快照和 pinned-host async bridge，具备这条路线的部分基础。FIFO gate 位于 `llama-context.cpp:1734-1740,1821-1823`；frame 输入快照位于 `ggml-backend.cpp:1673-1705`；bridge 特例位于 `:1794-1804`；3->3 异步复制 gate 位于 `ggml-backend-meta.cpp:2694-2724,2754-2817`。后者不满足条件时会回到同步复制，不能只看环境变量就宣布重叠。

下一步先利用现有实验工件确认 `async MIRRORED island copy active` 和 timeline。若未充分重叠，补 per-frame activation scratch、stage 队列和 event 依赖，保留每层同序列 GDN/KV 更新顺序，最后输出前 drain。这条路线不需第二份完整权重，也不需转换 KV 布局。原型属于数天级工作，完整验收可能一周以上，取决于 buffer 复用与 state 顺序问题。

真正改变 prefill/decode 布局时，除了 KV 还需处理 GDN recurrent/convolution state、QSA pooled/block cache、序列位置/掩码、MTP hidden-state 与 draft 状态。只迁移 K/V 不够。完整双执行器和状态转换是按周估算的改造。

## 方案 C 的加载能力与通信代价

meta allocator 能表达逐张量 MIRRORED、axis0、axis1、axis2 放置，因此稠密 replicate + 完整专家按 axis2 分片的物理布局可以编码。但执行器尚无对应 EP 路由语义：当前 `MUL_MAT_ID` 与普通 matmul 共用处理逻辑，不能仅凭 axis2 分片正确处理全局 expert ID、local ID、路由槽、mask 和跨卡输出归并。

当前 `_exps` gate/up 按 axis1、down 按 axis0 分片，专家 axis2 在每卡保留完整。三 rank 消费相同 route IDs；`GGML_CUDA_MOE_GROUPED` 负责单 rank 的量化 grouped GEMM，不是六卡专家并行或远端专家调度。因此当前报告中“rank 持有连续 expert ID 块，按热度重排即可削峰”的失衡前提不成立。

实现真正 EP 后，可选择发送激活给专家 owner，或把专家权重拉到消费者。两者都仍需数据依赖、事件等待和最终结果归并。任意热度装盒也不能只改 split 长度表达：现有 loader 拷贝连续区间，需要权重重排加 route-ID 置换，或新增逐专家 owner map。

稀疏性要按一个 ubatch 的专家并集估算。假设 token 间路由独立、每 token 在 512 个专家中均匀选 10 个，则

`E[unique experts] = 512 * (1 - (502/512)^T)`。

| token 数 T | 专家并集期望 | 全部专家比例 |
|---|---:|---:|
| 1 | 10.0 | 1.95% |
| 5 | 48.1 | 9.39% |
| 32 | 239.6 | 46.80% |
| 128 | 471.0 | 91.99% |
| 256 | 508.7 | 99.36% |
| 512 | 512.0 | 100.00% |

真实路由有相关性，需实际统计；这个模型只说明不能把单 token 的 10/512 热集直接外推到 prefill。按原稿 GPU 权重 46.4、稠密 2.6 的同单位估算，专家总量约 43.8，单 token 触及约 0.855；若消费者只有本地 1/6，每 token 缺失约 0.713。达到 94 tok/s 时对应约 67 的同单位每秒权重读取需求，还未计缓存复用、draft 和 workspace。

实时拓扑查询确认 GPU0/1/2 与 GPU3/4/5 各为 NV2/peer OK，跨岛为 SYS/peer CNS。不能让 kernel 像岛内一样直接读另一岛 GPU 权重；需要 NCCL 或 host-staged 中继。六路全模型专家分片会把大权重流量不断带过这条边界。

Strata 的 `PeerExperts` 值得借的是专家缓存、任务生命周期与 staging：它先将专家放进 peer 的本地 cache，forward 用该 peer 自己的 weight pointer，激活经 pinned-host staging，输出通过 host rows 回写；每层仍有完成等待。这不等价于六卡无同步 P2P 权重读取。源码位置：本地 `opensource/strata-engine/src/core/peer_experts.cpp:134-178,181-265`。

## 推荐的大项是逐层临时聚合

保留一份静态 GPU 权重，prefill 在岛内临时拼出当前层或当前专家组的完整权重，下一层复制使用独立 stream 与双缓冲。以较大的 layer-major slab 复用一次聚合，不要每个很小 ubatch 重拉一整层。先局部验证一个 MoE 层：临时 packed 权重、七种量化格式、路由输出、峰值显存及未隐藏的等待。

这借鉴 MoE-Prefill 的 AsyncEP，但完整双岛方案需要额外设计。优先岛内聚合，避免逐层传全套跨岛专家权重。当前层聚合只需局部 staging，不必常驻第二份全模型；然而它会改变算子分片和计算次序，仍需数值 fixture、既有 512-ID parity 及实际吞吐门。

若把一个长序列的 token 分给多卡独立做 dense/attention，仍须正确处理此前 KV 与 GDN 顺序依赖；不能把它当成互不依赖的 DP 请求。应先保留现有 QSA/GDN 状态布局，只局部替换 MoE，之后再决定是否扩大到完整 per-phase 执行器。工程规模明显大于双岛 FIFO 收尾，完整支持按周估算。

## 方案 D 与 E 的具体门槛

P2P_AR 可以覆盖大 prefill 的 F32 partial，不能因 trace 显示 BF16 NCCL 就判它不适用：NCCL 路径本来会将大 F32 partial 临时压成 BF16，P2P 路径保持 F32。没有 token 数硬拒绝门，但合并图要求所有归约 partial 支持 F32 contiguous；任一不符可使整图回退。

`GGML_CUDA_P2P_AR_DISABLE` 是存在即禁用，必须 unset。三卡组 reserve 每卡约 `2*n*max_partial_bytes = 6*max_partial_bytes`：512x2560 的 F32 partial 为 5 MiB，对应约 30 MiB/卡；若 partial 宽度为 10240，则约 120 MiB/卡。开启同时改变通信精度、流量与图调度，应记录真实 P2P 核、fallback 和峰值显存，并单独过 parity。

方案 E 不能只把 AllReduce 扔到 side stream 后立即等待，也不能跨不同层盲目合桶：下一层通常依赖当前归约结果。有效重叠来自另一微批或可独立的专家组；分桶还要保证 norm、QSA、GDN 等依赖满足完整输入要求。EPS-MoE 的专家组粒度与资源配额值得参考。49% 是 NCCL 覆盖时间，含等待且与计算重叠，不能直接预测藏掉后必得 1.8x 或将其与六卡利用率的倍数相乘。

## 建议执行顺序

1. 收现有七臂结果，核对六段加载修复、P2P 的真实启用及各卡显存。A 的失败要区分 loader、数值、无流水和性能问题。
2. 沿现有 3+3 路线确认并完善双帧 target prefill 重叠；这最容易保留 decode 与 KV 布局。
3. 做一个 MoE 层的岛内临时聚合原型，验证“聚合一次，计算多个 token”的传输摊销，再决定扩大。
4. 只有上述数据支持时，才投入完整 EP 或不同阶段布局的执行器。异步单流流水投机作为后续独立项目，需要解决 draft 依赖和回滚。

本轮新增的是这份评审、只读 trace 分析脚本、JSON 指标与源码快照；没有启动 benchmark、修改运行服务或写入服务器推理源码。正在变化的工作树与未完成的七臂结果，需要按后续实际工件继续裁决。
