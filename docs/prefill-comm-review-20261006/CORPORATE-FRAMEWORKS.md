# 大厂多卡推理项目与 AC922 可移植机制

日期：2026-10-06。范围：官方项目、技术文档与部分源码。结论：大厂已经实现双微批重叠、专家权重分布、不同阶段并行配置和异步执行；对当前 AC922，最直接的参考是 RTP-LLM 的通信接口、TensorRT-LLM 的微批执行器，以及其已合入的 DWDP 专家权重预取。

## 项目选择

| 团队与项目 | 已公开机制 | 对当前任务的价值与限制 |
|---|---|---|
| DeepSeek [DeepEP](https://github.com/deepseek-ai/DeepEP) 与 [公开推理设计](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md) | DP attention、EP 专家、双微批交错、专家负载均衡 | 借 dispatch/compute/combine 的执行依赖。当前 DeepEP V2.5 要求 Hopper+、CUDA 13.1+，不能直接用在 SM70；它也不是本机已有的特征维 TP |
| 阿里 [RTP-LLM](https://github.com/alibaba/rtp-llm) | 生产推理引擎、TP/DP/EP、阶段分离与微批通信重叠 | 官方安装说明列出 [V100 支持](https://rtp-llm.ai/build/en/start/install.html)。完整 Flash-Next GGUF、PLE、GDN/QSA、七格式和 ppc64le 组合仍未得到可直接运行的证明 |
| 阿里 [DashInfer](https://github.com/modelscope/dash-infer) | C++ runtime、连续批处理、量化权重与 KV、SM70 CUDA 路径 | 可看 V100 适用算子。其 InstantQuant 的 IQ 不等于 GGUF IQ3_XXS；POWER9 与本模型需要额外适配 |
| NVIDIA [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) 与 [Dynamo](https://github.com/ai-dynamo/dynamo) | TP/PP 执行器、CPU/GPU overlap、阶段分离、KV 路由；DWDP 分布专家权重 | 最值得借执行器和预取生命周期。当前 TRT-LLM [支持列表](https://nvidia.github.io/TensorRT-LLM/supported-hardware.html) 不含 V100；Dynamo 是编排层，不会补 SM70 算子 |
| 字节 [Flux 与 COMET](https://github.com/bytedance/flux) | AllGather/GEMM、Grouped GEMM/ReduceScatter 的细粒度融合 | 直接针对计算通信串行。现成构建覆盖 SM80/89/90，移植 SM70 与 GGUF 需要重做相关计算和同步路径 |
| Microsoft 起源的 [DeepSpeed-MII/FastGen](https://github.com/deepspeedai/DeepSpeed-MII) | token 预算、连续批处理、Dynamic SplitFuse | 借调度，不能把混合请求批次当成层流水。预编译高性能 kernels 的兼容说明是 SM80+，不等于 DeepSpeed core 的 Volta 兼容性 |
| Moonshot [Mooncake](https://github.com/kvcache-ai/Mooncake) | KV 与权重的异步传输、拓扑感知、多层缓存和 PD handoff | 对跨 worker 状态交接与重复前缀复用有用；单独引入不能消除每层 TP AllReduce |

DeepSeek 的链接是其 2025 年公开生产设计，不能视为今天私有集群的精确配置。项目宣称用于生产、源码中存在某机制，以及该机制已在 AC922 跑通，是三个不同证据层级。

## 与方案 C 最接近的实际实现

NVIDIA [DWDP](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog19_DWDP_Distributed_Weight_Data_Parallelism_for_High_Performance_LLM_Inference_on_NVL72.html) 复制 attention 权重，分布存放专家；各 rank 用 peer 异步复制预取下一层专家，并用双缓冲覆盖传输。它已合入 TRT-LLM，在 GB200 NVL72 的 DeepSeek-R1 工作负载上报告端到端 TPS/GPU 提升 8.8%，不能外推成 V100 收益。

这为原方案 C 提供了实际实现先例。但 AC922 的 peer access 只存在于两个三卡岛内部，因此应研究“保留两岛按层切分，每岛只管理自己阶段的专家与稠密副本”，再在岛内采用 DWDP 式预取。六卡扁平组会重新遇到跨岛权重传输。

这会改变岛内执行布局：DWDP 当前要求 TP=1，不能继续把稠密计算当成原有 TP3；可延续的是两个三卡硬件域和跨岛层边界。其现成路径还依赖 NVFP4/CuteDSL 与相应 Blackwell 平台，并与 Overlap Scheduler 有组合限制。因此这是自定义岛内执行器的参考，不能与现有 flag 直接叠加。

迁移时要计算本阶段稠密副本、专家常驻、两个预取 buffer、完整或分片 KV/GDN/PLE state、MTP 与尾部输出头的逐卡峰值。48 层平均账不能替代末卡峰值账。对于单条长 prompt，多 rank 如何分担计算仍需独立调度，DP 副本不会自动把一个请求分成可独立执行的三份。

文档对应源码 [dwdp.py](https://github.com/NVIDIA/TensorRT-LLM/blob/be1248293e7e337dc4a7ff544038e40d36701e38/tensorrt_llm/_torch/pyexecutor/dwdp.py) 已保存固定版本：

- `DwdpLayerHandleCollector`：记录权重与 CUDA IPC handles。
- `DwdpPrefetchBuffer`：两个 buffer、专用 stream、prefetch/compute 完成事件。
- `DwdpManager.prefetch_layer()` 与 `wait_prefetch_and_get_buffer()`：控制预取与消费时机。

本地文件：[dwdp-be12482.py](references/trt-excerpts/dwdp-be12482.py)。哈希与来源：[manifest.json](references/trt-excerpts/manifest.json)。这里保存的是参考源码，未安装或执行它。

## 双微批是最直接可借的调度结构

阿里 [DeepSeek 重放报告](https://rtp-llm.ai/build/en/references/deepseek/reporter.html) 明确描述双微批交错、统一通信回调与不同 TP 规格的阶段分离，并指出微批增加 launch 和延迟，需要 CUDA Graph/MTP 配合。可借的是提交、完成与消费接口，以及利用另一微批计算覆盖通信的依赖安排；无需把本机立即改成六卡 EP。

[vLLM DBO](https://docs.vllm.ai/en/latest/design/dbo/) 也有明确源码结构：`UBatchWrapper` 管理线程和 CUDA Graph，`UBatchContext` 管理同步，MoE 计算中的 yield points 交错两个微批。官方当前 DBO 要求 DP+EP 与 DeepEP，因此 `--enable-dbo` 不是现有 TP3 llama.cpp 的直接开关。

TensorRT-LLM [Overlap Scheduler](https://nvidia.github.io/TensorRT-LLM/features/overlap-scheduler.html) 解决 CPU bookkeeping 与 GPU step 的重叠；PP 微批队列解决不同层段的重叠；P/D workers 解决不同阶段资源隔离。这三者应分别移植和验证。其 `tensorrt_llm/_torch/pyexecutor/py_executor.py` 中的 `BatchStatePP` 与 micro-batch 队列可作为两岛 frame 管理参考，但相关 PP loop 当前还限制 drafter/speculative decoding，不能直接套用 MTP4。

## RTP 源码中的较小改动参考

固定检查版本为 `95ce33f43822cf6cccf012f635e805a814899b94`。部分原来的 C++ devices 路径已重构，不能拿旧技术报告的路径当当前源码。

### 合并两路专家输出后归约

[`GenericMoeLayer`](https://github.com/alibaba/rtp-llm/blob/95ce33f43822cf6cccf012f635e805a814899b94/rtp_llm/models_py/model_desc/generic_moe.py) 在纯 TP、存在 shared expert、路由器支持延后归约且 TP 组一致时，将 routed 与 shared expert 的 partial output 先求和，再做一次 AllReduce。它有 gate、路由器能力和单次归约的 contract tests。

这是本机可先审计的小项：若当前图分别归约两路同组、同布局的线性 partial，可减少一次 collective；若已经合并，则没有新收益。门控的顺序和 BF16 wire rounding 可能改变数值，需要 fixture 和既有 token parity。不能只设置 skip 或跳过归约。

本地保存的 [generic_moe.py](references/rtp-excerpts/rtp_llm/models_py/model_desc/generic_moe.py) 与 [测试](references/rtp-excerpts/rtp_llm/models_py/model_desc/test/generic_moe_allreduce_test.py) 可以直接对照，未运行外部项目测试。

### 把等待推迟到真正需要远端数据的位置

[`PCPAllGatherOverlapAttnOp.forward`](https://github.com/alibaba/rtp-llm/blob/95ce33f43822cf6cccf012f635e805a814899b94/rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/prefill_mha/allgather_overlap_impl.py) 用独立 stream 聚合 K/V，同时算本地与已缓存 prefix attention，消费远端结果前才 wait。这种依赖拆分可以参考，QSA/GDN 需要自己的正确性设计，不能直接替换成普通 attention 路径。

本地：[allgather_overlap_impl.py](references/rtp-excerpts/rtp_llm/models_py/modules/factory/attention/cuda_cp_impl/prefill_mha/allgather_overlap_impl.py)。参考文件来源与哈希：[manifest.json](references/rtp-excerpts/manifest.json)。

## Flash Next 的独立模型实现参照

NVIDIA 已发布 [Flash-Next 部署指南](https://nvidia.github.io/TensorRT-LLM/deployment-guide/deployment-guide-for-qwen3.8-flash-next-on-trtllm.html)，覆盖 GDN、QSA、PLE 与 MTP。它强调 handoff 需要包含 recurrent state、QSA index、PLE n-gram 与短卷积状态；本模型的该路径要求两个 workers 使用同一 parallel layout。一般模型支持不同 TP/PP 的能力，不能直接外推到这一混合模型。

这份实现适合作为状态管理与 PLE 正确性的独立参照。指南的 GPU 和权重格式是 Hopper/Blackwell 与 BF16/FP8/NVFP4，不能作为当前六 V100 IQ3_XXS 的直接替代配方。

## 对 AC922 的研究优先级

1. 审计 routed/shared expert 输出是否还能合并归约；这是新发现、范围最小。
2. 参考 RTP 通信回调和 TRT 微批队列，完善现有 3+3 prefill frame 的异步桥和 buffer 生命周期。
3. 参考 DWDP 管理器，做岛内下一层专家预取与双缓冲局部原型，量化传输是否能隐藏，再决定改变布局。
4. 若计算通信重叠仍不足，再借 Flux 的 tile 级 ready/consume 机制实现 SM70 版本。

本轮没有安装框架、改运行服务或启动 GPU 测试。新增该文档、一个仅含元数据的 RTP 浅克隆，以及固定源码摘录；没有把框架存在、支持 V100 或源码机制当作本机速度验证。
