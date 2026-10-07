# 工业界多卡并行技术调研——可吸收清单（2026-10-06）

> 调研动机：AC922（6×V100-16GB NVLink 一代，POWER9）跑 Qwen3.8-Flash-Next（MoE 512 选 10 + GDN/QSA 混合）。
> 当前：单流 125.1/108.8 tok/s（8K/64K，Codex 新组合），多流 2×8K 仅 75 聚合（MTP 链单序列限制）。
> 调研原则：**不要求现成轮子，重点是源码思路/改造方案**；按"我们能搬什么"逐个判决。

---

## 1. DWDP（NVIDIA，NVL72，TensorRT-LLM PR #12136）——专家权重分布 + 数据并行保持

**原版做法**：NVL72 上不复制专家权重也不用宽 EP 的 all-to-all，而是**专家权重按 GPU 分片**，每卡仍独立处理自己的 token（数据并行），token 需要别卡专家时按需取。官方路径限 TP=1、NVFP4、CuteDSL。

**对我们的可吸收面（对应"变体 7"设计）**：
- **思路完全同构**：我们已有的分析（稠密层 2.6GB 副本 + 专家表 43.8GB 分片 = 9.9GB/卡）就是 DWDP 的 SM70/GGUF 翻译版
- **要抄的三个源码思想**（TRT-LLM 开源，Apache-2.0）：
  1. **专家权重页表 + 按需取的调度**：TRT-LLM DWDP 用"weight page"粒度管理分布的专家权重，取权重与计算重叠——我们抄成 GGUF blob 级 page
  2. **下一层异步预取**：DWDP 的 prefetcher 在第 N 层计算时预取第 N+1 层路由命中的专家页——我们的 fork 已有 RouterLookahead（decode 侧），扩到 prefill/权重页
  3. **容量账方法**：DWDP 论文的 expert shard size × hot set 分析——我们已做过（3.5GB 热集）
- **改造量**：大（加载器双布局 + 页表 + 预取器），但它同时解决：显存、通信税（P2P 异步取替代 AR）、负载均衡（哪卡热专家多就多放）
- **判决：长期架构旗舰，值得立项设计文档**

## 2. vLLM DBO（Dual-Batch Overlap，#51700 全图版）——微批乒乓藏通信

**原版做法**：batch 拆两个微批，微批 A 做 MoE all-to-all 通信时微批 B 计算，乒乓掩盖通信。v0.30 起微批步进 **FULL CUDA Graph** 捕获。

**对我们的可吸收面**：
- Codex 双微批实验失败的教训正是缺 vLLM 的两个关键件：**独立的 frame 描述** + **显式的生产/消费事件管理缓冲**（不是共享 buffer 的时序祈祷）
- **源码具体抄什么**：`vllm/v1/worker/gpu_model_runner.py` 的 micro-batch split（每个微批独立 input buffer + attention metadata）；#51700 的图捕获策略（微批形状固定化——正好治我们的 shape drift）
- **v100 适配**：vLLM 的 DBO 依赖 DeepEP（通信库），我们没 DeepEP——但我们藏的不是 all-to-all 而是**岛内 AR**，DBO 思想同样适用：微批 A 等归约时微批 B 在算
- **改造量**：中——fork 的 FIFO 流水已有雏形（输入快照流水），把它升级成"双微批 + 事件"是渐进改造不是重写
- **判决：中期最有价值的 prefill 方向，Codex 已列为优先级 2，抄 vLLM 源码的具体帧管理**

## 3. DeepEP（DeepSeek 开源，MIT）——低延迟专家 all-to-all 库

**原版做法**：两种模式——正常模式（RDMA 高吞吐，prefill/训练）+ 低延迟模式（hook 式纯 RDMA，双缓冲重叠通信计算，launch 延迟 ~7µs，NVLink 内 20-50GB/s）。

**对我们的可吸收面**：
- **双缓冲思想**（low-latency 模式的核心）：发送/接收 buffer 成对轮换，通信与计算天然重叠——这个 kernel 级模式可以直接搬进我们的 MoE grouped kernel（当前 GGML_CUDA_MOE_GROUPED 无重叠）
- **dispatch/combine 分离**：token 打包发去专家所在卡、算完收回来——这就是我们"变体 7"需要的通信原语，DeepEP 的 intranode NVLink 路径（不走 RDMA）源码就是 NVLink 一代 mesh 上的参考实现
- **判决：抄 kernel 思想（双缓冲 + dispatch/combine），不引入库依赖**

## 4. Mooncake（月之暗面/Kimi 开源）——KVCache 中心化分离架构

**原版做法**：prefill 集群与 decode 集群物理分离，KVCache 通过 Transfer Engine（RDMA/加速器原生路径）在集群间搬，prefix cache "以存储换计算"。

**对我们的可吸收面**：
- 我们不做集群分离（单机），但 **Transfer Engine 的"KV 分块传输 + 零拷贝注册"** 思想可用于：冷启动时 PLE 表/KV 从 SSD 到 GPU 的搬运（TTFT 攻坚的第 3 项）
- **Conductor 调度思想**（预fill/decode 分池调度）在我们的多流场景有微缩版对应：MTP draft（微 prefill）与 target（decode）的调度分离
- **判决：抄调度与传输思想，规模缩减到单机**

## 5. SGLang EPLB——专家负载均衡（直击我们 64K 通信税真根）

**原版做法**：统计专家激活 → 热专家**冗余复制**多份分散到各卡 → 定期 rebalance。华为 CloudMatrix384 论文实测每卡放 10 专家含冗余副本。

**对我们的可吸收面（这可能是最直接的药）**：
- prefill-parallel 战役刚定位：**64K 通信税 ~90% 是岛内失步（MoE 专家按 rank 切分负载不均 → AR 等最慢卡）**——EPLB 正是治这个病的工业标准
- **我们的翻译版**：岛内 3 卡各自持有 512/3 专家的静态切分 → 改成"按激活统计的热度感知放置"：冷专家可以只放一份、热专家放两三份副本，每卡总 token 数均衡 → AR 等待时间均衡
- **数据现成**：我们的 expert-profile.bin（热画像，E 线战役产物）就是 EPLB 需要的激活统计！
- **代价**（研究提醒）：冗余副本增加内存流量 ~39%——但我们显存有 5.5GB 余量，放得起
- **改造量**：中——加载器的专家放置表从"均匀切"改成"读画像放置"，MoE grouped kernel 需支持同专家多副本路由
- **判决：短期最值得做，直击实测病根，数据已有**

## 6. Chunked Prefill / Split-K Fusion（TRT-LLM）——已部分拥有

我们已有 FIFO + KV_PAD8192 的 chunked prefill。TRT-LLM 的 Split-K Fusion 变体（**prefill 块与 decode 单步合批**）我们没做——多流场景下 decode 不被 prefill 饿死的关键。**判决：多流适配时顺带吸收**

## 7. 不可吸收清单（免得绕弯）

- **NVFP4/CuteDSL 全家**（DWDP 官方路径、FlashInfer）：sm70 无 FP8/FP4 硬件，1Cat-vLLM 已趟过路
- **WGMMA/TMA/Async-Barrier**（Hopper 生态）：sm70 无这些指令，V100 的 mma.m8n8k4 已是我们的上限
- **PagedAttention V2 / FlashDecoding+**：依赖 sm80+ 的 async 拷贝，我们的 QSA 块稀疏 kernel 已是替代品

---

## 汇总：吸收优先级（结合我方现状）

| 优先 | 技术 | 治什么病 | 改造量 | 数据/参照 |
|---|---|---|---|---|
| 1 | **EPLB 热度放置**（SGLang 思想） | 64K 通信税真根（岛内失步） | 中 | expert-profile 已有；CloudMatrix 论文 |
| 2 | **DBO 双微批**（vLLM #51700） | prefill 通信/计算不重叠 + shape drift | 中 | gpu_model_runner 源码；Codex 失败教训 |
| 3 | **多序列 MTP 链**（Codex 清单 #1） | 多流聚合 75 vs 单流 125 | 中 | Codex 已定位单序列限制 |
| 4 | **DWDP 式专家分片**（长期旗舰） | 显存+通信+负载三合一 | 大 | TRT-LLM PR #12136；我们的变体 7 账 |
| 5 | **DeepEP 双缓冲 kernel** | MoE grouped 无重叠 | 中 | DeepEP 低延迟模式源码 |
| 6 | Mooncake 传输思想 | 冷启动 TTFT | 小 | Transfer Engine 文档 |
