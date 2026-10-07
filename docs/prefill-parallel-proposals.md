# Prefill 通信税攻坚方案总结（供 Codex 研究用）

日期：2026-10-06 | 状态：ZCode 侧实验进行中（prefill-parallel 代理 7 臂队列），本文供 Codex 平行研究/评审
机器：AC922 = POWER9 + 6×V100-SXM2-16GB（sm_70，NVLink 一代 mesh，每卡 ECC 后 15.47GB 可用）
模型：Qwen3.8-Flash-Next GSQ-RCO（48 层 = 36 GDN + 12 QSA，512 选 10 MoE，PLE 表 26.8GB 在 CPU）
现役生产：llama.cpp fork 岛 3,3（`LLAMA_ISLAND_SPLIT=3,3`，岛内 TP3）+ H + FIFO + MTP4，IQ3_XXS 73GB 全驻留（GPU 46.4GB），decode 94.3 tok/s，prefill ~955-1055 档

---

## 一、问题定义（实测数据）

64K prefill 的 nsys 拆分（qsa-longctx/REPORT.md §1.3，profiles/qsa-lc-64k/）：

| 项 | 占比 |
|---|---|
| **NCCL AllReduce（岛内 TP3 每层归约）** | **49%** ← 主敌 |
| MoE grouped | 15% |
| QSA attention | 3% |
| QSA 重建机器 | 1% |
| dense/GEMV/杂项 | ~32% |

对照：Strata 单/双卡 V100 prefill 1300+ tok/s（无逐层归约架构）vs 我们 ~760@64K。差距全在通信税。
另：岛 3,3 任一时刻只有 3 卡活跃于一个 ubatch（两岛串行交替）——**prefill 期间另一半卡闲置**。

## 二、方案清单（按改动量排序）

### 方案 A：六段层流水 `LLAMA_ISLAND_SPLIT=1,1,1,1,1,1`（零代码，实验进行中）
- **机制**：6 卡×8 层/卡，岛间切分=层分布，每个层边界一次激活拷贝（每 token 2560 floats），替代每层两个跨岛 AllReduce。单成员岛的 comm_init/P2P_AR 有 n_devs>1 门控自动跳过 → **零 AR 构造**
- **显存**：权重 46.4/6=7.7GB/卡 + 132K KV（8 层全头 Q8_0 ≈1.1GB）+ 缓冲 ≈ 12-13GB/卡 ✓
- **预期**：稳态 6 卡全开（vs 现役 3 卡）+ 零 AR → prefill 吞吐上界 **~2×**
- **代价**：decode 塌（每 token 6 级串行 + MTP draft 穿全流程）→ 仅作 prefill 专项探针
- **代码验证状态**（ZCode 已完成）：解析器 src/llama.cpp:181 `llama_island_split_sizes` 只要求等尺寸+总和=卡数，6×1 合法；fork 源码注释明说岛间切分语义

### 方案 B：per-phase 混合调度（prefill 层流水 / decode 岛 TP）——需源码
- split_mode 在模型加载时固定权重放置与 buffer 类型（load_tensors 一次性分配），运行期无切换路径；两种模式 KV 布局也不同（TP 头切 vs 层切）
- 需源码级双放置/权重重载/双 KV 布局——工程量大，但这是"prefill 快 + decode 不塌"的终极形态
- Codex 研究点：llama.cpp 有没有已有的 per-batch-type 分支路径可借（prefill batch 走 layer 分布、decode batch 走 TP），或者用两份 context 交替的可行性

### 方案 C：变体 7——稠密层 3 卡副本 + 专家表 6 卡分片（需源码，Strata 有参照）
- **机制**：利用 MoE 稀疏性（每 token 只激活 512 选 10）：decode 单步热集仅 ~3.5GB（稠密 2.6 + 热专家 0.86）
- **布局**：每卡 = 稠密层完整副本 2.6GB + 专家表 1/6（7.3GB）= 9.9GB → 余 5.5GB ✓
- **通信**：稠密零通信（每卡全有）+ 专家命中本地 / miss 时一跳 P2P 异步取行——**同步栅栏全部消失**（vs AllReduce 的全卡栅栏）
- **参照实现**：Strata `peer_experts.hpp`（1+1 形态，我们已读码），我们做 3+3；llama.cpp 的 `GGML_CUDA_MOE_GROUPED` 是它的退化形态
- Codex 研究点：fork 的 meta backend 加载器能否支持"同张量双布局"（稠密 replicate + 专家 shard）；MTP draft 层怎么归入副本；PLE（在 CPU）不受影响

### 方案 D：P2P_AR 的 prefill 口径复活（零代码 env，实验进行中）
- R17 的 `AC922_TEST_P2P_AR=1`（图内 P2P 归约替代 NCCL burst）当年 decode 口径判"噪声"，但 prefill 的 NCCL 606 次集中图切换窗口是另一个世界——正在以 prefill 口径重测

### 方案 E：通信/计算重叠（需源码，6c 设计文档已备）
- R18 async bridge 复活（+7.47% 单变量被门拒）+ 归约与下一块计算重叠；门禁 64K prefill ≥+10%

### 已判死
- `--split-mode row`：加载即 throw（源码实证，加载器不支持本模型组合）
- 3 卡完整权重副本（46.4/3=15.5GB/卡 > 可用，KV 零空间，物理死）

## 三、同时进行的关联实验（另一代理）

K8V4 KV 格式四臂对照（Q8_0/K8V4/F16 × 132K/262K）+ MTP accept 率——K8V4 若 accept 损失小，3×262K 并发（KV 池 6.6GB/卡）是新的并发旗舰候选。

## 四、给 Codex 的具体研究问题

1. 方案 B 的可行性评估（最有价值）：llama.cpp 的 graph 执行器能否 per-batch 分支两种 split 布局？工作量估计？
2. 方案 C 的加载器改造：meta backend 的 tensor 放置表（model.devices）能否表达 replicate+shard 混合？
3. 六段流水（方案 A）跑通后若 decode 塌得严重，有没有"层流水 decode 微批化"（把 MTP 窗口的 4-6 token 作为微批填流水）的先例？
4. NCCL 49% 里有多少是 burst（图切换窗口）vs 稳态归约？R17 trace 有 606 次四爆的原始数据（profiles/gpt-r17-trace-*/decode.sqlite）

## 五、数据工件索引

- 64K prefill nsys：服务器 profiles/qsa-lc-64k/（qsa-longctx/REPORT.md §1.3 拆分表）
- R17 全量 trace：服务器 profiles/gpt-r17-trace-20261002-1603-a3/decode.sqlite（77.8 万 kernel 事件）
- 岛机制源码：src/llama.cpp:181（split sizes 解析）、ggml-backend-meta.cpp（岛调度）
- Strata peer 参照：opensource/strata-engine/include/strata/core/peer_experts.hpp
- 进行中实验：服务器 manifests/prefill-parallel/（7 臂队列跑完会陆续落盘）
