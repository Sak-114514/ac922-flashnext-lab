# llama.cpp 上游同步评估（2026-10-06）

## 版本距离
- 我们 fork 基线：`7fe450e` = v0.5.0（#29333，2026-09-23）
- 上游 HEAD：`6c73b3e`（2026-10-06）= **v0.6.0**（#29997）+ ggml 0.26.0
- 距离：**~100+ commit / 13 天 / 一个大版本**，其中 22 个与 qwen4exp/Flash-Next/MTP/KV/CUDA 直接相关

## 上游的 Flash-Next 相关更新（按价值排）

### 🥇 高价值（建议 cherry-pick）

**1. PR #29825 — qwen4exp indexer 显存减半**（已合并）
- indexer 逐头评分 + relu 原地 + 逐头累加，消灭"全头一次性算+复制"的 4GB 中间量
- **131K ctx + ub4096：CUDA compute buffer 6.1→3.0 GiB；262K：11.7→5.6 GiB**
- logits 逐位一致（仅 matmul 归约顺序差），pp/tg 速度不变
- **对我们的意义**：直接解锁更大 ubatch（prefill 提速）和 262K 的 compute 余量——与 K8V4/262K 战役完美互补！我们当前 compute buffer 在 262K 峰值占 85.8%，这个减半能压到 ~60%
- 落点：src/models/qwen4exp.cpp 的 indexer 评分段

**2. PR #29901 — lightning indexer CUDA tile kernel（4 头版）**（已合并）
- 64 keys×8 tokens 分块，keys 半精度 staging 一次复用，fp32 累加
- RTX PRO 6000 上 **indexer kernel 2.5×，prefill +4.4%**（3867→4038），indexer 占比 9.6%→4.0%
- **作者明说"老架构收益更大"**（3090 上 Vulkan shader 4× 于 CUDA 向量核）——**V100 可能比 2.5× 更高**
- 落点：src/kernels/lightning-indexer.cu（我们 fork 有此文件，sm70 走 SIMT 向量核——正是被优化掉的那个）

**3. PR #30020 — nextn 提取 flag 变化时的 sched 重留**（已合并，v0.6.0）
- MTP nextn 动态提取时的图重建修复——可能与我们的 mtp-msgraph 发现（双槽失效）同族问题

### 🥈 中价值（跟随同步吸收）

**4. PR #29824 — qwen4exp mask 构造优化**（Metal 生态收益为主，CUDA 侧小）
**5. #29943 — selective expert copying 重构到用户代码**（MoE 布局重构，与我们的 MOE_GROUPED 交互需查）
**6. #29994 — k-pool 共享序列 data race 修复**（并发场景正确性）
**7. NVFP4 MMQ 累积优化 #29857**（我们不跑 NVFP4，低优）

### 🥉 基建类
- vLLM 生态 modality 报告、LibreSSL、CI 类——随大版本走

## 同步策略建议（给 Codex 的决策输入）

**不建议整树 merge**（我们 fork 有 ~130 文件的补丁面：岛机制/meta backend/GDN X4/QSA 稀疏/FIFO/四个 kernel 开关，v0.5→v0.6 的 ggml 0.26 重构会造成大规模冲突）。

**建议 cherry-pick 三连**（按序，每个独立验证）：
1. **#29825**（indexer 显存减半）——纯模型层改动（qwen4exp.cpp），与我们 fork 补丁面的冲突最小；收益立竿见影（262K compute 余量）；parity 门现成
2. **#29901**（indexer tile kernel）——CUDA kernel 层，需检查与 sm70 路径的兼容（它说"batch 小于 token tile 走向量核"——decode 不受影响正好）；预期 V100 prefill +2-4%
3. **#30020**（nextn sched 重留）——与 mtp-msgraph 的移交项对照，可能解决双槽图复用的上游侧

冲突风险评估：#29825 低（模型层独立函数）；#29901 中（kernel 文件我们动过 MMQ 相关）；#30020 低（sched 层我们基本没碰）。

## 数据工件
- 上游 PR 详情：gh pr view 29825/29901/29824/30020 --repo ggml-org/llama.cpp
- 我们 fork 的 262K 显存数据：flashnext-llama-experiment/kv-262k/RESULTS.md（compute buffer 峰值 85.8%）
- indexer 现占比（我们 64K nsys）：qsa-longctx/REPORT.md §1.3（评分 GEMM 0.07ms/步 decode；prefill 侧更大）
