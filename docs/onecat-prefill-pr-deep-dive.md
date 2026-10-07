# 1Cat-vLLM prefill 注意力 PR 链深挖 + 我们 fork 的 QSA prefill 实施设计

- 日期：2026-10-06 · 执行：ZCode（纯本地只读：`gh api` 读 GitHub PR + 本地 clone 源码阅读，零服务器访问）
- 数据源：`gh api repos/1CatAI/1Cat-vLLM/pulls/{286,315,387,390,417,163,268,666,754}`（body + files patch 全量）；本地 clone `~/Desktop/ssh/opensource/super-ac922-clone`（= Sak-114514/super-ac922，上游 1Cat-vLLM fork，分支 ac922-ppc64le = 上游 db292f9a/PR#703/2026-09-27 + 5 个 AC922 commit——**全部目标 PR 的合并后代码都在本地树里**）；上游 ik-llama 镜像 `~/Desktop/ssh/opensource/ik-llama`（commit 5f89bfc/#2568，我们 fork v0.5.0 的上游基线）；我们 fork 的运行时事实取自 `flashnext-llama-experiment/TECH-FACTS.md`（[M]/[C] 标注）与 `reports/` 既有档案
- 红线遵守：GitHub 只读；本地 clone 只读；报告仅写入 reports/

---

# A 部分：PR 链机制拆解（diff 级证据）

## 0. 先校准两个容易搞混的模型几何（后续所有判决的基础）

| 层类型 | 模型 | 全局几何 | 1Cat TP4 本地 | 我们岛内 TP3 本地 |
|---|---|---|---|---|
| dense full-attn（16 层） | Qwen3.8-27B（FP8/NVFP4 线） | Hq=24, Hkv=4, D=256 | 6/1，**组大小 6** | 不适用（27B 非我们生产模型） |
| QSA 稀疏（12 层） | Qwen3.8-Flash-Next | Hq=24, Hkv=12, D=256，**组大小 2** | 6/3（kernel 常量 `kGroupedVerifyHeads=6`） | 8/4，gqa=2 |
| GDN 线性（36 层） | Flash-Next | 递推态，无 KV | per-slot state | 岛 TP3 头切 |

证据：1Cat `flash-attention-v100/kernel/flash_decode_paged.cu`（本地树）L1835-1841：`kGroupedVerifyQ8MaxQ=8; kGroupedVerifyHeads=6; kGroupedVerifyHeadDim=256; kGroupedVerifyBlockN=32`；79T 契约见 `csrc/attention/sm70_79t/VALIDATION.md`（"24 query heads / 4 KV heads / D256"）；我们 gqa=2 见 `reports/FLASHNEXT-FULL-EXPERIMENT-REPORT.md` L155（v70 判死记录）。

**推论（贯穿全文）**：GQA 打包的"每 KV 头组打包 N 个 Q 头"在 TP 头切分下组大小不变（Hq、Hkv 同除）。1Cat 的 6 头/12 头打包在 27B dense（组 6）上成立；Flash-Next QSA 组大小只有 2，**"6 头拼宽"对 QSA 层从来不是 6，而是把 Q 行折进 M 维的那套布局思想**。

## 1. 注意力 TFLOPs 主链（17.92 → 47 → 60.8 → 71/75+ → 79T 天花板）

README L66-75 的链条表格 + L470-620 的"Layer 2 — Rewrite D=256 Attention as a Volta-Native Pipeline"叙述（本地树 README.md 同内容）。全部为 27B dense 层 Q8000/Hq6/Hkv1/D256、scale 1/16 契约。

### 1.1 v1.3.0：D256 Split-D / N32（17.92 → 46.63-47.1 TFLOP/s，≈2.6×）

**证据**：release notes（`gh api releases/tags/v1.3.0`）+ 本地 `cmake/patches/sm70_flash_attn_d256_{pipeline,k_pingpong,splitkv3}.patch`（1523/90/444 行，vendored vllm_flash_attn 补丁族）+ `sm70_flash_attn_d256_gqa_arch.patch` L1452-1654 对 `flash_fwd_d256_splitd_sm70.cu` 的修改（暴露 `StoreState` 模板参数与 `sm70_d256_splitd_dense_state_fwd` 导出，grid = `(q_len/kBlockM, batch, heads_q)`）。

机制（release notes 原文 + patch 交叉）：
1. **Split-D 不是切 K/V，是切 head_dim 输出维**：D=256 拆 4 个 D64 片。QK^T 只算一次得到 P（概率），PV = P·V 的输出 O[×256] 按列拆给 4 个 warp 分别算——"Paired warps share QK probability work while increasing PV parallelism without recomputing the same QK work"。解决的是 D=256 时单 warp PV 输出寄存器/smem 过肥的问题。
2. **N32 online softmax**：不物化全 score 矩阵，保 causal online-softmax + FP32 累加契约。
3. **K-stage ping-pong**：K/D64 panel 在 smem 双缓冲 stage 间交替，降 barrier 等待。
4. **Operand flow 重建**：QK/TN、PV/TT 各配专用 Tensor Core 布局；phase-swizzled K 布局 + 向量化 smem 读写降 replay。
5. **软件流水**：下一 K tile 预取与当前 tile 计算重叠；PV operand 双缓冲。
6. **E5M2 KV 桥**：FP8 KV 页一次性反量化进可复用 FP16 workspace，热路径免 per-element FP8 开销。
7. **Split-KV3**：长前缀 KV 三分区 + FP32 partial 合并（#268 的 128-bit load 就服务这条路）。

**为什么在 V100 有效**：Volta WMMA 碎片是 16×16×16、smem 96KB/SM、无 cp.async（Turing 才有）——D=256 单块全做时 smem 塞不下 K+V 双 panel + Q，寄存器压力大。Split-D 把 PV 的输出维拆开 = 每 warp 持有的 O 碎片从 16×256 降到 16×64，QK 不重复算是省 FLOPs 的关键（naive 拆法 QK 会做 4 次）。

### 1.2 PR #286：GQA-packed wide QK/PV（→ ≈60.8T，merged 2026-08-25）

**文件**：7 files +2201/-1。kernel 交付形态是 **vendored FA2 的 CUTLASS 补丁**：`cmake/patches/sm70_flash_attn_d256_gqa_arch.patch`（1673 行，本地树全量可读），新增 `csrc/flash_attn/src/flash_fwd_d256_gqa_arch_sm70.cu`（1359 行）+ 修改 `flash_fwd_d256_splitd_sm70.cu`（StoreState 挂钩）+ `vllm/v1/attention/backends/flash_attn_v100.py`（+113 行调度）。

**GQA 打包的具体布局（patch L87-460，kernel 头注释原话："QK GEMM writes FP16 logits and row statistics. PV normalizes logits in its A-operand transform and writes one reusable FP16 block partial."）**：
- **打包方式 = 把 6 个 query 头折进 GEMM 的 M 维**：`kRows = 48000 = Q8000 × Hq6`，`kHeadDim=256`，`kPrefix=120000, kTail=8000`（基准 harness 常量，L765-770）。Q 矩阵逻辑上是 [48000, 256]，K/V 是 [KV, 256]，**因为 Hkv=1，整个宽 GEMM 共享一份 K/V——算术强度直接 ×6**。这不是"6 头拼成 12 列宽"之类的 N 维操作，纯 M 维堆叠。
- **QK GEMM**：CUTLASS `GemmSoftmax`（cutlass example 35 改），SM70 `OpClassTensorOp`，`QKThreadblockShape=128×128×32`，`QKWarpShape=32×64×32`，`QKInstructionShape=8×8×4`（Volta HMMA.1688），epilogue 输出 FP16 logits + 每 tile 的 row max（`LayoutN`）与 row sum（`LayoutS`）统计（patch L136-160）。softmax scale 1/16 作为 `alpha` 直接烘进 GEMM（L885-886）。
- **PV GEMM**：`DefaultGemm` + `MmaPipelined`，`PVThreadblockShape=64×128×32`，自定义 **A-operand transform `ExpRowSumTransformA`**（L234-330）：PV 的 A 操作数直接吃 QK 输出的**原始 FP16 logits**，在 GEMM operand 加载路径里逐元素做 `exp2f((value - row_max)*log2e) * row_inv_sum`——即 softmax 归一化融进 GEMM 数据通路，不需要单独一个 P 矩阵物化 pass。两个变体：`PREFIX_QK_FULL_STATS`（QK epilogue 已给全统计，transform 纯查表）与默认版（blockIdx.y==0 的 CTA 顺带 warp-shuffle 归约 row_sum 写回 `g_row_sum_out`，L313-330）。
- **前缀状态机（不存所有 block partial）**：每 8192-token prefix block（`kBlockN=8192`，生产入口 L1246）流水四步：
  1. `QKGemm` → scores[48000×8192 FP16] + stats
  2. `PVLauncher` → 一个可复用 FP16 block partial [48000×256]
  3. `prepare_prefix_update`（每行一个线程，算新旧 max 合并的 old_scale/block_scale，L569-609）
  4. `apply_prefix_update_half2`（half2/fma 向量化把 partial 折进 **FP32 prefix accumulator** [48000×256]，L603-628）
  循环结束后 `merge_prefix_accumulator_tail` 把前缀状态与 exact causal tail 的 (output,max,sum) 三元组做最终 online-softmax 合并（L647-674）。
- **exact causal tail**：走 `onecat_sm70_d256_dense_state_raw`（= Split-D kernel 的 StoreState 模式，导出 max/sum 以便与 prefix 状态同坐标系合并；patch L1466-1520 的状态导出注释："Export the dense-tail state in the same natural-log coordinate so the two states can be merged directly"）。
- **生产入口与 workspace**（L1148-1359）：`sm70_d256_gqa_architecture_fwd` 硬契约检查（Q8000/Hq6/Hkv1/D256、KV∈[16K,256K] 步进 8K、scale=1/16、FP16、contiguous）；**750MiB per-device score workspace**（48000×8192×2B）持久缓存于 per-device map，`cudaEventDisableTiming` event + `std::mutex` 串行化跨流复用；首次分配前 `cudaMemGetInfo` 预检 + **128MiB downstream headroom** 门禁，不满足抛 `OutOfMemoryError` → 上层 fallthrough 到 exact dense 路径。
- **输入输出 shape 约定**：q/k/v/out 均 4-D `[1, seq, heads, 256]`，q `[1,8000,6,256]`，k/v `[1,KV,1,256]`，out 同 q。GEMM 内部把 q 视为 [48000,256] row-major、K 转置当 B（`QKLayoutB=ColumnMajor`, stride 256）。
- **性能口径**：`kFrozenUsefulTflop = 60.0*0.101581`（该形状的有用因果 FLOPs 常量）÷ 实测秒。PR body：KV40K dense 39.29→30.28ms（1.298×）、KV128K 136.08→100.20ms（1.358×）；12 形状全快 1.288-1.355×；matched TP4 全模型 prefill +11.06%。QK/PV 编译 254/119 寄存器零 spill。

**本质**：把"许多碎片小 tensor-op"换成"少量大而规则的 QK/PV GEMM"（README L598-602），M 维 ×6 后 CUTLASS SM70 GEMM 的 tile 效率（128×128 TB）才有意义；prefix（非因果区，行间无依赖）与 causal tail（有依赖、需精确）分离调度，前缀区允许 8K 大 block 的两段式 GEMM+状态折叠。

### 1.3 PR #315：79T 实验天花板（merged 2026-08-28 但质量门 FAIL）

**证据**：PR body 全文 + 本地 `csrc/attention/sm70_79t/prefill.cu`（6899 行现役版本）+ `README.md`/`VALIDATION.md`（同目录）。

- 把 #286 的 KV40K..128K 门放宽到 **每个完整 8K chunk（KV8K..256K，32 档）**，chunk-sum 128K 1197.26→633.82ms（1.889×，**79.41 有效 TFLOP/s**）、256K 4711.80→2553.40ms（78.85T）。
- 79T 配方 = **half2 多项式 exp2 近似**（degree-5/range-4：`PREFIX_PV_HALF2_TAYLOR_C1..C5 = 1, 1/2, 1/6, 1/24, 1/120`，prefill.cu L88-102）+ 零偏移指数 + **无保护 FP16 累加**。
- **质量门 FAIL（关键负面证据）**：TP4 256K 全模型门：性能 128.41→88.13s（1.457×、prompt 2971.52 tok/s）但 **"FAIL: 32 token IDs are zero"**。VALIDATION.md 复盘："raw historical recipe assumes that unshifted exponentials and the FP16 PV numerator fit in FP16. They do on zero-mean random tensors, but they do not on the model tensors. All captured Q/K/V inputs were finite while the raw output contained infinities and the model emitted invalid token ID -1"（随机输入测不出的溢出在真实模型张量上爆）。首败 rank3 attention call 4；16× value headroom 的候选撑到 KV144K 后 rank2/3 在 KV152K 独立溢出。
- 状态：**Keep Draft; do not promote**。79T 保留为"experimental research ceiling; not the default production quality claim"（README L540）。

### 1.4 PR #666：FP32 QK/PV + wider local head layouts（→ 70-71T 历史纪录，现役 >75T）

**证据**：PR body（含 prefill power-policy audit）+ 51 文件清单 + 本地 `sm70_79t/prefill.cu`/`README.md` + `vllm/v1/attention/backends/flash_attn_v100.py` 的 #666 diff。

**(a) 为什么"FP32 反而更快"——两个正交原因**：

1. **Volta 上 FP32 累加本来免费**：Volta HMMA 的 FP16 输入 tensor op 无论 FP16 还是 FP32 累加，吞吐同档（V100 dense FP16 15.7 TFLOPS/精简 FP32 累加路径同源）。QK 改走 cuBLAS `cublasGemmEx(..., CUDA_R_16F in, CUDA_R_16F out, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`（prefill.cu L1939-1965，log2 域变体把 `0.0625*log2e` 烘进 alpha），PV 改 `PVAccumulator=float`（L462-468："Volta Tensor Cores accept FP16 operands with FP32 accumulation. Keeping the long-K dot product in FP32 makes larger prefix blocks numerically viable without materializing an FP32 probability or output workspace"）。FP32 累加不是买精度付性能，是同吞吐白拿。
2. **早期 FP32 路线慢的真因是寄存器 spill，被 tile 形状修掉**（sm70_79t/README.md 原话）："The 128x256 threadblock and 64x64 warp shape avoid the register spills that made the earlier FP32 path slow."（PV_TB_M=128/PV_TB_N=256 变体，prefill.cu L424-448）。FP32 累加器寄存器占用翻倍 → 错误 tile 下 spill 到 local memory 反而慢；tile 改对后，FP32 累加解锁了 **24K prefix block**（K=24576 的长点积，FP16 累加数值上撑不住）→ block 数从 8K 粒度降到 24K，块间合并开销和 score workspace 往返同步下降。**所以因果链是：FP32 累加（免费）→ 更大 block 数值可行（必要条件）→ 128×256 TB（免 spill 的充分条件）→ 净赚**。

**合格化护栏**（这是它没重蹈 #315 覆辙的原因，README.md）：score maxima 以 stride 8 采样 + 固定 margin + 指数帽；value 中心化 + 残差按 2 的精确幂缩放留 64× headroom；每 24K 前缀块以 FP32 写出；prefix/tail 合并全程 FP32。**它仍是近似路径**（scores/概率存 FP16、maxima 是采样的），必须过采样 FP32-oracle 检查 + 冷模型请求双门。现役合格中位数 **>75 TFLOPS @KV128K/256K**；`prefill_dense_d256_gqa_79t_fp32` 路由计数器族标识。Q8000 走 320-token tail 特化、Q8192 原生 256-token tail、Q8001-8191 前导 pad 到 8192（bottom-right causal 对齐，多余功 <2.4%）。
- **功率物理**（PR body 附录，对我们的 SXM2 卡极重要）：同一二进制 185W 功率帽下 Q8192/KV262144 operator 只有 **53.816T（~967MHz + 反复 power-cap 事件）**，300W 下同形状历史纪录 71.111T。"Full-FP32 attention remains 70-71T, below the requested 75T" 是 185W 生产帽下的数字——**SM70 长时注意力实测吞吐是功率策略的函数**，跨报告引用必须带功率口径。

**(b) "wider local head layouts" 是什么**：#666 在 `flash_attn_v100.py` 把准入从卡数白名单改成**张量几何判据**（diff 实证）：
- `_should_use_prefill_d256_gqa_architecture`：旧 `query.shape == (1, max_seqlen_q, 6, 256)` + `key.shape[2:] == (1, 256)`（即只认 TP4 的 6/1）→ 新 `query.shape[2] == 6 * key.shape[2]` 且 `key.shape[2] > 0`——**任意 Hkv_local，只要组大小 =6**（27B 的 TP1 24/4、TP2 12/2、TP4 6/1 全准入）。
- `_e4m3_batch_xqa_allowed`：`query.shape[1:] == (6,256)` → `query.shape[1] % 6 == 0`。
- 新增 `_sm70_gqa_has_fp32_accumulation()`：要求 FA2 库自报 `sm70_d256_gqa_accumulation_bits()==32`，否则打 warning 退回 exact dense（防旧库载入近似路径）。
- 新增 `_profile_sm70_prefill_workspace`：profiling 期用零张量预跑一次 op，把 750MiB workspace 计入显存 profile，防 KV allocator 先吃光预算导致运行期 OOM fallthrough。
- 同 PR 还带四个质量缺陷修复（Mamba 态未物化、QPN2 提前舍入全局 scale、Gemma 归约不确定性、采样 cutoff 用了第一个请求的温度），与 kernel 无关但说明其准入纪律。

### 1.5 A1 四代总结表

| 代 | 机制核心 | 数字 | 对我们的启示 |
|---|---|---|---|
| v1.2.2 旧路径 | （baseline） | 17.92T | ≈我们 fork fattn 系的量级（见 B1） |
| v1.3.0 Split-D/N32 | PV 输出维 4×D64 分 warp、QK 只算一次、K ping-pong、软流水 | 46.6-47.1T | Volta 原生 D256 kernel 的正确打开方式；我们 fattn-wmma 每 warp 独立持 O 全 256 维 |
| #286 GQA-packed | 6 头折进 M 维（48000 行宽 GEMM）、A-transform 融 softmax、前缀状态机、prefix/tail 分离 | ≈60.8T | **M 维堆叠 + KV 单份复用**思想可跨架构；具体 CUTLASS 契约绑 vLLM 布局 |
| #666 FP32 QK/PV | cuBLAS FP32-compute QK + CUTLASS FP32-acc PV + 128×256/64×64 tile + 24K block + 采样护栏 | 70-71T@185W / >75T@合格中位数 | FP32 累加免费 + tile 定生死 + 近似路径必须有 oracle 门 |
| #315 79T | half2 Taylor exp2 + 无保护 FP16 累加 | 79.4T 但质量 FAIL | 负面教材：随机输入的数值门是假门 |

## 2. PR #387：QSA full-model prefill（4446.64 → 5777.43 tok/s @64K，+29.93%）

**文件**：6 files +997/-92。kernel：`flash-attention-v100/kernel/flash_decode_paged.cu` +661/-48（`flash_attention_grouped_verify_e5m2_partial_kernel` 加 `SPARSE_PAGE4` 模板路径 + 新 planner kernel）；`include/fused_mha.h` +13（ABI v2：`grouped_sparse_page4_plan_fwd`/`grouped_sparse_page4_fwd`，支持量化 KV）；`vllm/models/qwen4_exp/nvidia/ops/qsa.py` +163/-25（调度层）。

**它解决什么**：旧 QSA 路径（Triton，55.940ms/4096 行）逐 query 行独立处理——每行的稀疏选择各自 load Page4 K/V，相邻 8 行的选中集重叠（局部性）但 K/V 反复读。

**机制三件套（diff 实证，本地树合并版同步核对）**：

1. **8 行分组 + 块去重 planner**（`grouped_sparse_page4_plan_kernel`，512 线程/CTA，每组一个 blockIdx.x）：
   - 开放寻址哈希表 8192 槽（`kGroupedSparseHashCapacity=8192`，乘法散列 `phys_microblock * 2654435761 & (cap-1)`），entry = 64-bit：低 32 位物理 microblock id（物理页 × 每页 microblock 数 + 页内偏移，`grouped_sparse_physical_microblock` 做逻辑 token→物理页→microblock 映射并校验 block_table_width/num_cache_blocks 边界），高 32 位 **token_mask：8 个 query × 每个 4 bit，恰好编码"这个 4-token microblock 内该组每行选中了哪几个 token"**（`atomicOr` 合并，full Page4 = `0xF<<query*4`，散 token 逐位插，尾部 partial page 单独处理——planner L400-470）。
   - 输出按 **active_m_tiles 类别（8 类 = 3 个 16 行 m-tile 的激活位图）分桶排序**：同类块连续存放 + 桶内 pad 到 8 的倍数（pad 行 mask=0）；每 group 输出 `[groups, ≥4160]` 的 (microblock, mask) 表 + 等效 seq_len（×4）。确定性：4 次 replay bitwise 一致。
2. **执行 kernel**（grouped verify kernel 的 `SPARSE_PAGE4=true` 变体，blockIdx.z = group）：
   - 每 tile（`kGroupedVerifyBlockN=32` = 8 个 microblock）把 mask word 读进 smem（`sparse_token_masks[BlockN/4]`，L2108-2119）；聚合 tile 内所有 mask 的 nibble 得 `active_query_nibbles`，展开成 **`active_m_tiles` 3-bit 位图**（Q8MaxQ=8 行 × 6 头 = 48 行 = 3 个 16 行 tile）；**QK 阶段 warp 级直接 return**（`grouped_verify_qk(..., active_m_tiles)` 里 `if ((active_m_tiles & (1<<m_tile)) == 0) return;`，L2135-2160）、**PV 输出阶段 `continue` 跳过死 tile**（L2333-2339）。
   - **精确 per-row 4-bit mask**：`grouped_verify_key_visible<SPARSE_PAGE4>`——`mask_bit = token_idx*4 + (kv_idx & 3)`，可见性判据从 dense 的 `kv_idx <= prefix_kv_len + token_idx` 换成查 bit（L1941-1964）。**去重只省 load，不省正确性：每行的注意力集合逐 bit 精确**。
   - QK/PV 用 **Volta WMMA**（`volta::fragment<16,16,16,half>`）；KV dtype 模板化 `KV_DTYPE`（FP16/E5M2）；全无效 group 输出直接写 0。
   - `qsa.py` 侧（本地树 L1816-1868）：planner → **pad fix**（`torch.where(token_masks==0, grouped_pages[:, :1], grouped_pages)` 把 padding 行重指到组内第一个真 microblock——注释写明 W8 bug：page 0 若存 GDN 态反量化出 E4M3 NaN，`0 * NaN` 在 P@V MMA 里存活）→ physical KV 视图 → forward。
3. **单行 fallback**：旧 single-row Page4 路径保留为"1024-partition fallback"。

**数字**：per-layer/rank 55.151→**9.632ms（5.518×）** + planner 0.362ms；4096 行混合 Page16/Page4 正确门 relL2 2.434e-4、cos 1.0；生产布局隔离路由（含 Python 派发+GPU planning+pad+attention）16.132ms vs Triton 55.940ms（3.468×）；全模型 no-MTP TP4 纯 prefill：32K/64K/131K = +32.36%/+29.93%/+32.69%（8K 单点 6394.74）。Nsight：per-GPU kernel duty 98.27-98.61%——**修完后下一热点是 NVFP4 MoE 链**（直接催生 #390）。

**契约注意**：`kGroupedVerifyHeads=6` 是 TP4 本地头数——kernel 的 48 行/组布局 = 8 query × 6 头绑死 TP4 几何，换 TP 需要 kernel 模板参数化（我们 TP3 是 8 头 → 64 行/组 = 4 个 m-tile，位图改 4-bit）。

## 3. A2：PR #390 Indexed NVFP4 MoE prefill（+8.03% @64K）

**文件**：13 files +484/-48。核心：`csrc/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.cu` +43、`csrc/sm70_turbomind/ops/awq_sm70_gemm.cu` +63、`vllm/model_executor/layers/quantization/nvfp4_sm70_moe.py` +147。

**机制（diff 实证）**：
- 旧路径把路由结果物化成 `[tokens × topK, hidden] = [tokens×10, 2560]` 的 **FP16 行扩展**（每 token 复制 10 份激活进 permuted buffer）再喂 grouped GEMM——64K chunk 即 655K 行 × 2560 × 2B ≈ 3.3GB 的纯搬运。
- #390 的替代：**轻量 metadata kernel**（`buildMoePermuteMetadataKernel`，每 expanded row 一个线程）只派生两样 int32：`input_row_indices[row] = expanded_source_row / topk`（原始 token 行号）和 `permuted_idx`；然后 grouped GEMM 的 A 布局描述符带 **`desc_A.idxs = a_indices_flat.data_ptr<int>()`**（awq_sm70_gemm.cu diff：`nvfp4_moe_gemm_sm70_out_impl` 新增 `logical_total_tokens` + `a_indices` 形参，TurboMind `MatrixLayout.idxs` 语义 = GEMM 读 A 时按 idx 间接寻址）——**激活零拷贝、零扩展，W13 直接按 token 行索引取数**。
- 严格准入：E512/K10、K2560/N320、group 16、≥128 token、TP4、`VLLM_SM70_NVFP4_QWEN38_MOE_INDEXED_PREFILL=0` 回滚；路由顺序/W2/unpermute/输出算术全部不动（"bitwise equal"）。
- 8K operator 链 6.027→4.235ms（1.423×，1.791ms/层/rank）；全模型 32K/64K/131K = +8.47%/+8.03%/+7.71%，8K +9.01%。
- 与 decode 共享：decode 的 M≤8 已走 QPN M1 路线，此 PR 只动 prefill（≥128 token 门），同一个 `nvfp4_moe_gemm_sm70_out_impl` 底座两个入口（indexed 与 dense-stage）。
- 后续 #393（README 提及）：FP16 SwiGLU 精确融合 + N320 W13 尾拆 N256+N64 + 砍尾 tile 空转。

## 4. A3：PR #754 Batch prefill across discarded Mamba checkpoints（零命中 +66% 之谜）

**文件**：4 files +286/-11。**native 零改动**——15 个 native 二进制 SHA256 与 main 完全一致；改动 = `vllm/v1/core/sched/scheduler.py` +34 行（其余是测试和文档）。

**真实机制（不是缓存红利）**：Flash-Next MTP4 prefix caching 的默认 sparse retention 只保留少数 recurrent state（replay 位置/共享前缀/周期边界），**但调度器仍在每个 816-token state 边界把整个模型前向切一刀**（`_mamba_block_aligned_split` 的旧逻辑 `next_block_boundary = (start//block_size+1)*block_size`——因为 align allocator 只在 chunk 末尾物化 state，跨多 state 块的 step 会在内部槽留 null state，所以"每个越过的边界必须结束一个 chunk"）。结果：64K prefill 被切成 ~80 个 816-token 的**小前向**——每刀都是全模型层的浅批，GPU 利用率塌方。修复：`retained_boundaries = coordinator.get_replay_boundaries(...) + 周期 retention 边界`，**只在真正保留的 checkpoint 收尾 chunk，跨过被丢弃的内部状态继续批**（scheduler.py diff L457-500，注释原话："Materializing discarded states must not split the whole model's prefill into one-state-block forwards"）。

**数字（matched 1.5.1 wheel，零缓存命中的 warm 复用场景）**：8K 3060→5356（+75%）、32K 3046→5068（+66%——任务书"零命中 +66%"即此）、131K 2818→4666（+66%）。prefill 墙钟 0.438→0.237s @8K。仍比 prefix-off 慢 12.6%/6.3%/1.4%（保留边界的额外前向是真实成本）。

**本质**：这是"chunked prefill 的切分粒度必须跟着**状态物化需求**走，而不是跟着状态块网格走"的一般性教训。批深度（每前向的 token 数）是 GDN/混合架构 prefill 吞吐的一阶变量。

## 5. A4：向下钻的相关 PR

### PR #268（128-bit E5M2 XQA load，operator 1.212-1.265×，全模型 decode +7.92%）
"不是表面向量化"的实锤（README L317-352 + PR body）：真实 paged-KV 分区里，naive SM70 路径每个 FP8 碎片都重复"load Page ID → 算地址 → load 窄转换组"。#268 的实质 = **Page ID 复用**（同一分区内的 256-token 页元数据只取一次）+ **两个 half8 转换组合并成一个对齐 128-bit load**（`p256 dual-CTA、page≥256、B≥4` 契约）。SASS 合同：对齐 128-bit cache load、168 寄存器、零 spill、同双 CTA residency。bitwise exact（max_abs_diff=0）。这是数据通路级优化——地址工作/依赖等待/标量流量才是被消灭的对象，"128-bit"只是表征。

### PR #417（DFlash2 NVFP4 prefill +30.1%/+37.7%）
**dispatch-only PR**："This PR changes dispatch only. It does not change kernel arithmetic"。把已存在的 lossless QPN2-packed bounded prefill 路径自动选为 DFlash2 TP4 B1 契约默认（gate：method=dflash、q7、top-K 16、PP1/TP4、精确 dense 层形状）。四路 route-hit 证据：QPN2 M≤8 自动、QPN2-packed prefill M≥1024 自动、D256 sidecar、FP8 E5M2 bridge、**DFlash2 context-only chunk skip** 全激活。32K 4069.25 / 64K 3566.94 tok/s。对"prefill 优化"的贡献是把一组已合格 kernel 的**组合调度**打通——与 #387/#390 的 kernel 战不同属一类。

### PR #163（DeepSeek-V4 sparse MLA -90.64% ms/token）
单 CTA sparse MLA decode 瓶颈 → graph-safe FP32 split-K attention-state reduction：C4 1.957→0.101ms、SWA 0.586→0.059ms/层（9.9×）、E4M3 全 256 code bitwise 相等、TP8 TPOT 76.27→33.85ms。**decode 侧 split-K，与 prefill 主链不同族**——但它与 v1.3.0 Split-KV3、#286 的三段式合并共享同一个技术族：**"分区并行 + FP32 状态归并"**。对我们 GDN/QSA 的 decode 侧（而非 prefill）是更贴的参照；我们 fork 已有的块稀疏 qsa-attn decode kernel 即同族。

### 族谱判断（任务问"是否同族"）
- **同族 A（大 M GEMM 化 + 前缀/尾部分离 + FP32 状态合并）**：v1.3.0 Split-D → #286 → #315 → #666（dense prefill 主线）
- **同族 B（稀疏选择 → 组化/去重 → 精确 per-row mask → WMMA）**：#387（QSA prefill）；我们 fork 的块稀疏 QSA decode 与之隔代同族
- **同族 C（分区 + FP32 归并，decode 侧）**：#163、Split-KV3、#268 的分区页复用
- **异族（调度/派发）**：#754（批深度）、#417（组合派发）、#390（索引化免物化——数据通路重构，不动算术）

---

# B 部分：我们 fork 的 QSA prefill 实施设计

## B1. 现状定位：我们的 64K prefill 里注意力走什么 kernel、占多少

### kernel 链（上游 ik-llama #2568 镜像代码级证据；我们 fork 基线 v0.5.0 同构）

V100 dispatch 全链（`ggml/src/ggml-cuda/fattn.cu` L83-176 + `common.cuh` L218-228）：
1. `fast_fp16_available(sm70)=true` → 不进 tile_f32 分支；
2. `new_mma_available(cc) = highest_compiled_arch >= CC_TURING`（common.cuh L221-224，注释明言"Volta technically had FP16 tensor cores but they work very differently"）→ **V100 恒 false**，new-mma/vec-gqa 路全跳过；
3. `Q->ne[1] > 32`（prefill 成立）且 D=256 → **`ggml_cuda_flash_attn_ext_wmma_f16` → `fattn-wmma-f16-case<256, 256, cols_per_block=32, half>`**（fattn-wmma-f16.cu L138-159；`Q->ne[1]<=32` 时 cols=16，`<=8` 时 cols=8）。
即：**我们 QSA 层 prefill 的 QK/PV 主 GEMM = fattn-wmma-f16 的 256-D 模板，每 CTA 只处理 32 个 query 列 × 单头，Q/K/V 全 FP16，WMMA 16×16×16**。GQA 头（岛 TP3 下 8 头）各自独立 CTA，K/V 依赖 L2 跨 8 个头复用——与 1Cat "6 头折进 M 维 + K/V 确保单份"正好是两个极端。稀疏性以 mask（fattn 的 `mask` 参数，行=query 列块）表达；fork 的 QSA 快路径（B 路径 `LLAMA_QSA_POOLED_MULTI` 转正，TECH-FACTS §二）在块/cell 选择层做稀疏，kernel 本体仍是这个 WMMA 模板。indexer 侧刚回移的 #29901（64-key×8-token tile）+ #29825（fused scorer/CUDA 四头）只动了评分链（TECH-FACTS 2026-10-06 节 [C/M]），**主 QK/PV GEMM 未动**——与任务书描述一致。

### 占比重估（prefill 口径）

| 数据点 | 数值 | 来源 |
|---|---|---|
| 64K prefill nsys：QSA attention | 3% | qsa-longctx REPORT §1.3（经 prefill-parallel-proposals.md 引用） |
| 64K prefill nsys：NCCL / MoE / QSA 重建 / 杂项 | 49% / 15% / 1% / ~32% | 同上 |
| S1 nsys（8K 窗口）attention/QSA 专用核 | 2.3-2.6% | TECH-FACTS §一 [M] |
| v70 判死时的目标占比评估 | "仅占 prefill 3%" | FLASHNEXT-FULL-EXPERIMENT-REPORT L155 |

**算术交叉验证**（说明 3% 不是测量误差而是稀疏性的必然）：64K=65536 token，UB512 → 128 个 ubatch；12 层 QSA；岛 TP3 每卡 8 头；每 query 选中集 ~2K token（QSA top-k 家族）。有用 FLOPs ≈ 4·M·N·K = 4 × (512×8) × 2048 × 256 ≈ 8.6 GFLOP/层/ubatch/卡 → ×12 层 ×128 ubatch ≈ **13.2 TFLOP/卡**。若 kernel 有效算力 ~7-10 TFLOPS（WMMA 32 列小 M、mask 过滤损耗、F16 进出），注意力纯算 ≈ 1.3-1.9s；64K prefill 墙钟 = 65536/1003 ≈ 65.3s → **2-3%，与 nsys 的 3% 闭合**。prefill-parallel 侧无更新的 64K nsys（本机不可得，服务器 manifests/prefill-parallel 未回流），以 qsa-longctx 数据为 prefill 口径基准。

**结论**：1Cat 注意力 kernel 链的直接对标对象在我们这的总盘子 ≈ 3%（QSA 注意力）+ 1%（重建）≈ **65s 里的 ~2s**。即使把 #387 的 5.5× 全部兑现，端到端上限 **+2.4%**，远低于 ≥+10% 门禁。**这与 1Cat 完全相反**：他们旧 QSA 路径 55ms/层/rank × 12 层 × 8 chunk ≈ 5.3s/11.6s ≈ 45%（所以 #387 一炮 +30%）。我们是块稀疏 fork（B 路径），起点已相当于他们 #387 之后的位置。

### 结构性差距的真正所在（对比表）

| 维度 | 1Cat @64K（5777→6241） | 我们 @64K（1003，通信税后 ~1250 预期） |
|---|---|---|
| chunk/ubatch 深度 | 8000-8192/前向（双 chunk 16384/step，#315 README "long_prefill_token_threshold=8192"） | UB512（UB1024 首次 prefill workspace **4824MiB OOM**，TECH-FACTS 2026-10-06 [M]）→ 128 个浅前向/64K |
| 逐层通信 | TP4 NCCL + 图内批大摊薄 + P2P LL 直连（#992） | 岛内 TP3 每层 AllReduce，NCCL 覆盖 48.8%（prefill-comm-review trace-metrics.json 重算） |
| QSA 注意力 | 9.632ms/层/rank（#387 后，占 ~8%） | ~3%（已块稀疏，相对优势项） |
| MoE prefill | 4.235ms/8K 链（#390 后） | 15% 占比（H 路径 MOE_GROUPED 已转正） |
| GDN/prefill 杂项 | copy 融合链 #927-936 等已修 | 归入"杂项 32%"，未拆账 |

## B2. 可搬性判决表（A1-A4 × 直接搬/改布局搬/绑死）

| 项 | 判决 | 理由（绑死点/可搬点） |
|---|---|---|
| **#286 GQA-packed 宽 GEMM（dense）** | **架构绑死（对我们无 dense 长前缀层）** | 我们生产模型 Flash-Next 无 dense full-attn 层；其 M=Q×Hq 打包要求 Hkv_local=1 的整批 KV 复用，且 750MiB score workspace + cuBLAS/CUTLASS 契约绑 vLLM 连续 KV 布局。**思想可搬**（见 B3 设计 2） |
| **#666 FP32 QK/PV 累加** | **改布局后搬（作为 tile 形态知识，不是代码）** | 两条经验普适：① Volta HMMA FP32 累加免费——我们 fattn-wmma 的 QK FP32 累加（dst_meta float2 已是 FP32 行统计）不必为速度妥协；② "FP32 慢"曾是 tile spill 假象（128×256/64×64 定生死）。代码契约绑 CUTLASS/cuBLAS |
| **#666 张量几何准入（组大小判据）** | **直接搬（纪律层）** | `query.shape[2] == 6*key.shape[2]` 式判据 + capability 自检 + profiling 期 workspace 预算——我们 fork 加 QSA kernel 开关时应照抄这套准入纪律（替代 v70 式硬编码） |
| **#387 grouped Page4 QSA prefill** | **改布局后可搬，但收益 ≤+2.4%，不值得单独做** | 绑死点：`kGroupedVerifyHeads=6` 绑 TP4 本地几何（我们 TP3 需 8 头/64 行/4 m-tile 位图重参数化）；paged KV block_table 语义绑 vLLM；ABI/调度绑 qsa.py。可搬点：active_m_tiles 位图跳过、mask 并入 4-bit/块、planner 哈希去重——**但我们块稀疏 fork 已有同族机制**（B 路径 + qsa-attn.cu），增量小 |
| **v1.3.0 Split-D/N32** | **改布局后搬（中等价值，被 3% 天花板压住）** | 我们 fattn-wmma 的 D=256 模板每 warp 持全 256 维输出；Split-D（PV 输出维 4×D64 分 warp、QK 单次）是 Volta 上 D256 的正确形态。但同样是 3% 池子里的事 |
| **#390 Indexed MoE（免物化）** | **不需要搬（我们已是 indexed 形态）** | llama.cpp 的 `cont(ids) → MUL_MAT_ID` 语义本来就是按 expert 分组 + 行索引，从不物化 [tokens×topk, hidden]；vLLM 的 permute 物化是它自家历史包袱。**可搬的增量只有 #393 族**：SwiGLU 融合与 N 尾拆——llama.cpp MMQ MoE 亦已融合。我们 MoE 15% 的优化方向不在这 |
| **#754 批深度调度** | **思想直接搬（高价值，非 kernel）** | vLLM 调度器代码绑死（我们无 chunked-prefill 调度器），但一阶变量"**每前向 token 数**"同样支配我们：UB512 vs 1Cat 8000-16384。我们的 blocker 不是调度逻辑而是 **UB1024 的 prefill workspace 4824MiB OOM**（fattn 的 F16 KQ workspace 随 M×KV 膨胀）。修掉 workspace 即解锁批深度 → GDN 串行步数 ÷2、NCCL 摊薄、launch 摊薄——这是 1Cat 链里唯一能同时攻"通信税 + 杂项"两个大头的搬法 |
| **#315 79T 近似 exp2** | **不搬（负面教材）** | 真实模型张量上 FP16 累加溢出 + 采样 maxima 护栏复杂；我们门禁要求 parity 红线，近似路径性价比为负 |
| **#417 dispatch-only 组合** | **纪律可搬** | "已合格 kernel 的组合自动派发 + route-hit 证据"是我们 FIFO/H/B/E 开关体系下一步的产品化形态 |
| **#163 split-K decode** | **不搬（decode 侧，另一战场）** | 我们 qsa-attn.cu 已同族 |
| **#268 128-bit/页复用** | **思想可搬（小池）** | 我们 KV 是 GGUF 连续布局非 paged，页 ID 复用不适用；"合并转换组 + 对齐宽 load"在 fattn 的 K/V 装载路径可用，但属微优化 |

## B3. 设计：最值得的 1-2 项

**排序逻辑**：门禁是 64K prefill ≥+10% 且 8K parity 红线。3% 的注意力池子无论如何填不满门禁；批深度（#754 思想）同时攻 49% 通信 + 32% 杂项。故选：

### 设计 1（P0）：UB1024/UB2048 解锁——fattn prefill workspace 减容（#754 思想 + #286 状态机思想的杂交）

**问题**：`UB1024 首次 prefill workspace 4824MiB OOM`（TECH-FACTS [M]）。fattn-tile/wmma 族为 parallel_blocks split-K 物化 `f16_KQ` 中间与 dst_meta：workspace ≈ heads × M × KV_partitions 量级，M 与 KV 同涨时平方级膨胀。1Cat #286 的对应解法正是"**不物化所有 block partial——一个可复用 partial + FP32 累加器在线折叠**"（其 750MiB score workspace 靠 `PREFIX_TRANSPOSED_SCORE_WORKSPACE`、24K block、A-transform 融 softmax 三件套压住）。

**改动点**（llama.cpp 侧，估计 +200-400 行）：
1. `fattn-common.cuh` 的 workspace 尺寸公式重审：把 `parallel_blocks` 的 split-K partial 物化改为**流式两段**——(a) 单份复用的 per-partition (m,l) FP32 状态（即 fattn 已有的 dst_meta 在线合并，等价 #286 的 prepare_prefix_update）；(b) KQ 中间 buffer 按 partition 分批发射（每 partition 用完即被下一覆写），workspace 从 `P × M × KV` 降为 `M × KV_tile`。
2. 岛 3,3 下的放置：workspace 每卡 1 份（fattn 是 per-device scratch，无跨岛问题）；QSA 层 UB1024 时预算目标 ≤ 2.4GiB/卡（16GB 卡上与 132K KV 池共存，参考现役 KV 6.6GB/卡上限配方）。
3. 环境变量门控：`GGML_FATTN_WS_STREAMING=1`，默认关；fallback 到现路径。

**伪码**（fattn 调度层，概念）：
```
for each flash_attn_ext call with ne[1] > 512:          # prefill 大 ubatch
    ws = device_workspace(key=(heads, M, KV))           # 命中则复用（1Cat 的 per-device cache + event 思想）
    if ws.bytes > budget:                                # OOM 预检（1Cat 的 cudaMemGetInfo + headroom 门）
        for p in 0..parallel_blocks step CHUNK:          # 分批流式
            launch fattn_kernel(partition_slice p..p+CHUNK, ws_reuse=ws)
            launch merge_meta(ws_reuse, state)           # FP32 在线折叠（等价 #286 步骤 3/4）
        drain
    else:
        legacy_single_launch(all_partitions)
```

**验证方案**：
- parity 红线：8K/512 全 512-token 逐位（现役 matrix_runner 口径）；加 64K×2 长上下文回归（现有 case 库）。
- 门禁：64K prefill UB512→1024 A/B（同频同功率口径记录——1Cat #666 的功率教训）；指标 PP ≥+10%。预期来源分解：GDN 串行步 ÷2（128→64 前向）、NCCL 调用次数 ÷2 且单次批大摊薄、launch/sync ÷2。**若 NCCL 单次开销不随批线性化（ rendezvous 协议固定成本主导），收益会低于预期——先跑一次 UB256 vs UB512 的斜率外推（零代码，`-ub` 参数即可），斜率 ≥ +15%/倍频才立项**。
- 风险：FIFO/图缓存变体数 ×2（TECH-FACTS §四的容量教训：变体 LRU 已知非瓶颈，可容纳）；GDN 状态边界对齐（`LLAMA_PREFILL_KV_PAD` 兼容）。

### 设计 2（P1，通信税修复后启动）：fattn-wmma 的 Split-D 化 + 头折进 M 维（v1.3.0 + #286 思想合并，模板参数化而非硬编码）

**触发条件**：通信税修到 ~1250 tok/s 后，重跑 64K nsys；若 QSA attention+重建 合计 ≥8%（占比会自然膨胀），此项立项；否则继续搁置（3% 时无门禁可达性）。

**改动点**（ggml/src/ggml-cuda/ 内新文件 `fattn-wmma-splitd.cu`，不动旧路径）：
1. **M 维头堆叠**：每 CTA 处理 `M_TILE=64 行 = 8 query × 8 头`（岛 TP3 几何）或 `16 query × 8 头=128 行`（对齐 WMMA 16 的倍数即可，**从张量几何推导而非硬编码**——#666 纪律）；Q smem 布局 [64, 256]；K/V 每 CTA 单份装载（对比现状 32 列 × 8 头 = 8 份）。
2. **Split-D**：QK^T（WMMA 16×16×16，K=D256 全宽一次）得 P 后，PV 按 D 输出拆 4×D64 分派 4 warp 组，共享同一 P——砍 3/4 的 QK 重复（相对 naive 每 warp 独立算）。
3. **QK FP32 累加 + 128×256 TB 形态**（#666 经验）；mask 仍在 score 后逐行施加（保 parity：我们的稀疏语义是行级精确集，不能学 1Cat 的采样 maxima——**他们是 dense 近似路线我们是精确路线**，FP32 累加免费拿精度即可，不做任何近似）。
4. 准入：`Hq_local % 8 == 0 && D == 256 && ne[1] >= 256 && type == F16`；`GGML_FATTN_WMMA_SPLITD=1` 门控默认关。

**伪码**（kernel 主体骨架）：
```
CTA: 64 rows (8 q × 8 heads, M-fold) × KV 扫描（块稀疏块表驱动）
  smem: Q[64][256] (FP16, 一次装载)
  loop over selected KV tiles (来自 fork 块稀疏块表):
    smem_tile K[64][256], V[64][256]                      # 单份
    warp-group g in 0..3:                                  # 8 warp = 4 组 QK + 4 组 PV 交错（Split-D）
      QK:  wmma(16x16x16) Q_tile × K^T → P (FP32 acc, 逐行 mask 施加, online softmax 更新 m/l)
      PV:  wmma(16x16x16) P' × V[:, g*64:(g+1)*64] → O_g   # 输出维分片，P 全体共享
    __syncthreads(); next tile
  写 O[64][256]（FP32 累加器 / l 归一）+ dst_meta
```

**验证**：kernel 级对拍 fixture（fork 现有 27+2 形状 × 12 replay 基建直接复用，TECH-FACTS R19）；8K/64K parity 红线；64K 门禁按立项时的占比折算（若占 8%，kernel 2× → 端到端 +4%，需与其他项打包过门）。**单独立项不承诺 ≥+10%——这是与任务门禁的诚实对齐**。

### Strata v70 判死案对照（任务 B4）

- **当时死因**（FLASHNEXT-FULL-EXPERIMENT-REPORT L155）：v70 QSA prompt attention kernel 硬编码 **G=12** 组几何；我们 TP 下 **gqa=2**，打包浪费 6×，吞吐塌 1/8，叠加目标只占 prefill 3% → 判死正确。
- **1Cat 解了这个问题吗？——解了一半，且解法是纪律不是代码**：#666 的准入判据 `Hq_local == 6 × Hkv_local`（组大小不变性）+ #387 的 per-KV-head 组处理证明他们**从不假设全局组几何**：kernel 按本地张量几何模板化（`kGroupedVerifyHeads` 是 TP4 契约常量而非模型假设），几何不符就 fallthrough。v70 的死因不是"GQA 打包这个思想错"，而是**把一个模型的几何写死成 kernel 常量**。若当时有几何自检 + 按我们 8 头/4KV 重参数化，G=12→G=2 的 6× 浪费本可避免——但 3% 占比的第二死因仍在，判死结论不变。
- **对设计 2 的直接输入**：头折进 M 维的行数从 `heads × q_rows` 运行时推导（我们 8×8=64），不写死任何模型的组大小；这同时是对 v70 尸检的工程化回应。

## B4. 与既有战线的接口（避免重复立项）

1. 本设计 1（UB 解锁）与 prefill-comm-review 的方案 A（六段层流水）/方案 B（per-phase 混合调度）**正交且前置**：批深度是所有调度方案的分母。层流水 V1 已失败（真实 token 37 分岔，已回退，TECH-FACTS 2026-10-06 [M/X]），UB 路线不依赖它。
2. Codex 通信税组合（FIFO+P2P，1003 生产）与本设计叠加顺序：先 UB 斜率外推（零代码）→ 通信税终局（1250 预期）→ 重估占比 → 决定设计 2 是否立项。
3. 1Cat 线（super-ac922 vLLM fork）不适用本报告结论——那边是 TP4 dense+QSA 混合，注意力 kernel 直接 merge 上游即得。

## 附：证据锚点索引

| 事实 | 位置 |
|---|---|
| #286 kernel 全文（1359 行） | 本地 `cmake/patches/sm70_flash_attn_d256_gqa_arch.patch` L87-1452 |
| #286 生产入口/workspace/event 串行化 | 同上 L1148-1420（`Sm70GqaScoreWorkspace`、750MiB、128MiB headroom） |
| 79T/FP32 现役 kernel + 护栏 | 本地 `csrc/attention/sm70_79t/prefill.cu`（L88-102 Taylor、L462-468 FP32 acc、L1939-1965 cuBLAS FP32 QK、L424-448 TB 形状）+ 同目录 README.md/VALIDATION.md |
| #387 kernel/planner diff | `gh api pulls/387/files`（flash_decode_paged.cu +661）；合并版本地同文件 L1835-1841（常量）、`vllm/models/qwen4_exp/nvidia/ops/qsa.py` L1776-1868（调度+pad fix） |
| #390 idxs 机制 | `gh api pulls/390/files`（awq_sm70_gemm.cu：`desc_A.idxs`；moe_permute_unpermute_kernel.cu：`buildMoePermuteMetadataKernel`） |
| #754 scheduler diff | `gh api pulls/754/files`（scheduler.py `_mamba_block_aligned_split` retained_boundaries） |
| v1.3.0 机制 | release notes + 本地 `cmake/patches/sm70_flash_attn_d256_{pipeline,k_pingpong,splitkv3}.patch` |
| 我们 fork dispatch 链 | 上游镜像 `ggml/src/ggml-cuda/fattn.cu` L83-176、`fattn-wmma-f16.cu` L138-159、`common.cuh` L221-224 |
| 我们 prefill 拆分/UB OOM/生产数字 | `flashnext-llama-experiment/TECH-FACTS.md` §一、2026-10-06 节；`reports/prefill-parallel-proposals.md` §一；`reports/prefill-comm-review-20261006/REVIEW.md`（NCCL 48.839% 重算） |
| v70 判死 | `reports/FLASHNEXT-FULL-EXPERIMENT-REPORT.md` L155 |
