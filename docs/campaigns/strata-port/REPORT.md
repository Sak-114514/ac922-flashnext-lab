# Strata Volta 特化移植评估报告

日期：2026-10-06 | 主机：ac922 (6×V100 SXM2 sm_70, CUDA 12.4) | fork：/home/debian/ac922-llama-next/sources/llama-v100-v3 @ ac922-flashnext-v100 (7fe450e+，生产树)
Strata 上游：本地 strata-engine origin/main，目标 commit 06a90a2/5fa9e7f/800ca98（v70 attention）+ 3f81bce/ff6f9f1（BF16→FP16 GEMM，PR #655/#540/#395）

## 一句话结论

**协调者"我们 prefill 侧 cuBLAS FP16 路径已等效"的说法不成立（对 BF16 权重）**：我们的 BF16 大 batch（prefill）GEMM 在 sm_70 走的是 F32 SIMT CUDA-core 路径（且每次调用整权重 BF16→F32 转换），正是 Strata #655/#540 修的同一个病。**最小修（27 行 env 门控补丁）已按分支隔离纪律完成 A/B：8K prefill +6.5%、64K prefill +6.2%、decode 不变、512 token 逐位 parity 全过——两项门禁全过，列为合入候选**。v70 tensor-core QSA prompt attention 判死（架构不可直接搬 + 收益上限 <2% prefill + 我们的 6 卡 TP 使 A-tile 利用率崩到 2/16）。

---

## 1. 阶段 1：差距判决表

### 1.1 移植目标 #2：BF16→FP16 tensor-core prefill GEMM（3f81bce + ff6f9f1）——**我们有病，Strata 有药，已治**

| 项 | 证据 |
|---|---|
| **协调者疑问的判决** | **"已等效"= FALSE。** `ggml-cuda.cu:1993-2005`（`ggml_cuda_mul_mat_cublas`）：BF16 && `!fast_bf16_hardware_available(cc)`（sm_70=V100 为假：`common.cuh:334-337` 要求 cc≥AMPERE）&& NVIDIA && `src1->ne[1] > 8`（Volta 阈值）→ `compute_type = GGML_TYPE_F32`。prefill 恒 ne[1]>8 ⇒ **全部 BF16 投影 prefill 走 F32 路径** |
| F32 路径实际内容 | `mul_mat_cublas_impl<GGML_TYPE_F32>`（:1783+）：**每次调用**把整个 BF16 权重转成 F32（:1832-1845，pool alloc + convert kernel）再 `cublasSgemm/GemmEx(CUDA_R_32F, COMPUTE_32F)` —— V100 无 FP32/TF32 tensor core ⇒ 纯 CUDA core ~13 TFLOPS |
| "FP16 路径"存在于何处 | 只对 **F16 权重**生效：`compute_type=F16` + `prefer_f32_output`（Volta, :1881-1884）→ `cublasGemmEx(F16 入, F32 出, COMPUTE_32F)` = HMMA tensor core。**BF16 权重从不进这条路**（上述分支先行改写成 F32） |
| Strata 的药 | `src/prefill/gemm.cu:381-516`：sm_7x 无 BF16 TC ⇒ cuBLAS BF16 落 SIMT fp32（magma_sgemmEx）。修法 = 在线 BF16→F16（权重每次调用转一次、激活按 32MiB 行切片转，`bf16_to_f16_kernel` 含 **±65504 钳制** #540）+ FP16 GEMM（F32 累加）。V100 prompt 提速（#540：BF16 SIMT 占 V100 prompt ~10%）。beta=1 余数 GEMM 与 N==1 GEMV 留 cuBLAS |
| D 线交叉验证 | D 线离线转换（`convert_bf16_f16.py`，同一批张量：HC mixers/indexer q,k/PLE/ssm/router，见脚本 DEFAULT_PATTERNS）实测 8K PP 532.6→544.6 **+2.25%**、parity=True，仅因 decode -4.7% 判死（manifests/zcode-d-f16）。即：**病的存在和药的数值安全性（F16 化后 512 token 逐位一致）都已离线证明过**，只是当时把 decode 一起毒了 |
| 微基准（V100, 本次实测） | 同 shape 三路对照（/tmp 微基准，产物在 artifacts/strata-port/bf16gemm_bench.sm70）：完整 [2560×2560]×T=2048：F32-SIMT 12.7-14.0 TF vs F16-TC 66.9-70.5 TF = **5.0-5.3×**；生产 6 卡 TP 切片 [427×2560]：T=2048 时 11.0→20.0 TF = **1.8×**，T=8192 时 12.3→23.9 TF = **1.9×**；router 切片 [86×2560] 仅 1.2-1.3×（太小，两路都不饱和）。relerr(B vs A)=1.84e-4（F16 激活舍入，与 D 线同类，parity 已证可过） |
| 与 Strata 的差异点 | 我们沿 fork 现有 traits 机制（`ggml_get_to_fp16_cuda(BF16)`=convert.cu:600 已存在；nc 变体 :682 也有）→ 补丁面更小；**缺 ±65504 钳制**（Strata #540）：权重有界、D 线同语义转换已过全模型 parity，合入候选可后补 |

### 1.2 移植目标 #1：Volta QSA prompt attention（06a90a2 + 5fa9e7f + 800ca98）——**有病同源，但药不能直接搬，且剂量不够 → 判死（存档理由如下）**

| 项 | 证据 |
|---|---|
| 病同源确认 | 我们 prefill QSA 确实是 Strata 修复前的形态：`qsa-attn.cu:4` "the same kernel serves decode and prompt batches"——块稀疏 Q8_0 KV、纯 FP32 FMA、每 lane 一个 (head,cell) 打分 + smem 转置归约。Strata commit 自述其旧路径"one query at a time FP32 FMAs"占 32K prompt GPU 时间 19-25% |
| Strata 的药（读完 1476 行全文） | `prompt_attn_v70_kernel`（:792-1084）：mma.sync.m8n8k4（每 warp 4 个独立 8×8×4，QP 划分 :744-748）；q/p 双 FP16 hi+lo 分半（q 预缩放到 ~2^14 防 lo 进 subnormal :811-829）；q.k 每 warp 64 维=1 个 int8 scale 组，部分和固定顺序累加（确定性）；在线 softmax；int8 码经 mantissa 技巧精确进 FP16（i8x2_to_h2 :65-71）；K8V4 的 V 在 gather 处反量化成 FP16（mode 3）。官方 V100 实测：int8 KV 每 2048-chunk 3.0×，fp16 1.3×；32K prompt 26s→22s |
| **不可直接搬 #1：A-tile 利用率** | v70 硬编码 **G=12**（每 KV 头 12 个 query 头，16 行 MMA tile 用 12）。模型侧完全同族：qwen4exp = 24 q / 2 kv / head_dim 256 ⇒ 单卡 GQA=12 正好。**但生产是 6 卡 TP**：每卡 4 q 头 / 2 kv 头 ⇒ **gqa=2**（qsa-attn.cu:320 支持 24..1 全档）。直接搬 = 16 行 A tile 只用 2 行 ⇒ qk 与 pv 的 MMA 吞吐 ~1/8，3× 优势塌缩到 ~1.2-1.5×。要救必须重排成"8 query × 2 头"装 tile——那是重设计不是移植 |
| **不可直接搬 #2：KV 格式** | 我们的 KV 是 **Q8_0（32 元素一组、每组一个 fp16 scale）**；Strata int8 模式是 64 维一组。v70 的不变量"warp=64 维=恰好 1 个 scale 组"（S.ks[CH][4]，part[4][…]）对我们破裂：qk 侧每 warp 要拆两个 32 维半程分别乘 scale；pv 侧 V-scale 无法折进 p（同一 warp 的 64 维跨两组）——须像其 mode 3 那样 gather 时把 V 反量化成 FP16，或按 v1 kernel 的 NG=8 结构（:262, :341 GPW=NG/4）重做 v70 没有 的泛化 |
| **不可直接搬 #3：选择结构** | Strata = pools + page_table 间接寻址 + per-query ids/steps；我们 = cache 直读 + per-token 块表 (ids I32) + F16 cell mask（qsa-attn.cu:8-14）。gather 循环要改写；n_split 分裂逻辑仅 decode 用（prefill blocks0 大 ⇒ n_split=1，无碍） |
| **收益上限 < 门禁** | qsa-longctx/REPORT.md §1.3（64K nsys）：prefill 切片中 `qsa_attn_kernel` 936ms/~3%（12s 切片），QSA 重建机器 ~1%，"QSA 全 ctx 扫描占 prefill 步 ~10% 可砍半 → 仅 ~2-5% prefill 收益"且那 10% 的大头是**选块机器/全 ctx 评分**（P2 钳制 kvlen 才是对症的药），attention 数学本体只有 ~3%。即便 3× kernel：64K prefill ~+2%，8K 更小 ⇒ **8K ≥+3% 门禁过不了**。Strata 自己 +15-18% 是单卡 32K 全 ctx 场景，与我们 6 卡 TP + top-k=2048 块预算的形态不同 |
| parity 风险 | v70 求和顺序与 FP32 kernel 不同（确定性但不逐位）。Strata 仅验证"一条 32K prompt greedy 输出一致"；我们红线是 512 token 逐位 × 测试矩阵。attention 数值变化比 GEMM 换路（权重/激活精确表示）风险高一档 |
| 判决 | **判死/存档**。触发复活条件（任一）：(a) qsa-longctx P2（kvlen 钳制）先落地且 64K prefill 成为绑定瓶颈；(b) 部署形态变化（单卡/少卡大 gqa）；(c) 上游出现 gqa=2 友好的 tile 打包参考实现。工作量如实评估：核心 mma 机构（~200 行）可搬，KV 格式/头打包/gather 是重写（合计 400-500 行 + parity harness，3-5 天专注迭代） |

### 1.3 判决汇总

| 对照项 | 判决 | 一句话 |
|---|---|---|
| BF16 prefill GEMM 走 FP16 TC | **我们有病 Strata 有药 → 已移植，门禁全过** | 27 行 env 门控补丁，8K +6.5% / 64K +6.2% / TG 持平 / parity 全过 |
| "我们 prefill cuBLAS FP16 已等效" | **不成立**（对 BF16 权重；对 F16 权重成立） | ne[1]>8 的 BF16 在 sm_70 被 :2000-2005 改写为 F32 SIMT + 每调用全权重转换 |
| QSA prompt attention tensor-core 化 | **不可比（拓扑+格式差异）且收益<门禁 → 判死存档** | 6 卡 TP gqa=2 使 v70 的 G=12 tile 利用率 2/16；Q8_0 32 维组破坏其 warp=scale 组不变量；kernel 仅占 prefill ~3% |
| decode 侧任何改动 | 无需（Strata 两个特化均不触及 decode） | 本次补丁 ne[1]>8 门限 = decode 形状（≤8）原样保留，实测 TG 持平 |

---

## 2. 阶段 2：BF16→FP16 prefill GEMM 移植（做了，门禁全过）

### 2.1 补丁（合入候选）

`artifacts/strata-port/sm70_bf16_prefill_f16.arm-only.patch`（27 行）：`ggml_cuda_mul_mat_cublas` 的 NVIDIA 分支内，当 `cc ∈ [Volta, Ampere)` 且 `src1->ne[1] > 8` 且 `GGML_SM70_BF16_PREFILL_F16=1` 时 `compute_type = GGML_TYPE_F16`（否则原 F32 逻辑逐字保留）。下游全部复用 fork 现有机制：`ggml_get_to_fp16_cuda(BF16)`（convert.cu:600）在线转权重、`prefer_f32_output`（Volta）→ `cublasGemmEx(F16 入, F32 出, COMPUTE_32F)` HMMA。默认 **0=off**（A/B 期间控制臂与生产行为逐位一致）。

### 2.2 门禁数据（同一二进制，仅 env 差异，systemd 单元隔离，1×8K/512 与 1×64K/512，seed 1234 temp 0）

| 口径 | arm0 (off) | arm1 (on) | Δ | parity |
|---|---|---|---|---|
| 8K PP 热（run1/run2） | 790.87 / 789.85 | 841.84 / 841.19 | **+6.5%** | **512/512 ×2 逐位** |
| 8K TG | 112.07 / 111.22 | 111.79 / 112.01 | ±0.2%（噪声内） | — |
| 64K PP | 722.78 | 767.28 | **+6.2%** | **512/512 逐位** |
| 64K TG | 94.83 | 94.79 | 持平 | — |

- 热运行间重复性：同臂两 run 相差 <0.15% ⇒ +6% 显著非噪声。
- **门禁：8K prefill ≥+3% ✓（+6.5%）；parity 逐位 ✓** → 列为合入候选。
- 为什么比 D 线离线 +2.25% 大：D 线测于 10-01 旧二进制（PP 基线 532，NCCL/MoE SIMT 未修）；现役链（岛+H+P2P，基线 ~790）修掉了其它瓶颈后 BF16-SIMT 份额相对放大；且本补丁覆盖**所有** ne[1]>8 的 BF16 GEMM（含 MTP draft prefill ubatch=32、运行期生成张量），不依赖离线文件转换的张量清单。

### 2.3 合入建议与残余风险

1. **建议转正**：翻默认值（`GGML_SM70_BF16_PREFILL_F16` 默认 on，`=0` 留作逃生门；同 `GGML_SM70_BF16_GEMV` 的模式），生产配方无需加 env。
2. 后补 Strata #540 的 **±65504 钳制**（现 convert_unary 直接 `__bfloat162float`→half，>65504 成 Inf）：权重有界 + D 线全模型 parity 已过，非阻塞，合入时一并加更稳妥。
3. 未测口径：2×64K 多路（预期同向，fair-share 下 prefill 占比更高）、2×/4×8K 并发。转正前按惯例补一轮短双路验收即可。
4. 隔离序列（备份→补丁→构建→A/B→还原→重构建→sha 校验 41bc2b28… OK→GPU 清零）全记录在 ac922 `artifacts/strata-port/SEQUENCE.md`；生产树与二进制已还原原样。

---

## 3. 附：微基准原始表（V100 SXM2, CUDA 12.4, 独立 .cu 无树改动）

| shape (N×K, T) | A: F32-SIMT (cvt+GEMM ms) | B: F16-TC (ms) | A TFLOPS | B TFLOPS | speedup | relerr |
|---|---|---|---|---|---|---|
| 2560×2560, T=2048 | 0.056+2.053 | 0.401 | 12.7 | 66.9 | 5.26× | 1.8e-4 |
| 2560×2560, T=2048 | 0.057+1.864 | 0.381 | 14.0 | 70.5 | 5.04× | 1.8e-4 |
| 427×2560, T=512 | 0.012+0.139 | 0.102 | 7.4 | 10.9 | 1.48× | 1.8e-4 |
| 427×2560, T=2048 | 0.013+0.395 | 0.224 | 11.0 | 20.0 | 1.82× | 1.8e-4 |
| 427×2560, T=8192 | 0.013+1.443 | 0.750 | 12.3 | 23.9 | 1.94× | 1.8e-4 |
| 86×2560, T=2048 | 0.006+0.145 | 0.114 | 6.0 | 7.9 | 1.33× | 1.9e-4 |

（A = 现产路径复刻：BF16→F32 全权重转换 + cublasGemmEx F32/COMPUTE_32F；B = Strata 路径复刻：BF16→F16(±65504 clamp) + F16 入/F32 出/COMPUTE_32F。TP 切片 N=427/86 按 6 卡 1,1,1,1,1,1 张量切分。）

## 4. 关键文件索引

- 我方：`ggml/src/ggml-cuda/ggml-cuda.cu:1993-2005`（病根）、`:1783-1950`（cublas impl + prefer_f32_output）、`convert.cu:600/682`（BF16→F16 转换已存在）、`qsa-attn.cu`（FP32 共用 kernel）、`common.cuh:334`（bf16 TC 门）
- Strata：`src/kernels/cuda/qsa_prompt_attn.cu:743-1122`（v70）、`src/prefill/gemm.cu:381-516`（#655/#540/#395 路径）
- 证据：ac922 `artifacts/strata-port/`（A/B JSON×8、补丁、序列、微基准）；本地 `qsa-longctx/REPORT.md` §1.3；`manifests/zcode-d-f16/`（D 线）
