# ik_llama.cpp 三 kernel 结构移植价值评估（standalone 微基准实测判决）

日期：2026-10-04 · 评估对象：`/home/debian/ik-llama/`（ik_llama.cpp，IQ 量化格式作者 fork，shallow clone @5f89bfc）vs 我们的 fork `/home/debian/ac922-llama-next/sources/llama-v100-v3`（两处源码零修改）
硬件：ac922 POWER9 + V100-SXM2-16GB sm_70（本批全部用 GPU 0/1，跑完已释放，6×0 MiB）
方法：照 qmmvq-audit（逐字节提取 kernel+模板+vecdot，真实形状，stream 循环计时，≥24MB 权重复制品轮转防 L2 命中，best-of-3；cudaGraph capture 本机不可用——沿用审计已证方法）。fork 侧基线为**同日同 GPU 重跑**（bench-fork，含审计原 binary 的全部用例），非跨session引用。

**结论先行：三个候选只有一个值得移植。**
1. **fused_mul_mat_vec_q（MoE gu+gate 单 launch 结构）——值得，且是本次最大单点收益**：在与 R17 部署完全一致的形状上，ik 的"1 warp 算 2 行 × 全部 token，gate/up 共享 x 读"结构比 fork 现行 mul_mat_vec_q_moe（warp-per-token）**快 1.4-2.3×**，甚至比 fork 自制 iq2s_volta 还快 1.2-1.3×。收益池折算 R17 MoE gu+gate 全家族 ≈ 省 265ms/3130ms trace（估算），现实落地预期 **+4-5% decode 速度**（估算）。
2. **`__launch_bounds__(nwarps*32,1)` + 自由寄存器 + nc≥4 rows=2——fork 已有等价物且更快**：fork mmvq.cu:646 同款 launch_bounds、:619-643 同款 rows=2（还多了 SK/TALL 变体）。实测同结构下 fork dense 比 ik 还快 1.04-1.07×。**无可移植内容**。
3. **ik 的 MMQ（stream-K，V100 强制启用路径）——明确不移植**：nc5/8 下比 fork MMVQ 慢 1.8-2.6×（33-56µs vs 14-34µs），nc32/64 直接塌到 31 GB/s。ik 自己在 V100 上强制 MMQ 的理由（避免 fp16 cuBLAS 临时矩阵回退，mmq.cu:255-257 注释）在我们的 fork 里不成立（fork 有 pre-Turing MoE grouped-MMQ 路径，dense decode 走 MMVQ）。

---

## 1. 静态差异清单（阶段一）

### 1.1 块结构与枚举兼容性（提取前置检查，python 逐 struct diff）

- **共享 IQ/K 格式 block 结构逐字节一致**：`block_iq2_s/iq2_xxs/iq2_xs/iq3_s/iq3_xxs/iq4_xs/q6_K` 全部 IDENTICAL（ik ggml-common.h vs fork ggml-common.h）。
- `block_q8_1/q4_K/q5_K`：union 写法不同（GGML_SCALE_TYPE1 vs 命名聚合），**字节布局一致**（36B/144B/176B 不变）。
- ggml_type 枚举：0-23 段（我们全部格式）两边数值相同 → bench 可各自独立编译、共享权重缓冲。
- VDR 表：**全部相关格式 VDR_*_MMVQ 数值一致**（IQ2_XXS/XS/S=2、IQ3_XXS/S=2、IQ4_NL=2、IQ4_XS=4、Q6_K=1）。

### 1.2 mmvq 结构差异（ik `mmvq-templates.cuh` 482 行 vs fork `mmvq.cu` 1855 行）

| 项 | ik（行号） | fork（行号） | 判定 |
|---|---|---|---|
| 融合 gu+gate+SwiGLU+bias 单 kernel | `k_fused_mul_mat_vec_q` **:152-280**，global `fused_mul_mat_vec_q` **:305-330**（ids/nb-stride 支持任意 ne2 通道） | 有（`mul_mat_vec_q<has_fusion>` :645-891 dense + `mul_mat_vec_q_moe` :897-1042），但 MoE 为 **warp-per-token**（block=(32,nc)，每 warp 独立重读重解码 x） | **结构实质不同**，见 §2 bench a |
| MoE nwarps | **:446-448**：`ne2 >= 2 → nwarps=1`（1 warp 算 2 行 × 全部 token；dense ne2<2 → nc≤4?4:2） | MoE block=(32,nc) 每 warp 1 token；dense VOLTA 表 :480-491 | 关键差异 |
| rows_per_block | :86/:171 `ncols_y<4?1:2`（编译期） | :619-643 nc2-8 全 2 + SK(4 rows)/TALL 变体 + IQ4_NL rpb8 | fork 覆盖更全 |
| launch_bounds 自由寄存器 | **:283-286** `__launch_bounds__(nwarps*WARP_SIZE, 1)` + 注释 "tell the compiler to use as many registers as it wants" | **:646** 同款 `__launch_bounds__(calc_nwarps(...)*warp, 1)` | **已等价** |
| SwiGLU 变体 | SILU/RELU/GELU/SWIGLU_OAI(带 bias) :245-275 | SWIGLU/GEGLU/SWIGLU_OAI/SWIGLU_CLAMP（glu_op 枚举） | 等价面 |
| 图级融合分发 | ggml-cuda.cu:2505-2601 图模式匹配（QKV+bias 三连、gate/up+SwiGLU、nc==1 才 fusion） | fork 走 tensor 级 fusion args | 移植时不需要（fork 已有 args 通道） |

**ik 生产路径事实**（移植面判断依据）：ik 的 fused MoE kernel 在 ik 自己的 host 分发里**只在 nc==1 启用**（ggml-cuda.cu:2877-2923 "Fast TG path"：`local_src1.ne[1]=1`、`nb[2]=0` 共享 y）；nc>1 的 MoE 落到 `ggml_cuda_mul_mat_q_id`（MMQ_ID）。**kernel 本身原生支持任意 ncols_y×ne2**（j 循环+per-channel expert），本 bench 用 kernel 原生语义在 nc2/5 下实测——这正是"移植结构"的形态。

### 1.3 vecdotq 解码差异（对我们格式）

- **IQ3_S、Q6_K、IQ4_XS：逐字节 IDENTICAL**（除 iq4xs 表传参写法，数学相同）。
- **IQ2_S / IQ2_XS / IQ2_XXS / IQ3_XXS：sign 处理不同**——
  - ik：`ksigns64[128]`（1KB 常量表，ggml-common.h:888）直接给出两半 sign 掩码，剩一个 `__vsub4`（ik vecdotq.cuh:923-970/849-888/974-1006）；
  - fork：`unpack_ksigns`（popc+xor 位运算构造广播）+ `__vcmpne4` + `__vsub4`（fork vecdotq.cuh:97-103,1045+）。
  - fork 的 iq2s_volta 注释（fork mmvq.cu:1088-1091）自认 "__vcmpne4/__vsub4, which Volta emulates" 是 sm_70 痛点 → ik 少一半模拟指令。**但见 §2：ik 通用核已赢，此项为次要**。
- VDR 全一致（见 1.1）。

### 1.4 MMQ 差异（ik mmq.cuh 4770 行单文件 vs fork mmq.cu+mmq-config-*.cuh）

- ik `mul_mat_q` **:4386-4406** stream-K（arXiv:2301.03598）：`use_stream_k = cc >= CC_VOLTA && cc < CC_OFFSET_AMD`（:4585）→ **V100 grid = (nsm=80,1,1)** + fixup kernel（:4462+）。
- `MMQ_ITER_K=256`（:19）、`MMQ_NWARPS=8`（:20）、Volta `mmq_y=128`（:159-161）、`__launch_bounds__(WARP_SIZE*nwarps,1)` for `__CUDA_ARCH__>=CC_VOLTA`（:4379-4385）——**ik 在 MMQ 里也用了同款 launch_bounds 手法**。
- 分发门槛（ik mmq.cu:239-260）：`cc < CC_TURING → return true`——**Volta 上 MMQ 对所有 batch 强制启用**，注释明言 "On Volta, large-batch quantized matmuls otherwise fall back through fp16 cuBLAS temporaries. Keep using MMQ for pre-Turing NVIDIA."（:254-257）。fork（mmq.cu:328-346）：Volta dense 仅 ne11<64 且实际被 MMVQ(≤8) 截胡；MoE pre-Turing 才强制 grouped-MMQ。
- fork 的 MMQ 为重构版（ampere config 用于 Volta、无 stream-K）——两版结构不同源。

### 1.5 git log 取证

本地 clone 为 **shallow（仅 1 commit 5f89bfc）**，无历史可挖 Volta 调优提交。Volta 证据全部来自源码注释/条件（§1.4 引行号）：stream-K 的 Volta 条件、mmq_x_max 的 Volta 注释（mmq.cuh:118-130）、should_use_mmq 的 Volta 注释。**如需提交级证据需重新 full clone（本地未做，属可选项）**。

---

## 2. 微基准（阶段二，全实测）

### 2.1 提取与构建（工件见 §5）

- ik 侧（`/home/debian/ik-eval/bench/`）：`vecdotq.cuh/mma.cuh/mmq.cuh/mmq_kt_tail.cuh/iqk_cuda_common.h/ggml-common.h/mmvq-args.h` **逐字节副本** + `mmvq_ik.inc`（mmvq-templates.cuh 14-482 行 sed 提取）+ `quantize_ik.inc`（quantize.cu 85-333 行）+ shim common.cuh（cc=700/nsm=80/smpbo=96KB）。仅有的非逐字节改动：副本加 `#pragma once`、mmq.cuh 删 42 行 `extern DECL_MMQ_CASE`（单 TU 实例化需要）、quantize 提取去掉依赖 plain-q8_1 kernel 的一个 host 函数——均不影响 kernel 代码。
- fork 侧（`/home/debian/ik-eval/bench-fork/`）：qmmvq-audit bench 原样复制 + 加 N640/nc2 用例。
- CPU 参考（`bench_ref.cpp`）：ik ggml-quants.c 的 `dequantize_row_*` 逐字节提取（host 表分支）。
- 构建：`nvcc -arch=sm_70 -O3 -use_fast_math -std=c++17`（=fork 旗标）。
- MMQ y 布局实证：从 `quantize_mmq_q8_1`（block(chunk,row) 在 chunk*kx1+row，k-major）推导 `mmq_args.stride11 = ne11`，CPU 参考 1e-7 级验证通过（顺带修复：y 缓冲须按生产补 64×block_q8_1_mmq 填充，否则 tile 过读触发 illegal access）。

### 2.2 判决表 A —— MoE gu+gate 融合（10 专家，K=2560；up+gate 权重字节）

| 形状/格式 | fork 现行（mul_mat_vec_q_moe / iq2s_volta） | ik fused（nwarps=1 结构） | **ik 加速比** |
|---|---|---|---|
| **N640×2560 nc5（任务书形状，全专家宽）** | | | |
| IQ3_S | 157.4 µs | 69.3 µs | **2.27×** |
| IQ2_XS | 132.7 µs | 68.9 µs | **1.93×** |
| IQ2_XXS | 110.0 µs | 61.3 µs | **1.79×** |
| IQ2_S（vs fork 自制 iq2s_volta 86.1） | 86.1 µs | 65.0 µs | **1.32×** |
| **N224×2560 nc5（R17 部署形状，岛内 TP3）** | | | |
| IQ3_S | 53.9 µs（R17 原位 64.9） | 28.9 µs | **1.86×** |
| IQ2_XS | 48.2 µs（R17 55.7） | 28.3 µs | **1.70×** |
| IQ2_XXS | 40.4 µs（R17 50.5） | 28.6 µs | **1.41×** |
| IQ3_XXS | 43.4 µs*（R17 53.7） | 27.1 µs | **1.60×** |
| IQ2_S（vs iq2s_volta 36.2） | 36.2 µs（R17 31.3） | 29.4 µs | **1.23×** |
| **nc2（MTP 2-token 窗）** | | | |
| IQ3_S N640 | 67.4 µs | 39.4 µs | **1.71×** |
| IQ3_S N224 | 26.8 µs | 17.0 µs | **1.58×** |
| IQ2_S N640（vs volta 42.0） | 42.0 µs | 39.0 µs | 1.08× |
| IQ2_S N224（vs volta 17.0） | 17.0 µs | 16.8 µs | ~1.01×（平） |

\* fork nc5 数字为本次重跑；nc2/xxs 系列同为本次。ik 侧可复现性：跨 run、跨 GPU（0/1）±0.5%。
结构解读（与审计 §2.2 根因吻合）：fork 每 warp 1 token → 同一 x 块被 5 个 warp 各取各解 5 次（请求×5、解码 ALU×5，仅 L1 缓解）；ik 单 warp 一次解码向全部 token 的 y 投影，寄存器累加器组更大（tmp[nc][2]×2），`launch_bounds(32,1)` 放开 255 寄存器。带宽仍只到 12-23% copy——**离审计的 40-55% 目标还有距离，移植是"便宜的第一步"而非终点**。

### 2.3 判决表 B —— dense MMVQ（同日同 GPU 对跑）

| 形状 nc5 | fork | ik | 差距 |
|---|---|---|---|
| IQ3_S 2560×2048 | 14.1 µs | 15.1 µs | fork 快 1.07× |
| IQ4_XS 3840×2560 | 17.0 µs | 18.1 µs | fork 快 1.06× |
| Q6_K 6144×2560 | 34.3 µs | 35.6 µs | fork 快 1.04× |
| （nc1：IQ3_S 9.9 / IQ4_XS 11.1 / Q6_K 23.8 ik 侧；fork 同结构略快，同趋势） | | | |

同 launch 结构下 fork 的 channel/fastdiv 机器不亏反赚。**候选 2 判死：fork 已含该结构且更优**。

### 2.4 判决表 C —— ik MMQ（stream-K DP4A）vs fork MMVQ

| 形状 | nc | fork MMVQ | ik MMQ | ik 劣化 |
|---|---|---|---|---|
| IQ3_S 2560×2048 | 5 | 14.1 µs | 33.4 µs | 2.4×慢 |
| IQ3_S 2560×2048 | 8 | —（fork nc8 同核） | 33.4 µs | — |
| IQ4_XS 3840×2560 | 5 | 17.0 µs | 40.6 µs | 2.4×慢 |
| Q6_K 6144×2560 | 5 | 34.3 µs | 56.2 µs | 1.6×慢 |
| Q6_K 6144×2560 | 32/64 | — | 412/444 µs（31 GB/s） | 塌方 |

nc5 时 mmq_x 扫描选 8（stream-K nparts=1），tile 浪费 3/8 列；stream-K 的 80-block 网格在小 N（nty=20-48<80）下大量空块。**候选 3 判死**。

### 2.5 正确性（CPU 参考，双精度反量化+点积）

| 候选 | 格式 | max rel err | 2e-4 门槛 |
|---|---|---|---|
| ik fused MoE / ik dense / ik MMQ | IQ3_S / IQ4_XS / Q6_K | 3e-7 ~ 7e-7 | **PASS** |
| ik MMQ | IQ3_S nc8 / Q6_K nc64 | 3.8e-5 / 8.7e-5 | PASS |
| ik fused MoE | IQ2_S / IQ2_XS / IQ2_XXS / IQ3_XXS | **3e-4 ~ 8e-3** | FAIL（见下） |
| ik MMQ | IQ4_XS nc8 / Q6_K nc8 | 6.6e-4 / 9.7e-4 | FAIL（见下） |

两条 FAIL 的归因（非 kernel 缺陷证据，但需跟进）：
1. **iq2 家族 vs 浮点 CPU 参考的 1e-3 量级差**：GPU 核含整数截断合并恒等式（如 IQ2_S 的 `(sumi0*(sc&0xF)+sumi1*(sc>>4)+(sumi0+sumi1)/2)/4` 整除截断；截断界 ≤0.25/块×10 块×d·db ≈ 观测 1e-3）。**该恒等式在 fork 的同名函数中逐字节同样存在**（diff 证实唯一差异是 sign 掩码构造）→ 不是移植引入的风险面，但意味着"浮点 CPU 参考"对 iq2 家族天然只能对到 1e-3。
2. **MMQ nc8 的 6e-4**：我方 CPU 复刻 D4 量化器在 .5 舍入边界的 round/roundf 差（同一 kernel nc5/nc32 对到 1e-7）。
3. **跟进项（移植验收门槛）**：在真实 GGUF 权重上对拍 ik 结构 vs fork 现核输出（fork↔ik 数学等价已由静态 diff + 三格式 1e-7 支撑，iq2 家族需实测补一环）。

---

## 3. 收益池折算（标注：估算）

R17 trace（6 卡和，3130ms）MoE gu+gate 家族现状 → 按 §2.2 加速比折新：

| 家族 | R17 现状 | 加速比 | 省 |
|---|---|---|---|
| moe IQ3_S gu+gate | 237.7 ms | 1.86× | 110 ms |
| moe IQ2_XS | 155.1 ms | 1.70× | 64 ms |
| moe IQ2_XXS | 122.9 ms | 1.41× | 36 ms |
| moe IQ3_XXS | 92.5 ms | 1.60× | 35 ms |
| iq2s_volta（被 ik 通用结构取代） | 105.3 ms | 1.23× | 20 ms |
| **合计** | 713.5 ms | | **≈265 ms** |

- 265/3130 ≈ **理论上限 +8.5% decode**（隔离 bench 比值直折，未计原位稀释）。
- 按 BF16 战役落地折扣（实现拿到 kernel 级收益的 ~50-60%）→ **现实预期 +4-5%**。
- 附带机会（未测，标估算）：同一 1-warp-全-token 结构对 down_exps（IQ4_NL down 179.9ms + Q2_0 down 368.5ms 池）大概率同向收益——ik kernel 原生支持 IQ4_NL；Q2_0 是 fork 独有格式，vecdot 可直接复用 fork 的（结构移植不依赖解码函数来源）。若同 1.4×，再省 ~190ms（上限再 +6%，估算）。

## 4. 移植建议

**做：把 ik 的 MoE 融合核结构（k_fused_mul_mat_vec_q 形态）作为 fork 的新 MoE kernel**（候选 1）
- 形态：grid=(ceil(N/2), n_expert_slots)，block=(32,1)；每 warp 2 行 × 全部 token（j 循环），tmp_u/tmp_g[nc][2] 累加，SwiGLU 在写回处；`__launch_bounds__(32,1)`；x 的 ids/nb02 寻址与 fork 现有 fusion-args 机制对接。
- 工作量：kernel ~150 行（ik :152-280+:305-330 直改）+ fork 侧接线 ~100 行（mmvq_args→fork fusion args/tensor 布局映射，dst 索引 j*nrows_dst 需适配 fork 的 [token][slot] 排列）+ 验证。**估 1-2 天**。依赖面小：只依赖 fork 已有 vecdotq（解码函数不动）。
- 顺带（可选，+0.5 天）：iq2 家族 vecdot 的 `ksigns64` 表替换 `unpack_ksigns+__vcmpne4`（少一半 Volta 模拟指令；独立可测）。
- 验收：真实 GGUF 权重上 vs 现核输出 2e-4（iq2 家族另做 ik↔fork 对拍）；R17 原位 nsys 复测 moe 家族时长。
- 保留 fork 的 iq2s_volta？——ik 通用结构 N224 nc5 已 1.23× 于它、且覆盖全格式；建议移植落地后由 chooser 择优（iq2s_volta 的 q8_1-cache 副产物 `mmvq_moe_q8_out` 被 down 链复用，摘除前需查引用）。

**不做**：候选 2（fork 已有且更优，§2.3）；候选 3（全面劣化，§2.4）。

## 5. 工件

- 服务器 `/home/debian/ik-eval/`：`bench/`（bench_ik.cu、bench_ref.cpp、shim common.cuh、ik 逐字节副本×7、mmvq_ik.inc/quantize_ik.inc/cpu-ref.inc/ik-enum-*.inc、bench_ik 二进制、runF.log）；`bench-fork/`（bench_fork.cu+审计 inc 副本、runF.log）。
- 本地 `flashnext-llama-experiment/ik-eval/`：本报告、`bench/`（全部源+两份 runF.log）。
- 两 fork 源码零修改；GPU 已释放。

## 6. 数据边界与诚实声明

- bench 为隔离条件（独占 GPU、随机数据），审计已证系统性快于原位 0-25%；ik/fork 两侧同条件对跑，**比值可信、绝对值偏乐观**。
- ik fused MoE 的 nc>1 启动用的是 kernel 原生语义（ik 生产 host 在 nc>1 走 MMQ_ID）——这是"结构移植"的正确测法，但报告此事实以免误读为"ik 生产同配置"。
- iq2 家族 1e-3 正确性差与 MMQ nc8 6e-4 的归因是分析性结论（截断恒等式/量化器复刻舍入），未做逐位溯源实验；已在 §2.5 列跟进项。
- 收益折算含折扣假设；down_exps 收益未实测。
- git 历史取证因 shallow clone 不可用；源码注释证据已列行号。
