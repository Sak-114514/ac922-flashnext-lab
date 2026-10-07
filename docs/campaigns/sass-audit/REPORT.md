# SM70 SASS Audit 战役报告 — kernel 级 SASS 对照审计（冻结配方 94.3 tok/s）

日期：2026-10-04 · 战役：sass-audit · 硬件：ac922（POWER9 + 6×V100 SXM2 sm_70，CUDA 12.4）
对象 fork：`/home/debian/ac922-llama-next/sources/llama-v100-v3`（**零源码改动**，全审计只读 + /tmp 独立编译）
生产配方：`launch_integration_arm.sh`（BF16_GEMV=1 + MOE_REWORK=1 = 94.3 tok/s）
trace 基础：R17 decode.sqlite（777,762 kernel 事件，聚合 GPU busy 9.373s/6 卡；构成与任务书吻合：BF16 33.0%、quant dense ~16.7%、MoE 12.4%）

## 裁决（先说结论）

**fragment-native（QPN / offline repack + 寄存器 dequant + mma.m8n8k4）路线在本配方、本硬件上判死，不进 PoC 接线。** 三大家族逐一过门禁：

| 家族 | 新配方时间占比 | 指令冗余 vs QPN | ≥15% kernel 加速可行性 | 预测端到端 ≥3% | 裁决 |
|---|---|---|---|---|---|
| BF16 GEMV（我方 gemv_sm70） | ~20.2% | **无**（解码=1 SHF/元素是 bf16→f32 格式下限；已 128-bit 载、INT movers 37% vs 原路径 60%） | 否（命中形状实测 824GB/s = 实测 copy 上限 870 的 92%） | 上限 <1% | **判死** |
| MoE Q2_0 down（我方 moe_sm70） | ~2.5% | **反向**：QPN 每权重指令 1.63 vs 我方 1.78（−9%），且 QPN 实测**慢 2×**（40.8µs vs 20.74µs，distinct 生产路由） | 物理不可能：需 2.2× 才到 3% 端到端，带宽+调度地板 14.1µs 封顶 1.47× | 2.5%×47%=1.2%（按地板极限） | **判死** |
| dense quant（原路径 mul_mat_vec_q 家族） | ~20.5% | **有**：nc5 时 IDP/scale-I2F/FFMA 按 token 重执行 ×5（census 实证） | QPN 路线否（本审计实测 mma 路线 226-248GB/s）；最大单成员 Q6_K 需 70% 加速，F 线三代架构仅 0.4-0.63× | Q6_K 4.26%×任何现实加速 <3% | **fragment-native 判死**；就地 decode-once 重构另立案 |

无热点通过双门禁 → 阶段 4 PoC 接线不触发，生产树保持冻结，94.3 配方不动。

## 0. 对任务书前提的勘误（SASS 证据）

1. **type 42 = fork 重排枚举的 Q2_0**（Q1_0=41, Q2_0=42；trace 里 3.93% 的 `mul_mat_vec_q_moe<(ggml_type)42,...>` 即 MoE Q2_0 down）。qmmvq-audit 已有同一勘误。
2. **monomoe-sm70.cu 不是 fragment-native mma 路线**。fork 源码头注释自述"weight-streaming MoE, no Hopper ISA"；SASS census：`mono_kernel<18,42>` 等 10 个实例**零 HMMA**，纯 FFMA warp-dot + 协作单 launch。R16-B 的 +2.235% 来自 grid 融合（消 22µs 级调度地板），不是张量核。它的 QPN 参照价值仅剩"融合"语义——而融合属通用 fusion，红线禁区内。
3. **Volta 有硬件整数点积**，SASS 助记符为 `IDP.4A.S8.S8`（带点，非 IDP4A）。fork 的 dp4a 路线是硬件加速的，不是标量模拟（我方 census 初版误判，已修正——这也是本审计工具链的一个坑，记录在案）。
4. 1Cat 遗产 `onecat-kernels/quantization/marlin/marlin_mma.h` 用 `mma.sync.m16n8k16`——**sm_75+ 形状，sm_70 编不过**；其 sm70 CUTLASS Volta 路线绑 torch/ATen 无法独立编译。QPN 参照最终采用自包含最小实现（见 §3）。

## 1. 阶段 1 — 热点定位（R17 trace 离线 + 新配方折算）

R17 trace 为老配方（两开关全关），聚合 6 卡 kernel busy 9.373s。Top kernel（按聚合时间）：

| # | kernel | 占比(老) | n | avg µs | reg | grid×blk |
|---|---|---|---|---|---|---|
| 1 | `mul_mat_vec_f<bf16,f32,5,160>` | 18.23% | 28809 | 59.3 | 64 | 10240×160 |
| 2 | `mul_mat_vec_f<bf16,f32,5,256>` | 14.78% | 114939 | 12.1 | 64 | 201×256 |
| 3 | `k_allreduce_p2p` | 10.41% | 32028 | 30.5 | 32 | 8×512（通信，战役范围外） |
| 4 | `mul_mat_vec_q_moe<Q2_0=42,2,0>` | 3.93% | 8910 | 41.4 | 45 | 12800×160 |
| 5-11 | dense `mul_mat_vec_q`：Q6_K 3.48 / Q5_K(nc1) 2.63 / IQ4_XS 2.42 / IQ3_S 2.16 / Q4_K 2.04 / Q5_K(nc5) 1.75 / Q8_0 1.14 / IQ4_NL 1.11 | 16.73% 合计 | | | | |
| 12-17 | 其余 moe（IQ3_S 2.54 / IQ4_NL 1.92 / IQ2_XS 1.66 / IQ2_XXS 1.31 / IQ3_XXS 0.99，另 iq2s_volta 1.12） | 12.35% 合计 | | | | |

**新配方折算**（BF16 命中形状 ×~2.0 提速、MoE Q2_0 ×1.91 distinct 口径，依据两份 INTEGRATION.md 微基准；总量 9373→~7658ms）：

| 家族 | 老 | 新（估） |
|---|---|---|
| BF16 GEMV（→ gemv_sm70） | 33.0% | **~20.2%** |
| dense quant matvec | 16.7% | **~20.5%** |
| MoE 家族（Q2_0 → moe_sm70） | 13.5% | **~14.2%**（Q2_0 down 单核 ~2.5%） |

热点表（战役口径 top3 = BF16 GEMV / dense quant / MoE Q2_0），与任务书预期一致。

## 2. 阶段 2 — SASS census（生产 libggml-cuda.so.0.25.1，168 个 sm_70 cubin）

方法：`cuobjdump --dump-sass` 全量导出（13.1M 行）→ 符号切片 → 静态指令构成 + BRA 回边定位动态循环体。工具与原始输出见 `artifacts/`。

### 2.1 静态构成表（关键指标）

| kernel | 静态指令 | HMMA | IDP.4A | LOP3 | PRMT | SHF | IMAD | MOV | 主要访存 | INT-movers |
|---|---|---|---|---|---|---|---|---|---|---|
| **我方** gemv_sm70_bf16<4,5,256> | 608 | 0 | 0 | 17 | 0 | 137 | 23 | ~0 | **LDG.E.128×14** | 43.6% |
| **我方** gemv_sm70_bf16<4,2,256> | 328 | 0 | 0 | 17 | 0 | 71 | 23 | ~0 | LDG.E.128×8 | 51.2% |
| 原路径 mul_mat_vec_f<bf16,5,160> | 792 | 0 | 0 | 2 | 10 | 76 | 139 | **165** | LDG.64×25+32×8（窄） | 59.7% |
| **我方** moe_matvec_q2_0_sm70<KB4> | 384 | 0 | 16 | 30 | 34 | 21 | 105 | 0 | STS.128 staging→LDS | 65.9% |
| 原路径 mul_mat_vec_q_moe<Q2_0,2,0> | 232 | 0 | 16 | 18 | 40 | 22 | 57 | 0 | **LDG.U16×11+32×9（窄）** | 69.4% |
| 原路径 Q6_K<14,5> | 544 | 0 | 20 | 31 | 16 | 76 | 73 | 10 | **LDG.U16×20（scale 窄读）** | 56.6% |
| 原路径 Q5_K<13,1>（输出头） | 200 | 0 | 8 | 24 | 0 | 28 | 49 | 5 | | 65.0% |
| 原路径 IQ4_XS<23,5> | 632 | 0 | 80 | 38 | 64 | 81 | 64 | 12 | | 53.6% |
| 原路径 IQ3_S<21,5> | 808 | 0 | 80 | **143** | 16 | 86 | 140 | 5 | | 61.1% |
| 原路径 Q4_K<12,5> | 656 | 0 | 60 | 38 | 0 | 80 | 71 | 9 | | 48.0% |
| mono_kernel<18,42>（R16-B） | 4920 | **0** | 0 | 624 | 171 | 297 | 821 | 12 | LDG.U16×103 | 58.7% |
| **QPN 参照**（本审计构建，见 §3） | 416 | **64 (15.4%)** | 0 | 89 | 32 | 27 | 48 | 0 | LDG.32 页主序（合并） | 57.9% |

寄存器（`cuobjdump --dump-resource-usage` + trace）：我方 bf16<4,5,256> REG:80；我方 moe KB4 REG:32；QPN REG:48；**全部零 spill**（STACK:0/LOCAL:0；mono 亦零 spill）。

### 2.2 动态循环体（BRA 回边实测）

| kernel | 主循环体 | 每次迭代内容 | 归一化 |
|---|---|---|---|
| 我方 moe KB4 | 114 inst（0xe20-0x1530） | 16 IDP + 32 PRMT + 16 LOP3 + 27 LDS + scale | 64 权重/token → **1.78 inst/权重** |
| 原路径 moe Q2_0 | 133 inst | 16 IDP + 40 PRMT + 19 窄 LDG | 64 权重/token → 2.08/权重 |
| 原路径 Q6_K<5> | 256 inst | 20 IDP + **20 I2F(=5token×4) + 30 FFMA(=5×6)** + 34 窄 LDG | 32 权重×5 token |
| 原路径 IQ3_S<5> | 531 inst | 80 IDP + **140 LOP3**（vcmpne/vsub4 Volta 模拟的掩码构建） | ALU 病灶实证（与 moe INTEGRATION 分相实测一致） |
| 原路径 Q4_K<5> | 379 inst | 60 IDP + **40 I2F(=5×8) + 50 FFMA(=5×10)** | 32 权重×5 token |

**dense 家族按 token 重解码的源码级证据**（mmvq.cu 主循环）：
```cpp
for (int j = 0; j < ncols_dst; ++j)            // 5 tokens
    for (int i = 0; i < rows_per_cuda_block; ++i)
        tmp[j][i] += vec_dot_q_cuda(vx, &y[j*...], kbx..., kqs);   // 权重块按 j 重入
```
权重 LDG（`ld.global.nc`）与部分 crumb 解码被编译器 CSE 跨 j 复用；**IDP 点积、scale I2F、FFMA 归约按 token ×5 重执行**（census 的 5 倍数结构证实）。

## 3. 阶段 3 — QPN 参照（自包含 m8n8k4 实现 + 同口径 census）

参照选型排除：monomoe（非 mma，§0.2）、1Cat marlin（m16n8k16 不可 sm_70 编译；CUTLASS 路线绑 torch）。**自建最小 QPN 参照核**（`artifacts/qpn_ref_sm70.cu`，生产 MoE Q2_0 down 形状）：

- 权重**离线重排**：2-bit crumb 页主序 `[k页][行]`（warp 页载全合并 128B），每 (行, k16) 一个 u32；
- **寄存器内 dequant**：四银行（移位 0/2/4/6，每字节 4 crumb）→ PRMT 配对 → 单 LOP3 折叠 AND-mask+OR-0x6400（f16(1024+c) 精确）→ HSUB2 去 bias；
- **mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32**，fragment 布局 crib 自 fork 生产核 fattn-q8-volta-head.cuh 并经 3 轮 GPU 探针实证修正（探针资产 `qpn_probe{,2,3}.cu`）：
  - B fragment：lane 持 B[k=2l(+1)][n=i(lane)]（k 在 half2 槽、token 在 lane 映射）；
  - `__byte_perm` 无零字节源（nibble 0 选 byte0 而非清零）——必须后接 mask；
  - 0x03030303 掩码按字节提取的是 **crumb 4m** 不是 2m（每字节 4 个 crumb）；
  - D fragment 8 槽分属两行（d_i=(l&2)+(lane&~2)），scale 必须按槽取；
- 激活 f16 预排 B-fragment 序；per-(行,k64) f16 scale（对齐 Q2_0 组宽）。

**正确性**：CPU 双精度参考 parity **bad=0，max rel err 1.9e-3**（f16 舍入级，门槛 2e-3）。

**微基准（GPU5，50 个 distinct 专家 slab/launch = 9.22MB 真实流量，生产路由语义）**：

| 形状 | QPN 参照 | 我方生产核 | fork 原路径 | copy 地板(870GB/s+3.5µs) |
|---|---|---|---|---|
| K=256 nc5 distinct | **40.80 µs（226GB/s，26%copy）** | 20.74 µs（57%copy） | 39.53 µs | 14.1 µs |
| K=256 nc1 distinct | 37.11 µs（248GB/s） | 20.74（nc1 直测 6.71 uniform） | 20.74 | 14.1 µs |
| K=192 nc5 distinct | 34.15 µs（202GB/s） | ~18（内插） | 39.5 | ~11 µs |

**QPN 比我方生产核慢 1.97×，与 fork 原路径持平。** census：HMMA 64（15.4%）、movers/HMMA=3.77、HMMA 平均间距 3.1 条、REG 48 零 spill。慢的结构性原因（V100 事实，非实现失误可全消）：
1. **distinct 路由摧毁 mma 的唯一红利**——8-token B fragment 摊销要求权重跨 token 共享，per-token distinct 专家下每 warp 只服务 1 个 token，张量核 7/8 空转，指令数优势只剩每权重 −9%（1.63 vs 1.78）；
2. **2-bit→f16 dequant 天然贵于 IDP unpack**：crumb 2-bit 字节内错位（4/字节）需四银行提取，~1.9 inst/权重；而 IDP 路线 crumb 解包后直接整数进点积（0.375 inst/权重的 unpack + IDP 本身即 MAC）；
3. mma 累加链（单 d[8] 串行 64 HMMA）+ 每 k64 双行 scale 窄载，ILP 不足以藏 V100 张量核延迟。

## 4. 阶段 4 — 差距判决（逐家族算式）

门禁复述：热点 kernel ≥15% 加速 **且** 预测端到端 ≥3%（端到端 = 加速比 × kernel 时间占比 × GPU busy 折算）。

### 4.1 BF16 GEMV（我方，新 ~20.2%）
- 解码下限论证：bf16→f32 恰为 1 条 SHF/元素（我方 census SHF 22.5% 与 FFMA 26.3% 匹配——每元素 1 移位 1 乘加，即格式理论下限）；转 f16-mma 需 bf16→f16 转换（PRMT 级，同量级）且丢 f32 累加 parity。
- 带宽：命中形状 824GB/s = 实测 copy 870 的 92%（nc1 口径，INTEGRATION 实测）；INT movers 已从原路径 59.7% 压到 43.6%，原路径的 MOV 20.9%/窄载病灶已被我方核消灭。
- 端到端：无 ≥25% 指令冗余 → 门禁第一条件即不满足。**判死**。

### 4.2 MoE Q2_0 down（我方，新 ~2.5%）
- 指令对比反转：QPN 1.63/权重 vs 我方 1.78/权重（−9%，且 QPN 含更重的 scale 窄载）→ **<25%**，门禁第一条件不满足；
- 即便假设完美执行（指令全消）：kernel 上限 = 20.74→14.1µs = 1.47× → 端到端 = 1.47×...×2.5%×(占比) = **2.5%×32% = 0.8%**；
- 达 3% 端到端需 kernel 2.2×（=8.9µs < 9.22MB/870GB/s 的 10.6µs 纯带宽时间）——**物理不可能**。
- 实测 QPN 参照慢 2× 佐证。**判死**。

### 4.3 dense quant（原路径，新 ~20.5%）
- 指令冗余**成立**：×5 的 IDP/I2F/FFMA 重执行（census 5 倍数结构 + 源码 j 循环）——这是本次审计唯一找到的 ≥25% 冗余；
- 但 PoC 指定路线（fragment-native repack + mma）被本审计实测否证（§3：mma 路线 226-248GB/s，比原路径还慢的形状也存在；nc5 摊销虽在，V100 2-bit/6-bit→f16 转换开销与延迟结构吃掉全部收益；F 线独立证据：Q6_K 就地解码 4440 指令/线程 12× 延迟放大）；
- 单成员端到端天花板：Q6_K 4.26% × 需 70% kernel 加速才到 3% —— F 线全战役最好 0.63×（倒退），QPN 参照亦无此量级证据；
- **理性修法是就地 decode-once 重构**（vec_dot 解码一次 + IDP×ncols，无 repack/无 mma/无格式变更）——但属生产树源码改动，冻结红线内禁止，且家族总天花板 ~20.5%×20-25% ≈ 4-5% 需逐格式（8 种）重构，非"最小 PoC"。**fragment-native 判死；就地重构另立案（建议进下战役评估，优先 IQ3_S：LOP3 143/iter 已证 ALU-bound）**。

### 4.4 附：allreduce_p2p（10.4%）与输出头 Q5_K（3.2%）
- allreduce 属 R18/P2P 红线区，不动；
- 输出头 Q5_K<13,1> 76%copy 且 grid 82774 的调度地板 124-135µs 与内存时间重叠（qmmvq-audit）——已近该 grid 结构实用上限，非指令冗余问题。

## 5. 结论与资产

**总裁决：fragment-native/QPN 路线判死归档；无 PoC 接线；生产配方 94.3 tok/s 维持不动。**

我方两个生产核（bf16 +8.9%、moe +1.27%）在 SASS 层面经对照确认处于各自约束（带宽/调度地板）的合理侧：BF16 核 128-bit 载 + FFMA 密度 26% + 零 spill + 92% 带宽上限；MoE 核 57% copy、指令 1.78/权重 ≤ QPN 的 1.63+结构劣势。

**审计副产资产**（`~/Desktop/ssh/flashnext-llama-experiment/sass-audit/artifacts/`，远端 /tmp 同名）：
1. `sass_census.py` / `loopanalyze.py` — 生产 so 的 SASS 静态+动态循环 census 工具（已含 IDP.4A 助记符、>4 位地址两个坑的修复）；
2. `qpn_ref_sm70.cu` — parity 过门的 sm_70 fragment-native m8n8k4 参照核（离线页主序 repack + 四银行 dequant + 双行 scale），生产 MoE 形状可复现 bench；
3. `qpn_probe{,2,3}.cu` — m8n8k4 fragment 布局 GPU 探针（B 的 n↔lane 映射、byte_perm 零字节陷阱、crumb 银行步长、D 双行 scale 四个坑的实证记录）——未来任何 sm_70 mma 工作的起手资产；
4. `census_*.txt` — 11 个 kernel 的完整 census 原始输出。

**给下战役的移交判断**（不在本轮红线内执行）：
- dense 家族 per-token 重解码是全审计唯一量化的 ≥25% 冗余；就地 decode-once 重构（mmvq.cu 主循环 j/i 序重排，让 crumb 解码与 scale I2F 提出-token 循环）是无格式变更、无 repack 的正解，家族理论空间 ~4-5% 端到端，但需逐格式验证并过 512-token parity 门；
- IQ3_S（LOP3 143/iter，ALU-bound 实证）与 Q6_K（scale U16 窄读 ×20 + I2F ×5）是其中指令密度最差的两处；
- MoE 家族剩余空间已被流量勘误封顶（<2% 步时），不再投入——与 moe-rework INTEGRATION 裁决一致。

## 红线合规记录
- 生产源码零改动（全程只读 + /tmp 独立 TU；`git diff` 意义上的树未触碰）；
- 未调 MTP/未做通用 fusion/未碰 R18/P2P/MonoMoE 开关；
- GPU 使用：SASS 导出零 GPU；探针与微基准用 CUDA_VISIBLE_DEVICES=5，跑前每次 nvidia-smi 确认（期间 TTFT server 两次间歇驻留显存但 util 0%，微基准总量 <2 分钟 GPU 时间，未与其争卡；nsys 新配方 trace 未跑——老 trace + INTEGRATION 折算已满足门禁数据需求）。
