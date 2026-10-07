# 64K 长上下文 QSA/indexer 专项审计与修复报告

日期：2026-10-04 | 主机：ac922 (6×V100 sm_70, CUDA 12.4) | fork：/home/debian/ac922-llama-next/sources/llama-v100-v3 @ ac922-flashnext-v100 (7fe450e+)

## 一句话结论

**三处嫌疑在本 fork 全部不成立或不致命**：① sm_70 indexer "空壳" 与本模型无关（qwen4exp 根本不走 lightning-indexer kernel）；② 单路 64K decode 已有 blk 固定预算快速路径，逐步成本平坦、QSA 仅占 decode 墙钟 ~1.7%/GPU——外部社区"三宗罪"的等价修复**本 fork 已内置**；③ top-k radix 已在 CUDA 路径激活。**A 组 2×64K 下界异常的真凶是预填充调度串行（缺 LLAMA_PREFILL_FAIR_SHARE），一个环境变量实测恢复聚合 40.8 → 74.7 tok/s（+83%，达历史锚点 77.9 的 96%）**。无需改动源码（生产树原样保留，分支隔离程序备而未用）；两处真实的残余低效已量化并给出精确移交判断。

---

## 1. 诊断数据

### 1.1 静态审计（三宗罪逐一核对）

**嫌疑①：lightning-indexer.cu 在 sm_70 是空壳 → 判定：不适用（moot）**
- `ggml/src/ggml-cuda/lightning-indexer.cu:213-236` 的 NO_DEVICE_CODE 空壳只是 `lightning_indexer_kernel_wmma`（TURING_MMA 门控，common.cuh:290-292 只给 sm_75+）的占位；调度逻辑（:451-479）在 Volta 上走 `lightning_indexer_kernel_vec` SIMT F32 内核，**并非无实现**。
- 更关键：`GGML_OP_LIGHTNING_INDEXER` 只被 deepseek4/hy-v4/deepseek32/glm-dsa/dots3note 使用（grep 证实）。**Qwen3.8-Flash-Next（src/models/qwen4exp.cpp）的 indexer 评分 = mul_mat+relu+头求和（:1677-1697），从不调用 lightning kernel**。R17 8K trace 见 0 条 indexer kernel 是正确行为，不是 bug。

**嫌疑②：单路 64K pool/n_kv 扫描 → 判定：decode 已修（fork 内置），prefill 有全 ctx 扫描但占比小**
- 单序列 decode 走 `qwen4exp_qsa_blk_fast`（qwen4exp.cpp:1023-1045）：增量 pooled 块缓存（只重池 touch 的块）+ 固定预算 top_k/ratio=512 块 + tail。门条件 `(c0+1)/ratio >= n_sel+1`（≈2052 cells 后切换）。
- 每步重建 O(n_kv) 的普通路径（build_qsa_top_k，:1566-1733）只在 prefill 和 <2052 cells 时使用。
- 多序列 decode 走 mblk 快速路径（LLAMA_QSA_POOLED_MULTI=1，B 开关）：A 组日志 43 条 qsa-mblk trace、**0 条 demoted**——B 门在 2×64K 全程生效。
- 残余：prefill 普通路径的 QSA 图按**整个分配的 ctx**（kvlen=132096 单路 / 266240 双路）评分而非已用 cells（PATH_TRACE 证实：第一条 prefill ubatch 即 kvlen=132096）。开销见 §1.3。

**嫌疑③：radix/CUB top-k 未启用 → 判定：已启用**
- common.cuh:114-117：CUDA≥11.7 即定义 `GGML_CUDA_USE_CUB`（12.4 ✓）。CCCL<3.2 → `CUB_TOP_K_AVAILABLE` 未定义 → top-k.cu:443-447 走 **`top_k_radix_cuda` 自定义多趟 radix 路径（ncols>1024 时）**。nsys 证实 decode/prefill 均有 `top_k_radix_histogram_select` 实例（decode 14604 次 avg 9.5us）。外部 qwen4exp 的"radix 未启用"缺陷不适用于本 fork。

### 1.2 8K vs 64K 单路实测（生产配方原样，r1，PATH_TRACE+STEP_TIMING）

| 指标 | 8K/512 | 64K/512 | 变化 |
|---|---|---|---|
| PP tok/s | 625.2 | 720.3 | +15%（更长 prompt 摊薄固定开销） |
| **TG tok/s** | **93.38** | **83.67** | **-10.4%** |
| decode draft 步 (compute) | 1.405ms 平坦，rebuilt=0 | 1.942ms 平坦，rebuilt=0 | +38%，**窗口内零增长** |
| decode target 验证步 | ~22.1-27.3ms | ~23.8-26.0ms | +5-8% |
| prefill draft 32-token ubatch | ~4.7ms (n_kv≈8K) | 4.7→8.7ms 随 n_kv 线性 | 线性（稠密注意力，固有） |

**判定：无超线性**。decode 全程逐步成本平坦（固定上下文内），8K→64K 的 -10% 来自稠密层/draft 注意力对 KV 的固有线性项 + draft 步 +0.54ms，**不是** QSA 预算泄漏（top_k=2048 预算被遵守，decode 无任何 qsa-path 普通路径 trace）。

### 1.3 64K 单路 nsys kernel 级拆分（2023.2.1，profiles/qsa-lc-64k/）

decode 窗口 6.64s（~205 个 MTP 验证步），每 GPU 占比（6 卡求和/每卡均匀）：

| 类别 | 每 GPU 总量 | 占墙钟 | 每验证步/GPU |
|---|---:|---:|---:|
| **NCCL AllReduce (f32 LL)** | **661-961ms** | **10-14.5%** | ~3-4.7ms |
| MoE/GEMV 族（gemv_sm70_bf16 等） | ~3.6s 合计 | ~30% | — |
| QSA: topk 选择 (radix) | 25-27ms | 0.4% | 0.13ms |
| QSA: indexer 评分 GEMM (sgemm_128x32) | 14-15ms | 0.2% | 0.07ms |
| QSA: pool get_rows (f32) | 8-12ms | 0.15% | 0.05ms |
| QSA: pooled norm+rope | 8-12ms | 0.15% | 0.05ms |
| QSA: gather 行 (q8_0→f16) | 17-19ms | 0.3% | 0.09ms |
| QSA: gather 后 FA (flash_attn_tile) | 18-19ms | 0.3% | 0.09ms |
| cont 拷贝（QSA 路径部分） | 9-11ms | 0.15% | 0.05ms |
| **QSA 合计** | **~110ms** | **~1.7%** | **~0.54ms** |
| 稠密/draft FA (flash_attn_ext_f16, 仅 GPU4,5) | 209/207ms | 3.1%(那两卡) | 随 n_kv 线性 |

**64K decode 的 QSA 全链路 ≈ 0.54ms/步/卡（~3.2ms 六卡合计）**——任何 QSA 侧修复在此处上限 ~2%，不存在 2-4× 空间。decode 中**没有形状随 n_kv 无界增长的 kernel**（选择面=已池化块数 16640@64K，线性但常数极小：250K 时约 +0.8ms/步）。

prefill 切片（12s）：NCCL 49%、MoE grouped ~15%、`qsa_attn_kernel`(块稀疏注意力) 936ms/3%、QSA 重建机器（argsort+topk+getrows+rope）~320ms/1%。**prefill 的 QSA 全 ctx 扫描在单路 64K 占 prefill 步 ~10%、可砍半 → 仅 ~2-5% prefill 收益**；在 2×64K（ctx 266240）翻倍。

### 1.4 A 组 2×64K 下界异常（2.7/38）根因：**预填充调度串行，非 QSA**

证据链：
1. **历史锚点 77.9（manifests/zcode-bh-2x64k）的测量形态**：两流 PP 完全相等（175.16/175.16，prompt_ms 均 374140ms）→ 预填充**公平交错**，两流 decode 窗口完全重叠 → 各 38.7/39.3。
2. **A1-A3（multi-concurrency-bench.sh scenario_env 证实）没设 `LLAMA_PREFILL_FAIR_SHARE=1`**（只有 B1/B2/B3 有）→ 两 prompt 串行预填充：流 0 先完成（PP 478-864），其 decode 窗口 189s 几乎全程与流 1 的 180-210s 预填充重叠 → 流 0 TG=2.3-3.3（等待稀释），流 1 完成后联合解码 36-38 正常。
3. **联合解码本身健康**：A1-A3 中流 1 实测 36.2-38.1 ≈ 锚点 38.7/39.3。B 门（POOLED_MULTI+MIN_CTX=32768）全程生效（0 demotion）。
4. **复现实验（本审计 A2F）**：A2 配方 + 仅加 `LLAMA_PREFILL_FAIR_SHARE=1` → 两流 PP 232.4/232.4（完美交错），**TG 33.84/40.89，聚合 74.73**（vs A1-A3 的 36.6-40.8）。日志：/home/debian/ac922-llama-next/logs/conc-A2F-20261004-194401.log，结果：concurrency/qsa-lc/c2-p65536-fair.json。
5. 剩余 74.7 vs 77.9（-4%）：锚点为 6 卡 TP 配方 vs A2F 岛 3,3 + 现代 flag 集，属配方差异非回归（A1 与 A2 原差也在噪声内）。

### 1.5 多流 decode 的真实残余低效（移交项，本轮不动代码）

从 A2/A2F 的 STEP_TIMING：
- **多流 draft 图重建抖动**：双流联合 decode 中 draft 步 8-14.6ms（A2 无 fair-share 时段，rebuilt 4-5/32 步，per-rebuilt ~60ms）vs 单流 1.9ms rebuilt=0。单流 MTP_PAD=6 使形状恒定（tokens=6..6）；双流时 draft ubatch 变 32-token 且 token 数 2..10 漂移 → 图缓存 miss。A2F（fair-share）下已缓解（decode 段 compute 1.7-2.5ms，但仍有 rebuilt 12-14/32）。
- **双流验证步 50-62ms vs 单流 24-26ms**：统一 cache 下稠密/draft 注意力按聚合 n_kv（131K）+ 2× token 计费——部分固有。
- 潜在收益上限（若 draft 抖动归零 + 验证步仅按 2× 计）：联合 decode 每流 38 → ~48-52（聚合 ~100）。

---

## 2. 修复清单

**源码改动：无（诊断否决了全部三处候选病灶；生产树原样冻结）**。分支隔离程序（git stash push -m prod-kernels → checkout -b qsa-longctx → …）按预案备好但未启用；`git status` 132 个 M 文件与会话开始时一致，无 stash 残留。

**配置级修复（已实测验证，建议并入生产 A 组场景定义）**：
- `scripts/multi-concurrency-bench.sh` 的 A1/A2/A3 scenario_env 增加 `LLAMA_PREFILL_FAIR_SHARE=1`。
- 实测效果：2×64K 聚合 36.6-40.8 → **74.73**（+83%），双流平衡 33.8/40.9（原 2.7-4.4/33-38）。
- 代价：PP 名义速率下降（串行时流 0 独享 478-864 vs 交错 232），但**墙钟总时长相当或更好且 decode 吞吐翻倍**——多用户口径下严格更优。

---

## 3. 前后对照总表（本机同口径实测）

| 场景 | 修复前 | 修复后（fair-share） | 锚点 |
|---|---|---|---|
| 1×8K TG | 93.38（生产配方复测） | 不适用（未动源码） | 94.3 配方 |
| 1×64K TG | 83.67 | 同左（无改动） | — |
| 2×64K 聚合 TG | 36.56-40.78（A1-A3，2.7-4.4/33-38 不均衡） | **74.73（33.84/40.89）** | 77.9（bh-2x64k） |
| 8K parity | — | 未动任何代码/权重路径，parity 定义上不变；A2F 两流 512-ID 与 cache_prompt=false 参考可再对拍 | — |

（A2F 未跑 ID 对拍：fair-share 只改预填充切分顺序，不改任何数值路径；流 1 的 40.89 vs A2 的 38.06 同配方同 seed 差异来自测量窗口对齐，非输出差异。）

---

## 4. 裁决

1. **单路 64K**：继续（现状健康）。TG 83.67，QSA 仅 1.7% 墙钟，逐步成本平坦，无 n_kv 无界 kernel。剩余 8K→64K 的 -10% 是稠密层+draft 固有线性；若要再压，方向是 dense/draft 注意力（如 draft KV 量化/滑窗）而非 QSA。
2. **A 组 2×64K**：**已修（配置级，+83%）**。fair-share 应并入生产并发配方。历史 77.9 未"复现"是因为它本来就是交错调度的产物——A1-A3 的串行调度从未给过两流重叠的 decode 窗口。
3. **移交（不立项判死，按优先级）**：
   - **P1 多流 draft 图形状稳定化**（MTP draft 双流时 tokens 2..10 漂移 + 32-token ubatch → 图重建 40-60ms/次）：改 `LLAMA_MTP_PAD` 语义为跨流恒定 shape 或给 draft 图加 shape 桶。落点：speculative.cpp / qwen4exp draft 构图；预期联合 decode 每流 +20-30%。验收：单流 8K parity + rebuilt/32 → ≤2。
   - **P2 prefill QSA 全 ctx 扫描钳制**（kvlen=allocated→used 按 8192 桶取整）：落点 llama-memory-hybrid-idx.cpp 的 get_n_kv 消费侧或 build_qsa_top_k 的 n_kv（qwen4exp.cpp:1575）；预期 8K/64K/2×64K prefill +2-5%；需验证图复用桶化不引入重建抖动。
   - **P3（老问题重申）NCCL 10-14.5% decode 墙钟**：S1/R17 已证岛内 P2P/图内通信优化空间有限，维持不立项。
   - 判死：sm_70 lightning-indexer 替代实现（本模型不用）；top-k radix 启用（已启用）；单路 decode pool 重池（blk 路径已修）。

## 5. git 序列（在案）

- 会话全程未 checkout、未 stash、未 commit：生产分支 ac922-flashnext-v100 及其 132 个未提交 M 文件保持会话开始原状（结束复核一致，stash list 空）。
- 环境改动：仅 systemd 测试单元 ac922-flashnext-test（已停止并清除）；logs/conc-A2F-20261004-194401.log、profiles/qsa-lc-64k/{prefill-slice,decode}.nsys-rep、concurrency/qsa-lc/*.json 为新增产物。
- GPU 收尾：六卡 0 MiB / 0%。

## 6. 证据文件索引（远程）

- 生产配方 8K/64K 单路：logs/qsa-lc-r1.log（STEP_TIMING+PATH_TRACE）；concurrency/qsa-lc/c1-p8192.json、c1-p65536.json
- nsys：profiles/qsa-lc-64k/{prefill-slice,decode}.nsys-rep + .sqlite + capture_qsa_64k.py
- A 组复验：logs/conc-A2F-20261004-194401.log；concurrency/qsa-lc/c2-p65536-fair.json
- 历史锚点：manifests/zcode-bh-2x64k/c2-p65536.json；A 组原始：concurrency/RESULTS.tsv、scripts/multi-concurrency-bench.sh
