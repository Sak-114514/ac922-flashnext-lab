# env-switches — AC922 Flash-Next 开关总表

> 代码：[Sak-114514/llama.cpp](https://github.com/Sak-114514/llama.cpp) 分支 `ac922-flashnext-v100`。
> 本文是文档档案；以分支内实现为准。数值口径见 [performance-contracts.md](performance-contracts.md)——
> 本文所有百分比均为**合成口径同臂 A/B**。

## 一、SM70 kernel 战役开关（ZCode 六战役产物）

| 开关 | 默认 | 实测 | 状态 / 出处 |
|---|---|---|---|
| `GGML_SM70_BF16_GEMV` | **1** | TG **+8.9%**（111.24/110.50 vs 100.17/103.35），parity 4×512 逐位 | 生产。uint4 流读 + bf16 位移转换单 kernel，白名单 `k>=10240 || (nc>=5 && N>=8192)`。`docs/campaigns/bf16-gemv/` |
| `GGML_SM70_FUSED_PROJ` | **1** | 微基准 1.41-1.50×，端到端中性（-0.05%） | 生产（无害）。wave 融合实际只收 n_task>=2 组（6 wave）；DESIGN 的 BF16 清单被 GGUF 运行时证伪。`docs/campaigns/layer-fusion/` |
| `GGML_SM70_MOE_REWORK` | **0** | 开启即 **+1.27%**（93.06 vs 91.90 臂均值） | 开关就绪。grid 12800→1000 blocks，调度地板 22.2→~3.5µs；白名单 K∈{192,256}。`docs/campaigns/moe-rework/` |
| `GGML_SM70_BF16_PREFILL_F16` | **1** | PP 8K **+6.2%** / 64K +5.8%，TG 持平，parity 逐位（2047/2048 token） | 生产。修 fork BF16 权重 batch>8 走 F32 SIMT 的病（源自 Strata #655 思路 + #540 饱和钳制）。`docs/campaigns/strata-port/` |
| `GGML_IK_FUSED_MOE` | 0 | 全模型 **-3.6%**（88.76 vs 91.98） | **判死**留档。ik 移植收益是编译器 CSE 幻觉（channel-uniform 语义消失后蒸发）。`docs/campaigns/ik-eval/` |
| `GGML_CPU_PLE_ROW_CACHE` | 0（opt-in） | 冷 prefill 241.95→360.02（+49%），TTFT 68→41.8s | 归档。热 prefill 稳态税 -26~29%，不进生产；生产用 dd 预热替代（零稳态税）。`docs/campaigns/ttft/` |
| `LLAMA_QSA_Q4V` | 0 | 8K PP +3.4% 但 parity token 0 分岔、TG -19% | **判死**不合入（kernel prefill 数值 bug，方向未死）。`docs/campaigns/qsa-q4v/` |
| `LLAMA_MWE_ATTN` | 0 | 64K PP 仅 +0.39%（门 >=+5%），TG -1.2% | 判死。segment 置换 parity 全真但归因（头失衡）被 QSA gather 稀疏路径证伪。`docs/campaigns/mwe/` |

## 二、结构性 preset（Codex 时期转正，启动器配置）

| 开关 | 值 | 实测 | 说明 |
|---|---|---|---|
| `GGML_CUDA_MOE_GROUPED` | 1（H） | PP 532.6→586.2（+10.1%）；2×1K +118% | MoE 死块消除 |
| `LLAMA_QSA_POOLED_MULTI` | 1（B，限长上下文） | 2×64K TG 13.2/17.5→25.5/27.9；与 H 合体 30.7→77.9（+154%） | QSA 池化多路；8K 下有劣势 |
| `LLAMA_QSA_MULTI_MIN_CTX` | — | B 的最小上下文门 | 防止 8K 退化 |
| `LLAMA_ISLAND_SPLIT` | `3,3`（C） | 单路 PP 532→749（+40.8%） | 六卡 3+3 两岛，AllReduce 限岛内；GPU 顺序必须 UUID 固定 |
| `GGML_META_ASYNC_COPY` + `GGML_META_FIFO_PIPELINE` + `LLAMA_PREFILL_KV_PAD=8192` | 1（R18 三件套） | PP 749→966.51（+28.9%），TG +2.27% | FIFO prefill 组合；对长上下文 prefill 增效 +81% |
| `LLAMA_DRAFT_DETP` + `LLAMA_DRAFT_DETP_TRUNK_W=0.5` | 0（E，与岛 preset 互斥） | TG 78.85→84.52（+7.2%） | 草稿不对称去 TP；岛转正后被显式关闭 |
| `LLAMA_PREFILL_FAIR_SHARE` | 1 | 2×64K 聚合 40.8→74.73（+83%） | 修多流预填充串行（A 组悬案真凶） |

## 三、运行时 / 通信（run_test_server 基线）

| 开关 | 值 | 说明 |
|---|---|---|
| `GGML_CUDA_P2P` / `GGML_CUDA_ALLREDUCE=nccl` | 1 / nccl | NVLink P2P + NCCL AR |
| `GGML_CUDA_P2P_AR_DISABLE` | 默认设；图内 P2P AR 生效时 unset | 正确 3+3 岛序后图内 P2P AR 才真正生效（64K +0.0% 判噪声，保留开关） |
| `GGML_CUDA_DISABLE_GRAPHS` / `GGML_CUDA_GRAPH_OPT` | 按 preset | CUDA graph 控制 |
| `GGML_FT_GRAPH_CACHE` / `GGML_FT_GRAPH_CACHE_MIN_NODES=32` | 1 / 32 | 图缓存（TTL/容量 cap 已被双实验证伪） |
| `LLAMA_QWEN4EXP_HC_FUSION` / `LLAMA_HC_FUSED=2` | 1 / 2 | qwen4exp HC 融合 |
| `LLAMA_PLE_SHARD` / `LLAMA_PLE_SHARD_ROWS` | 1 / 0 | PLE 行分片读取 |
| `LLAMA_GPU_ARGMAX` | 1 | argmax 上 GPU |
| `LLAMA_MTP_N_UBATCH=32` / `LLAMA_MTP_PAD=5` | 32 / 5 | MTP 链微批/padding |
| `GGML_CUDA_TOPK_STABLE` | 1 | 确定性 top-k（R19 正确性修复之一） |

## 四、判死开关速查（保留代码，勿再立项）

`GGML_IK_FUSED_MOE`（CSE 幻觉）、`LLAMA_MWE_ATTN`（归因证伪）、`LLAMA_QSA_Q4V`（parity 分岔）、
`GGML_CUDA_FATTN_ONECAT`（数值语义不兼容打崩 MTP 链）、`GGML_CUDA_GRAPH_VARIANTS_CAP`（证伪）、
NCCL 调参全灭（PROTO/NCHANNELS/BUFFSIZE）、FP8 KV、KV sidecard、`--cache-reuse`（正确性雷区）。

## 五、KV 量化配方

| 配方 | 命令 | 结论 |
|---|---|---|
| 现役 | `--cache-type-k q8_0 --cache-type-v q8_0` | 基线 |
| **K8V4** | `--cache-type-k q8_0 --cache-type-v q4_0` | 262K 池 parity 零翻转、decode 无罚、prefill -4~5%；3.51GiB/卡 |
| F16 | `--cache-type-k f16 --cache-type-v f16` | TG 反而 +1.6% 但 262K 上不了（显存） |

注意：QSA sparse kernel 要求 q8_0+q8_0；K8V4 下 block-sparse 失效走 gather 回退（Q4V 262K 显存账见 `docs/campaigns/fattn-stream/REPORT.md`）。
