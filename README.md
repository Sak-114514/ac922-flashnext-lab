# ac922-flashnext-lab

> AC922 POWER9 / 6×V100 上跑 Qwen3.8-Flash-Next 的 build notes、战役报告与方法论档案。
> 本仓库只收录文档；代码在 [Sak-114514/llama.cpp](https://github.com/Sak-114514/llama.cpp) 的 `ac922-flashnext-v100` 分支。

## 这是什么项目

在一台 **IBM AC922**（POWER9 ppc64le）上，把 **Qwen3.8-Flash-Next GSQ-RCO**（ISTA-DASLab 量化的 GGUF）跑到可日常使用的速度。这个模型对推理栈很不友好：

- 架构 `qwen4exp`：**48 层 = 36 GDN（gated delta net 线性注意力）+ 12 QSA（稀疏注意力）**
- **512 选 10 MoE** + 共享专家，~167B 总参 / 6B 激活
- **PLE 稀疏嵌入表 16 头 × 2000 万行 ≈ 26.8-28.8GB（IQ4_NL）**——显存放不下，常驻 CPU 侧
- 模型文件 `IQ3_XXS-MTP-Q8.gguf` 73.2GB（含 1 层 MTP 草稿头）；GPU 实载约 46.4GB
- 执行底座 llama.cpp fork（上游 v0.5.0 / #29333 + qwen4exp 全架构支持），六卡行分片

全部优化由 Codex 与 ZCode 子代理编队以"微基准达标才准接线"的两阶段闸门推进，胜败全部留档——**判死清单与成功清单同样有价值**。

## 硬件 / 软件栈

| 项 | 值 |
|---|---|
| 整机 | IBM AC922，POWER9 ppc64le，176 线程，32GB RAM |
| GPU | 6× Tesla V100-SXM2-16GB（sm_70，NVLink 一代 mesh，每卡 ECC 后可用 15.47GB） |
| CUDA | 12.4 + driver 525（服务器侧 `/usr/local/nccl`） |
| 推理 | llama.cpp fork（[ac922-flashnext-v100](https://github.com/Sak-114514/llama.cpp/tree/ac922-flashnext-v100)），`--split-mode tensor --tensor-split 1,1,1,1,1,1` 六卡行分片 + 岛 3,3 结构 |
| 模型 | [ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ISTA-DASLab) |

## 性能汇总（口径列是第一公民）

**两种基准口径严禁混写、不可互相换算百分比**（详见 [recipes/performance-contracts.md](recipes/performance-contracts.md)）：
- **合成口径** = `matrix_runner --cases 1:8192` 重复文本 prompt，MTP 接受率虚高（≈100%），用于同臂 A/B 对照；
- **真实语料口径** = 真实文本 8K prompt / 256 输出，R19 锚点为 warm 1038.65 PP / **83.03 TG**。

| 指标 | 数字 | 口径 | 配置 |
|---|---|---|---|
| 单流 decode | **125.1 tok/s** | 合成 8K/512 | 岛 3,3 + P2P AR + 上游回移（正确 GPU UUID 岛序）+ R19 全家 |
| 单流 decode | **83.03 → 98.5 tok/s** | 真实语料 8K/256（R19 锚点 → 上游回移后热态 949.2 PP / 98.5 TG） | 同上 |
| 单流 prefill 8K | **920.6 tok/s** | 合成 8K/512 | 同上（vs 同轮冻结基线 648.9 = +42%） |
| 单流 prefill 64K | **1003.4 tok/s**，TG 108.8 | 合成 64K | 同上（基线 738.8 = +36%） |
| 单流 decode（前一配方） | 94.3 tok/s | 合成 8K/512 | + BF16_GEMV + MOE_REWORK，parity true |
| 4 路并发聚合 | **103.43 tok/s** | 合成，4×8K slots | 岛+H+FIFO+fair_share（B3，最优并发配方） |
| 6 路极限聚合 | 85.52 tok/s | 合成，6×8K | C1 |
| 2×64K 长上下文 | 36.6-40.8 → 74.73 tok/s | 合成 | 下界 → 补 `LLAMA_PREFILL_FAIR_SHARE=1` 修复预填充串行后（+83%） |
| MTP | 4 token 草稿链 | — | `--spec-type draft-mtp --spec-draft-n-max 3`（真实语料接受率 ~63%） |

注意：跨批次合成基线会漂移 17-20%，增量结论一律以各战役自身同臂 A/B 为准。

## 关键开关表

完整表见 [recipes/env-switches.md](recipes/env-switches.md)。核心五枚：

| 开关 | 默认 | 实测 | 状态 |
|---|---|---|---|
| `GGML_SM70_BF16_GEMV` | **1** | TG **+8.9%**，parity 逐位 | 生产 |
| `GGML_SM70_FUSED_PROJ` | **1** | 端到端中性（wave 融合 1.4-1.5×，但 BF16 可融合组不足） | 生产（技术储备） |
| `GGML_SM70_MOE_REWORK` | **0** | 开启即 **+1.27%** TG | 开关就绪 |
| `GGML_SM70_BF16_PREFILL_F16` | **1** | PP 8K **+6.2%** / 64K +5.8%，TG 持平，parity 逐位 | 生产 |
| `GGML_IK_FUSED_MOE` | 0 | 全模型 **-3.6%**（收益是编译器 CSE 幻觉） | **判死**留档 |

外加结构性 preset：`GGML_CUDA_MOE_GROUPED=1`（H，MoE 死块消除）、`LLAMA_QSA_POOLED_MULTI=1`（B，限长上下文）、`LLAMA_ISLAND_SPLIT=3,3`（岛内 3+3 分组）、FIFO 三件套 `GGML_META_ASYNC_COPY=1` + `GGML_META_FIFO_PIPELINE=1` + `LLAMA_PREFILL_KV_PAD=8192`（R18，PP +28.9%）。

## K8V4 KV 量化配方（262K 上下文）

```
--cache-type-k q8_0 --cache-type-v q4_0    # 现役 q8/q8 改为 K8V4
```

四臂对照实测（docs/campaigns/kv-262k/RESULTS.md）：**parity 零翻转、decode 无损（TG 持平）、262K 池零 decode 罚**；代价 prefill **-4~5%**（V-q4 写路径格式代价）。262K 全家桶显存 3.51GiB/卡（+63%），单流峰值 85.8% 放得下；3×262K 并发需先重算显存。定价：**花 4-5% prefill 换 2× 上下文**。

## 判死清单摘要（全部有实测证据，防止重蹈）

这是本仓库最宝贵的部分之一——每条死路都有数据、有归因：

- **岛间合并整步图**（nsys 实测候选上界 13.4% < 50% 准入门槛）
- **连续批处理短上下文**（2×8K 聚合 92.13 < 单路 100.04）
- **cooperative 单 launch**（+2.2% < 8% 门）、**wrapper 13.65ms 解剖**（可回收上界仅 13.4%，双重判明）
- **BF16→F16 全量权重转换 D 线**（TG -4~5%，V100 无 BF16 tensor core）
- **fused proj 端到端**（GGUF 运行时验证主力 wave 不存在——目标清单必须运行时验证，不能代码推断）
- **ik fused MoE 移植**（正确性 40/40 但收益是 CSE 幻觉，-3.6%；MoE ids 的 per-token distinct 语义两次让 uniform 微基准收益蒸发）
- **dense K-quant 流式化三代架构**（GGUF 原生 Q6_K 布局在 V100 无法就地解码打穿；唯一出路是离线 repack/行交织 R4）
- **fragment-native (mma.m8n8k4) 三家族**（自建 QPN 参照核反而慢 2×；BF16 已 92% 带宽上限）
- **dense-decode-once**（SASS 逐指令证明 nvcc 早已 CSE 掉 weight 侧解码，审计的 ×5 是激活侧 dp4a 本体）
- **GPU PLE 六卡分片 R20**（207→966 prefill 但 CPU/GPU 第 71 token 分岔；根因是 BF16 舍入传播的 logits 系统性差，可解但不值得修——dd 预热+R21 行缓存已零成本拿走收益）
- **MTP 多流图复用 / PAD_MULTI**（-44%；根因是 llama-context 单活动指针 + QSA 池化拓扑 churn）
- **MWE 成员负载均衡**（parity 全真但 64K PP 仅 +0.39%：生产注意力走 QSA gather 稀疏路径，头失衡归因被证伪）
- **`-sm row` / 六段流水 meta / v70 QSA prompt attention / 1Cat marlin（sm_75+）/ sglang-sxm2 换引擎（NVFP4 显存超限）**
- **QSA q4_0 V 支持**（8K PP +3.4% 真实但 parity token 0 分岔，kernel 数值 bug，方向未死）

死因全表见 `docs/FLASHNEXT-FULL-EXPERIMENT-REPORT.md` §1.4 与各战役报告。

## 仓库索引

| 仓库 | 内容 |
|---|---|
| [Sak-114514/llama.cpp](https://github.com/Sak-114514/llama.cpp)（分支 `ac922-flashnext-v100`） | 全部代码改动：qwen4exp 支持、SM70 kernel、岛结构、启动脚本 |
| [Sak-114514/super-ac922](https://github.com/Sak-114514/super-ac922) | vLLM 线（1Cat-vLLM fork，SM70 特化；29.19 tok/s 后弃线，档案保留） |
| 本仓库 | 文档：总报告、22+ 战役报告、4 份设计文档、口径学 |
| 模型 GGUF | <https://huggingface.co/ISTA-DASLab>（Qwen3.8-Flash-Next-GSQ-RCO） |

外部参考：[llama.cpp](https://github.com/ggml-org/llama.cpp)、[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)、[Strata](https://github.com/Niko1221/Strata)、[Luce](https://github.com/Luce-Org/luce-megakernel)、[Marlin](https://github.com/IST-DASLab/marlin)、[vLLM](https://github.com/vllm-project/vllm)。

### 本仓库结构

```
docs/                    全部报告（脱敏原样收录，文件名保留）
  ├── FLASHNEXT-FULL-EXPERIMENT-REPORT.md   ← 总报告（先读这个）
  ├── FINAL-REPORT-20261004.md              ← 阶段收官报告
  ├── campaigns/<战役名>/                    ← 22 个战役的 REPORT/RESULTS/INTEGRATION/DESIGN
  ├── designs/                              ← 4 份立项设计文档（DBO/EPLB/MTP-MULTI/PLE-PREFETCH）
  ├── lab-notes/                            ← 实验根目录方法论与交接档（TECH-FACTS/GPT-HANDOVER/BENCHMARK-METHOD 等）
  ├── onecat-six-run-20261007/ 等           ← 上游 1Cat-vLLM PR 深挖与会话档案
recipes/
  ├── env-switches.md                       ← 开关总表
  └── performance-contracts.md              ← 口径学（合成 vs 真实语料，为什么不能混算）
```

启动/压测脚本（`launch_integration_arm.sh`、`multi-concurrency-bench.sh` 等）随代码仓库分支发布，不在本档案库。

## 快速开始（编译，ppc64le + sm_70 + CUDA 12.4）

```bash
git clone -b ac922-flashnext-v100 https://github.com/Sak-114514/llama.cpp
cd llama.cpp
cmake -S . -B build-ac922 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_CUDA_NCCL=ON \
  -DNCCL_INCLUDE_DIR=/usr/local/nccl/include \
  -DNCCL_LIBRARY=/usr/local/nccl/lib/libnccl.so.2
cmake --build build-ac922 --parallel 6 --target llama-server llama-cli llama-bench
```

服务形态（要点）：`--split-mode tensor --tensor-split 1,1,1,1,1,1 --n-gpu-layers 999 --flash-attn on --kv-unified --cache-type-k q8_0 --cache-type-v q8_0 --spec-type draft-mtp --spec-draft-n-max 3`，GPU 顺序必须按 NVLink 拓扑用 UUID 固定成 3+3 两岛（`CUDA_DEVICE_ORDER=PCI_BUS_ID` 会交错混岛）。完整环境变量见 recipes/env-switches.md。

## 方法论遗产

1. **两阶段闸门**：微基准达标（真实形状 + 正确语义）才准接线
2. **MoE kernel 评估必须 distinct ids 对拍**（同一坑踩两次的教训）
3. **目标清单运行时验证**（GGUF tensor dump / graph census），不信代码推断
4. **三大数据陷阱**各有实测案例：微基准 L2 污染、ids 语义、CSE 幻觉
5. **子代理编队**：并行战役 + GPU 分卡 + 时间盒 + 诚实报败纪律（Codex/ZCode 协作过程在报告中如实保留）

## License

MIT（见 [LICENSE](LICENSE)）。报告中的性能数字均标注口径与证据文件。
