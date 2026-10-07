# 最终报告：AC922 Qwen3.8-Flash-Next 速度战役（2026-10-03~04）

## 一句话答案
**做到多快**：单路 **94.3 tok/s**（合成口径 8K/512；真实语料锚点 83.03，跨口径不可算百分比）；4 路并发全开**聚合 103.4 tok/s**；6 路极限 85.5。
**预估还能多快**：单路剩余 +2-4%（小池子），并发配方优化 +10-20% 空间，R4/repack 是最后的单路大鱼（+3-5%）。

## 1. 单路战绩（全部实测，parity 全过，R19 基线 83.03）

| 优化 | 提升 | 状态 |
|---|---|---|
| BF16 GEMV 单 kernel（kernel_sm70_bf16.cuh）| +8.9%（111.2 vs 100.2 同口径）| ✅ 默认开 |
| MoE Q2_0 重组（kernel_sm70_moe.cuh）| +1.16% | ✅ 开关就绪 |
| **两开关整合实测** | **94.3 tok/s（93.46/95.20）** | ✅ matched=true |

## 2. 多并发矩阵实测（8 场景，RESULTS.tsv）

| 场景 | 配置 | 聚合 decode | 备注 |
|---|---|---|---|
| **B3 4×8K** | 岛 C+H+FIFO 全开 | **103.43** 🏆 | 四路各 25-27 |
| B1 4×8K | 岛 C+H | 94.94 | |
| B1 2×8K | 岛 C+H | 91.15 | 与历史 C18 92.1 一致 |
| B3 2×8K | +FIFO | 92.50 | |
| C1 6×8K | 岛 H+B 极限 | 85.52 | 六路各 13-16 |
| C2 6×8K | 六路 H+B | 81.62 | |
| B2 2×8K | B 强制 | 66.28 | 证实 B@8K 劣势 |
| A1-A3 2×64K | 长上下文 | 36.6-40.8 ⚠️ | 双流不均衡(2.7/38)，历史锚点 77.9 未复现，视为下界待查 |

**生产配方**：岛 3,3 + H(MOE_GROUPED) + FIFO 三件套 + fair_share + slots=4（B 开关留默认门）
**A3 的 FIFO 对 prefill 增效显著**：PP 478→592→864（+81%）——长上下文 TTFT 大改善

## 3. 攻坚失败但归档的战役（两次诚实报败 + 判死清单）

| 战役 | 死因 | 资产 |
|---|---|---|
| F: dense K-quant 流式化 | GGUF 原生 Q6_K 布局解码依赖链压不住（三代架构实测），**佐证 R4/repack 是唯一出路** | 骨架 78.8% 验证 + 方法链 |
| I: ik fused MoE 移植 | 收益全部是 CSE 幻觉（ids distinct 语义下消失），实测 -3.6% | kernel 留树默认关 |
| 判死（历史+本轮）| 岛间合并图/连续批处理/cooperative/wrapper 13.65ms/IQ3_S 解码 | 记录在案 |

**跨战役教训**：MoE ids 的 per-token distinct 语义两次让 uniform 微基准收益蒸发——MoE kernel 评估必须 distinct ids 对拍。

## 4. 剩余空间预估（诚实版）

| 方向 | 预期 | 依据 |
|---|---|---|
| R4 行交织/离线 repack | +3-5% | F 证明布局是病根 + ik PR 落地先例 + F 骨架 78.8% |
| 输出头 nc5 + 杂项小池 | +0.5-1% | 审计 |
| A 组长上下文不均衡修复 | 并发聚合翻倍潜力（36→78+）| 历史锚点证明可达 |
| 并发配方调优（slots/池/B门）| +10-20% 聚合 | B3 4×8K 超预期 24% 的同向证据 |

**单路天花板判断：~97-99**（94.3 + R4 + 小池）；并发聚合天花板 ~120-140（B3 配方 + A 组修复 + 调优）。

## 5. 资产索引
- 战役记录：reports/sm70-kernel-campaign-20261003.md、各 INTEGRATION.md/REPORT.md
- fork 开关：GGML_SM70_BF16_GEMV(默认1)、GGML_SM70_MOE_REWORK(默认0)、GGML_SM70_FUSED_PROJ(默认1，中性过门)、GGML_IK_FUSED_MOE(默认0)
- 并发：concurrency/PLAN.md + RESULTS.tsv + multi-concurrency-bench.sh
- 启动器：scripts/launch_integration_arm.sh（单路全开）
