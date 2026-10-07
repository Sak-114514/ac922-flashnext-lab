# performance-contracts — 口径学（为什么两套数字不能混算）

> 本项目所有性能数字都隶属两套基准口径之一。**跨口径相除/相减得到的百分比全部无效。**
> 这不是免责声明，是被实测反复教育出来的方法论（见下"翻车记录"）。

## 两套口径的定义

### 合成口径（synthetic，A/B 对照专用）

- 工具：`matrix_runner --cases 1:8192`（decode-first 压测 `benchmark_server.py` 走 llama-server `/completion`）
- prompt 是**重复文本** 8K，输出 **512** token，greedy、固定 seed、`ignore_eos`
- 后果一：**MTP 接受率虚高（≈100%）**——重复文本对 MTP 草稿是"作弊题"，真实语料只有 ~63%
- 后果二：重复文本会**掩盖冷/热差异**（冷热差仅 8%，高多样性 prompt 才暴露真实 TTFT 问题）
- 用途：**只做同臂 A/B**（同一批次、同配置、只动一个变量）。六战役全部用此口径给增量

### 真实语料口径（real-corpus，对外报价专用）

- 真实文本 8K prompt / **256** 输出
- 锚点：**R19 = warm 1038.65 PP / 83.03 TG**（`final-results/ROUND19`，修复 chain 同步缺口/stale 携带/确定性 top-k 之后）
- 同代配置的合成口径锚点约 100 TG——比真实语料高 ~20%，主要来自 MTP 接受率差
- 上游回移战役后热态实测：949.2 PP / **98.5 TG**（真实语料）

## 三条铁律

1. **增量只认同臂 A/B**。跨批次合成基线漂移 17-20%（同代码不同轮），任何 "+X%" 必须来自同轮对照臂。
2. **跨口径不可换算**。例：合成 125.1 与真实 83.03 之比不是任何"真实开销系数"——两口径的 MTP 接受率、prompt 分布、输出长度都不同。
3. **聚合口径单独标注**。多流数字必须写明是 `decode_tps_sum_per_flow_server`（各流 server decode 之和）还是 `decode_tps_wave_wall`（整波墙钟聚合）——后者才是完成吞吐，**不得把逐流速率当聚合报告**（`docs/lab-notes/BENCHMARK-METHOD.md`）。

## 翻车记录（为什么要有这份契约）

- 单路整合 94.3（合成）一度被直接对比真实语料锚点 83.03 讨论百分比——总报告 rev3 修订时全部改为分列标注。
- B 开关（QSA pooled）在合成 2×64K 上 +93%，8K 场景却倒退（20.9 vs 34.2）——合成口径的长上下文行为不能外推短上下文。
- MTP 多流"接受率"在冻结重复臂上饱和 1.000，真实文本混杂续写族根本无法测准（K8V4 验收因此改用 parity 零翻转为主证据）。
- 1Cat-vLLM 线 prefill 819.2 与 decode 5.18 出自同一 run 的不同阶段——**跨阶段亦不可比**。

## 报告数字时的模板

```
<数值> tok/s（<合成|真实语料>口径，<8K/512|8K/256|...>，<单流|N路聚合 wave-wall>，
臂对照 <基线→实验>，批次 <manifest/日期>，parity <逐位|relerr|n/n>）
```

示例：`125.1 tok/s（合成口径 8K/512，单流，vs 同轮冻结基线 93.7，批次 codex-upstream final，parity 512 token 逐位一致）`

## 各战役报告的数字可回源性

总报告（`docs/FLASHNEXT-FULL-EXPERIMENT-REPORT.md`）声明：所有数字可回源到文内引用的工件文件
（manifests / RESULTS / nsys trace），无源可查的会话推断均已标注。本档案库收录文档层；工件原始数据在代码仓库分支侧。
