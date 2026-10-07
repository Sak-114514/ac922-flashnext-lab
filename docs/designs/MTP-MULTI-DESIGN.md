# 多序列 GPU MTP 链（多流 unlock）— 可行性论证与设计（2026-10-06）

> 结论先行：**值得做，优先级 P1（MWE 之后、DBO 之前）**。多流聚合 75 vs 单流 125 的缺口中，可归因于"快链全家桶被 n_seq>1 禁用"的部分远大于 mtp-msgraph 判死战役的 +6-7% 上限（那是另一个池子：图形状重建）。本设计把 fold/carry/chain/argmax 四件套按 lane 多路化，并按 Codex 提醒把**动态深度接入快链**（现状 `!mtp_adaptive` 是硬门）。
> 判死先例对照（为什么这次不同）：mtp-msgraph 判死的是"跨槽图复用/padding/KV 桶粗化"三条修复路线；本设计不碰那三条，正面解决的是**快链状态的单序列数据结构**——不同的改动面，且其判死报告恰好为本设计提供了必需的形状实测数据。
> 本轮零 GPU、零生产改动；行号均亲核于 `/home/debian/ac922-llama-next/sources/llama-v100-v3` @ ac922-flashnext-v100（134 脏文件态）。

---

## 1. 论证（占全文 >30%）

### 1.1 现状与缺口的三层拆解

数字链（全部 [M] 有档案）：

| 场景 | 聚合 TG | 出处 |
|---|---:|---|
| 单流 8K（快链全家桶：fold+PAD+carry+chain+DETP+图） | **125.06** | codex-upstream-20261006 bundle-r19 |
| 2×8K 多流聚合 | **~75** | Codex final-mtp-multi（manifests/final-mtp-multi*） |
| 旧基线 2×8K（无快链时代，B+H） | 69.6（34.2+35.4） | speed-bundle 冻结基线 |
| 单流无 MTP | 43.1 | TECH-FACTS |

缺口 = 2×125 − 75 = **175 tok/s 的名义缺口**。多流永远到不了 2× 单流（注意力/KV/调度共享），但 75 甚至低于"两路各自无 MTP"(~2×43=86 的量级考虑 MTP 增益后合理落点 ~110-120)。缺口来源分三层：

1. **快链禁用（本设计的目标池）**：`n_seq>1` 时 fold 关闭（`common/speculative.cpp:1511-1512`）→ PAD/GPU carry（`src/llama-context.cpp:1150`：`n_seq_max() != 1` 直接 return false）/GPU chain（`:1184`）全部失活。单流实测这些资产合计贡献约 +25%（R19 前后 100→125 的主项是组合，快链是其中大头）。多流下两路每路退回"慢链 + 逐序列 catch-up 分帧"。
2. **draft 图交替重建**（mtp-msgraph 已量化）：catch-up(10 行) ↔ draft-step(2 行) 结构性交替，`gf_res_prev_active` 单活动指针（`src/llama-context.cpp:1773` 一带）每轮 2 次拒绝复用 → 全量重建 2×2.7ms/轮，池子 **+6-7% 上限**（mtp-msgraph 判死结论，本设计不重复收割，但其数据被 §2.3 复用）。
3. **验证步聚合 KV 计费**（mtp-msgraph §4-3）：50-62ms vs 单流 24-26ms，聚合统一 KV 注意力固有，需 per-slot KV 隔离才能再压——**超出本设计范围，不做**（记录为远期）。

### 1.2 收益上界（通信/显存/调度三重约束下的诚实算）

- 约束 1（调度）：fair-share 下两路 prefill/decode 交错，decode 窗口内两路的 draft/target 图交替发射——POWER9 单核发射能力是隐性上限。参照：单流 decode 125 时 host 发射已非零压力（R16 拆账：wrapper 12.7ms/步）。
- 约束 2（显存）：快链 embedding replica（LLAMA_MTP_GPU_EMBED，~300MB 级）与 carry buffer 需 ×n_seq 份；速度候选配置单卡最小空余 2281MiB，n_seq=2 可容、n_seq=4 需预算（见 R2）。
- 约束 3（接受率）：多流两路各自的真实接受率 ~64%（367/573）；快链不改变接受率（贪心等价），收益纯来自每步时间。
- 上界估算：把第 1 层缺口全部收回 = 两路各自接近单流快链的每步效率。用 mtp-msgraph 的实测结构折算：2×64K 联合 decode 窗口 11.5s 中，draft 慢链 vs 快链的每步差（单流 R19 前后 draft 3.4-4.7ms → 2.2ms 量级）× 交替步数 ≈ 池子 15-25%。**预期 2×8K 聚合 75 → 95-110**；两路合计 MTP 效率恢复后逼近显存允许的并发上限。
- 不做的损失：多流场景（K8V4 组合验证的目标形态）聚合吞吐停留在 75，比单流跑两请求还差（125+125 串行=等效聚合仍高于 75 并行？不——串行两路合计吞吐仍 125，并行 75 更差！）。**多流现在是负收益功能**，这是必须 unlock 的直接理由。

### 1.3 成本与风险

| 项 | 评估 |
|---|---|
| 改动面 | 中：`common/speculative.cpp`（fold/pad_batch/chain 驱动多路化，~250 行）+ `src/llama-context.cpp`（carry/chain 张量按 lane 扩维，~200 行）+ `src/llama-kv-cache*.cpp`/hybrid-idx 的 ubatch 切分交互（~50 行验证性改动）。不动 meta backend（避开 2909 断言红线） |
| 失败先例对照 | mtp-msgraph 三路线判死≠本路线；但其暴露的硬约束（hybrid-idx 把多序列 batch 按序列切 ubatch，`src/llama-memory-hybrid-idx.cpp:183` trailing-rows 规则）本设计必须正面处理（§2.4） |
| 验证周期 | 中：mtp-chain-fixture（服务器 `experiment/mtp-chain-fixture/` 已有）扩多 lane + 整模型多流 parity（两路 512-ID 各自逐位）+ 真实语料多流回归 |
| 与 Codex 树冲突 | Codex 清单第 1 项与本设计同题——**必须分工**（§4），否则重复开工 |

### 1.4 "不做也行吗"

不做：多流聚合 75（负收益），K8V4 组合验证的多流口径全部失真，生产只能单流跑——6 卡系统退化为"单请求 125"的服务能力。做了：多流回到 ~100 聚合，2-4 路并发服务成为正收益。**这是四个方向中唯一"修复已存在功能的倒挂"性质的工作**，且不依赖 MWE/DBO 先行。判决：值得立刻做（与 MWE 并行开工也成立，改动面无交集：MWE 在 llama-model.cpp，本设计在 speculative/llama-context）。

---

## 2. 设计：per-lane 快链 + 固定形状批量图

### 2.1 现状单序列假设的精确定位（行号）

| 组件 | 位置 | 单序列假设 |
|---|---|---|
| fold 开关 | `common/speculative.cpp:1511-1512` | `fold = (...) && n_seq == 1 && !is_mem_shared && !chain_heads` |
| GPU carry 输入 | `src/llama-context.cpp:1150` | `n_seq_max() != 1 → return false`（carry 张量 `{n_embd_out, rows=n_pad}` 固定 GPU 地址，无 lane 维） |
| GPU chain | `common/speculative.cpp:1528-1531` | 要求 fold + n_pad>0 + carry + `!mtp_adaptive` + p_min≤0 + 无 backend 采样 |
| chain token 历史 | `src/llama-context.cpp:1239-1243`（begin/fill 32 token） | 单条 32-token pinned 历史 |
| draft batch | `common/speculative.cpp:1467` | `llama_batch_init(n_b, n_embd, /*n_seq_max=*/1)` |
| pad_batch | `common/speculative.cpp:1420-1439` | `seq_id[last][0]` 单 seq、整批 pad 到 n_pad |
| verify 图形状 | mtp-msgraph 实测 | 双槽 catch-up 恒 (10,0)=Σ(n_drafted+1)、draft-step 恒 (2,0)——形状其实已固定，重建源于 `gf_res_prev` 单活动指针 |

### 2.2 数据结构多路化

```
# carry 输入: {n_embd_out, n_pad}                → {n_embd_out, n_pad * n_seq}
#   行寻址: lane l 的第 r 行 = l*n_pad + r        （一次 H2D/D2D 快照覆盖两路）
# chain tokens: {n_pad}                          → {n_pad * n_seq}（I32）
# pinned host 历史: 32                            → n_seq × 32（end_mtp_gpu_chain 按 lane 段读出）
# argmax: ARGMAX_SPLIT 已是 per-row 语义（meta init 注入 row offset，
#   ggml-backend-meta.cpp:1766-1784 op param 注入），logits {vocab, n_seq} → tokens {n_seq} 天然批量化
# GPU embedding replica: 单份共享（只读表，get_rows 行 id 含 lane 偏移即可，无需复制）
```

配置门修改：`llama-context.cpp:1150` 的 `n_seq_max() != 1` 改为 `n_seq_max() > LLAMA_MTP_MAX_LANES`（建议 MAX_LANES=4）；`speculative.cpp:1511` 的 `n_seq == 1` 删除，fold 的 catch-up 逻辑按 lane 独立记账（`def_tok/def_pos/def_embd/def_seq/def_keep` 已是 per-lane vector 化前体，扩成 `vector<vector<...>>`）。

### 2.3 固定 verify 图形状（mtp-msgraph 移交项的解）

mtp-msgraph 判死报告的关键数据：双槽满深度时 catch-up 恒 10 行（2×5）、draft-step 恒 2 行——**形状集合有限且已知**。判死的路线 B（padding 到单一形状）败于"用 5× 注意力换 replay"+hybrid-idx 把 10 行切成 2+8。
本设计的解：**不用 padding，用双形状双图驻留**——catch-up 图（n_seq×(n_max+1) 行）与 draft-step 图（n_seq×1 行）各捕一份，交替时靠 §2.5 的 gf_res_prev 多槽化避免重建。与判死路线 A（跨槽复用撞 meta 2909）的区别：路线 A 复用图对象，本设计**两图各自独立 capture、独立 arena**（draft 图小，arena 代价可忽略——mtp-msgraph 实测 draft 图 compute 重建成本 2.7ms 中主机 build 只占 0.15ms，arena 重 alloc ~1.0ms 是大头，独立 arena 消掉它）。

### 2.4 hybrid-idx 切分交互（判死路线 B 的教训强制项)

`llama-memory-hybrid-idx.cpp:183` 的 trailing-rows 规则把多序列 batch 按序列切成多 ubatch——**多 lane 的 draft step 必须 pad 成"每 lane 等行数"的合法多序列 batch 且不被切分**。方案：draft step 的 batch 构造改为显式 n_seq 行（每 lane 1 行，seq_id 显式），catch-up 为 n_seq×(n_drafted+1) 行——两形状都满足"每序列连续行"的 hybrid-idx 约束（切分只发生在序列间间隙不齐时；等行数 batch 无间隙）。pad_batch() 多路化：每 lane 独立 pad 到 n_pad（pad 行复制该 lane 末行，`seq_id` 用该 lane 的）。

### 2.5 动态深度接入快链（Codex 提醒的坑，正面解）

现状：`common/speculative.cpp:1529` `!this->params.mtp_adaptive` 是 chain 硬门——adaptive 直接开启会**静默退回慢链**（SPC_WRN 一行日志）。且 adaptive 构造期把 n_max 钳到 ≤3（`:1434`）。
设计：**固定图形状按 n_max 上限捕获，实际深度用 mask 表达**：
- catch-up 图固定 n_seq×(n_max+1) 行；lane l 实际接受 a_l 行时，多余行的 logits/hidden 由图计算但被丢弃（mask 在 host 侧读出后裁剪，`end_mtp_gpu_chain` 已按 count 截断，天然支持）。
- 多算成本：深度 n_max=4 图 vs 实际深度 2 的 lane 多算 2 行 draft 前向——单层 draft 前向 2.2ms/5 行 → 每多 2 行 ~0.9ms，被"图不重建省 2.7ms"覆盖，净赚。这反转了 mtp-msgraph 路线 B 的败因（那次是 5× 注意力；draft 是 1 层小图，代价小一个量级）。
- adaptive 状态（`common_speculative_mtp_adaptive`，`:1332-1345`，ema/limit/accept）已是 per-seq vector（`adaptive.resize(n_seq)`，`:1436`）——无需改动，只解锁门槛。

### 2.6 工作量分解

| 阶段 | 内容 | 估计 | 验证门 |
|---|---|---|---|
| W1 | fixture 先行：`experiment/mtp-chain-fixture` 扩 2-lane（carry 双份行、chain 双条历史、batch argmax 一致性） | 1-2 天 | fixture 逐位（含 lane 交叉污染用例） |
| W2 | carry/chain 张量扩维 + 配置门解锁（2.2） | 2-3 天 | 单流 8K parity 逐位（回归保护：n_seq=1 路径字节级等价） |
| W3 | fold/pad_batch/batch 多路化 + hybrid-idx 等行数 batch（2.1/2.4） | 2-3 天 | 2×8K 双路各自 512-parity |
| W4 | 双形状双图驻留（2.3）+ gf_res_prev 按 (ctx, shape) 二槽 | 2 天 | mtp-msgraph 的 rebuilt/32 ≤2 门复刻 |
| W5 | 动态深度 mask 化（2.5） | 1-2 天 | adaptive on/off 双跑 parity + 接受率一致 |
| W6 | 性能收口：2×8K/4×8K/2×64K 聚合 + 真实语料多流 | 1 天 | 门：2×8K 聚合 ≥ 95（上界论证 §1.2） |

### 2.7 风险清单

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 显存：n_seq=4 时 carry 4×（512KB 级，无虞）+ 双图 arena + embedding replica 共享（单份）——主要压力仍是 KV | W3 后实测 n_seq=2/4 显存账；超限时 MAX_LANES 收缩 |
| R2 | verify 侧聚合 KV 计费不随本设计改善（mtp-msgraph §4-3 判"不立项"） | 预期收益算式已按此折算（75→95-110 而非 2×125） |
| R3 | 多 lane 交叉污染（lane A 的 hidden 写进 lane B 的 carry 行） | W1 fixture 专设污染用例；行寻址 l*n_pad+r 无交集代数 |
| R4 | hybrid-idx 切分规则有未文档化分支（判死路线 B 的 2+8 切分形态） | W3 先写只读 trace：2-lane batch 的实际 ubatch 切分日志，确认等行数假设 |
| R5 | 图 × 形状 × lane 组合数膨胀（图缓存 cap 4） | `GGML_CUDA_GRAPH_VARIANTS_CAP` 提到 8（env 已存在，R16 资产）；draft 图小、捕获成本低（mtp-msgraph：capture 537 次无感） |
| R6 | backend_sampling 与 chain 互斥门（`:1528`）在多流 server 路径被打开 | 保持互斥不变，多流 preset 显式钉死 greedy（与单流快链同前提） |

---

## 3. 最小验证方案汇总

1. **W1 fixture**（见上）；2. **单流回归保护**：每个 Wx 合入点跑 1×8K/512 parity（n_seq=1 不得有任何位差）；3. **多流 parity**：2×8K 两路 512-ID 各自逐位 vs 各自单流参考；4. **性能门**：2×8K 聚合 ≥95（fail 则拆层归因：快链回收多少/重建消除多少，用 mtp-msgraph 的 rebuilt/32 计数器）；5. **守护**：温度 58°C、GPU 显存余量 512MiB（复用 Codex run_arm.py 框架）。

## 4. 与 Codex 清单的分工建议

Codex 清单第 1 项（多序列 MTP 链）与本设计同题，其描述（"分离每路 token/hidden state，批量 GPU argmax、反馈与 padding 的多路化；动态 MTP 深度也应接入这条链"）与本文件 §2.2/2.4/2.5 一致——**共识已存在，需要的是切分**：

- **建议本侧（ZCode 线）**：W1-W3（数据结构多路化 + hybrid-idx 交互）——改动在 `common/speculative.cpp` + `llama-context.cpp` 前半，与 MWE（llama-model.cpp）无冲突。
- **建议 Codex 侧**：W4-W5（双形状双图驻留 + 动态深度 mask 化）——它持有 mtp-msgraph 战役的全部形状实测与 gf_res_prev 上下文，且其正在查的 stage pipeline 失败（DBO-DESIGN §2）与图驻留机制共享认知。若 Codex 无余力，W4-W5 顺延本侧，W1-W3 独立可交付（快链多路化本身已收回大半池子）。
- **接口对齐点**：① carry 张量行布局 `lane*n_pad + r` 写入两份文档；② `LLAMA_MTP_MAX_LANES` env 名与默认值；③ W4 的二槽化若触碰 `gf_res_prev` 结构，须与 Codex 的 stage pipeline 复查（DBO-DESIGN §2 C5）协调，避免两线同时改 `llama-context.cpp:1773` 区域——**该文件是冲突热点，开工前拉一次对表**。
- **共用数据**：`manifests/final-mtp-multi*/`（多流骤降日志）、`experiment/mtp-chain-fixture/`、mtp-msgraph 的 shape 直方图（concurrency/msgraph/*.json）。
