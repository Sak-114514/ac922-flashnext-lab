# DBO 双微批重叠 — 可行性论证与设计（2026-10-06）

> 结论先行：**值得做，但排后（MWE 之后）**。收益上界与 MWE 同源（都是吃 AR 自旋的 ~20% 墙钟），但 MWE 用 80 行改放置消除自旋，DBO 用一套新并发机制填充自旋——前者必须先行，否则 DBO 的收益会被 MWE 吃掉大半且数据无法归因。
> Codex 的 stage pipeline 失败（真实 8K token-37 分岔）不构成对本方向的否决：失败实现走的是"单图指针轮换重绑"路线，与 vLLM DBO 的"帧完全独立"原则相反。本文件给出根因候选清单（§2）与正确的本地化设计（§3）。
> 参考：vLLM DBO 引入 PR [vllm-project/vllm#23693](https://github.com/vllm-project/vllm/pull/23693)、全图捕获 PR [#51700](https://github.com/vllm-project/vllm/pull/51700)、[官方设计文档](https://docs.vllm.ai/en/latest/design/dbo)。本轮零 GPU、零生产改动。

---

## 1. 论证（占全文 >30%）

### 1.1 我们要藏的通信是什么（与 vLLM 的差异）

vLLM DBO 藏的是 MoE 稀疏 all-to-all（DeepEP，DP+EP 部署）；我们没有 all-to-all，要藏的是**岛内 3 成员 AllReduce 的自旋等待**。实测（EPLB-DESIGN.md §1.2-1.4，同一次 nsys 复挖）：

- 每活跃 GPU 约 50% 时间在 AR 内自旋（等最慢成员）；AR 自旋占 64K prefill 墙钟 20-22%。
- 自旋 = 到达错位（corr 0.83），错位来自成员间计算不对称（other 段 busy spread p50 3.9ms），**不是带宽**（线时间地板 ~350µs）。

DBO 的作用机理（本地化语义）：微批 A 的成员 X 在 AR 自旋时，同一 GPU 上跑微批 B 的计算。NCCL LL AR kernel 只占 1-2 个 SM（自旋 flag），SM 不会被打满——微批 B 的计算可以物理并行。这成立的前提是**微批 B 在独立 stream + 独立图**上（见 §3.2 的关键约束）。

### 1.2 收益上界（诚实算，两个口径）

口径 A（仅填 AR 自旋，岛内）：假设能把自旋时间的 f 比例填成有效计算，prefill 墙钟 −0.20×f/2（自旋分布在两岛串行相位中，填一半相位）→ f=0.7 时 **PP +7~8%**。注意：填进去的 B 计算本身也在推进 B 的 prefill，所以吞吐口径的收益 ≈ 自旋占比 × 填充率 ≈ **+14%（PP 1003→~1140）** 上限。
口径 B（DBO 兼做跨 ubatch 段间流水 = C2'，岛间）：微批 A 在岛 1 时微批 B 在岛 2 → 攻击岛间串行的 ~50% 空转，理论上限 ~2×（prefill-parallel 报告 §5.2 的 C2' 分析：8K smlayer 实测 +14.6% 是其无流水子集）。**DBO 与 C2' 在工程上是同一件事的两个相位调度**，本设计合并陈述。
口径声明：64K 单流 nsys；多流下 prefill 相位本就与另一流的 decode 交错（fair-share），DBO 收益要重测。

### 1.3 成本、风险与失败先例

| 项 | 评估 |
|---|---|
| 改动面 | 大：scheduler 双图驻留 + 每微批独立 input buffer + 事件管理 + FIFO 快照扩展（估计 600-1000 行，跨 `ggml-backend.cpp`/`ggml-backend-meta.cpp`/`llama-context.cpp` 三文件） |
| 失败先例 | ① Codex stage pipeline（2026-10-06）：合成 8×128 logits PASS（1.16e-10）但整模型真实 8K 第 37 token 分岔——小 fixture 不覆盖整模型依赖。② mtp-msgraph 路线 A（跨槽图复用）撞 `ggml-backend-meta.cpp:2909` 断言（meta 图缓存假定张量对象新建）——**双图驻留是红线邻区**。③ R18 的 FIFO 单微批版本过门（+28.9% PP）但仅 1×8K 验证 |
| 与 Codex 树的冲突 | Codex 正在查 stage pipeline 失败（其清单第 5 项）；本设计与其共享 ggml-backend.cpp 的 fifo 结构区，需协调先后 |
| 验证周期 | 双微批正确性门 = 合成 fixture + 整模型重复文本 parity + 真实语料重复一致性（Codex 教训：三层都要，fixture 过≠整模型过） |

### 1.4 与 MWE 的优先级关系（为什么排后）

1. MWE（EPLB-DESIGN §2）修齐成员负载后，AR 自旋的 20-22% 里可消除 ~15-18%；DBO 剩余可填的部分 ~5-8% + 岛间流水的大头。**先 MWE 后 DBO 能把两者的贡献干净归因**；反过来 DBO 先行会让 MWE 的 A/B 失去稳定基线。
2. MWE 是 80 行单文件改动；DBO 是 600-1000 行三文件改动。同一个人-week 先花在 MWE。
3. 若 MWE 后重测 nsys 显示 AR p50 已 <1ms，DBO 的口径 A 部分自动缩水，只剩 C2'（岛间流水）作为主收益——届时 DBO 设计应直接按 C2' 形态（跨 ubatch 段间流水）立项，不做岛内双微批。

### 1.5 "不做也行吗"

不做 DBO：损失 = AR 自旋的填充机会（~14% PP 上限）+ 岛间流水（上限 ~2× 但被调度器复杂度折扣）。有 MWE + smlayer（8K 场景 +14.6% 已实测可用）打底，不做 DBO 的系统仍能达标（64K ~1180-1250）。但 64K 的下一步上限（>1500）只有 C2'/DBO 能打开。判决：**值得但排后**（MWE 落地并重测 nsys 之后，按 §1.4-3 的数据决定做岛内 DBO 还是直接做 C2'）。

---

## 2. Codex 失败根因候选清单（帮助唯一定位，按嫌疑排序）

失败签名：tiny 跨 SYS 双目标层 fixture 全 PASS（4-copy 轮换、reset/tail/drain，maxAbs 1.164e-10）；8K/64K 重复文本 512-parity PASS；**固定真实 8K 冷/热两次输出从第 37 个 token 分岔**；关闭新流水后同配置 512/512 一致（350/644 接受）。源码：`experiment/codex-upstream-20261006/artifacts/rejected-stage-pipeline/stage-pipeline.patch`（531 行，本轮已通读）。

实现机制回顾：单图复用 + `ggml_backend_sched_next_copy()` 在帧间把 split INPUT 的 `node->src[k]`/`view_src`/`buffer` **原子重绑**到下一个 copy 槽；GPU peer copy 槽复用前在 host 上 `event_synchronize` 等前消费者。

| # | 候选 | 机理 | 判别实验（最小成本） |
|---|---|---|---|
| C1 | **CUDA Graph 捕获地址 vs 指针重绑**（头号嫌疑） | CUDA graph 捕获时烘焙 input 的设备地址；replay 读捕获地址。`next_copy` 重绑 tensor 结构后，若 host 把新帧数据 stage 进新槽、而某条 replay 路径仍指向旧槽地址（或反之），replay 读到陈旧/半新数据 → 数值分岔且仅在特定 replay/重绑时序组合下出现（真实语料的路由/形状序列才能触发，重复文本触发不了） | 对比实验：`GGML_CUDA_GRAPHS=0` 关图重跑真实 8K——若分岔消失即坐实 |
| C2 | **非 INPUT 张量的帧间写冲突** | 快照只覆盖 host INPUT leaf；两帧在飞时，帧 N+1 的 split 计算可能写帧 N 仍在读的中间张量（galloc arena 单份地址）。事件链只挂在 split 边界（`events[backend][copy]`），图内部无保护 | 双帧在飞时用 compute-sanitizer racecheck 跑一次 8K（分钟级） |
| C3 | **host 阻塞等待位置错误** | GPU peer copy 槽复用用的是 host `event_synchronize`（阻塞发射线程）而非目的流上的 `event_wait`——若上一帧对目的槽的读尚未入队就 synchronize，等待提前返回 → 拷贝覆盖在飞读取 | 日志核对：`slot_waits` 计数与分岔 token 的时序对齐（arm 日志已有该计数） |
| C4 | **meta split-state 缓存陈旧** | 重绑后 tensor 的 split_state cache（`split_state_cache`，按 tensor 指针键控）与旧 buffer 关联；uid 键控的 meta rebuild 清容器假定张量新建（mtp-msgraph 路线 A 的 2909 断言同源） | `GGML_META_DEBUG=1` + `LLAMA_CALLBACK_TRACE=2000` 重跑分岔场景，比对第 37 token 前后的回调 sums |
| C5 | **与请求重置/图缓存交互**（Codex 诊断倾向） | 诊断配置是"新实例直接跑真实语料"，失败配置先跑了 8K/64K 回归——请求历史不同。stage 路径的 `sched->fifo->snapshots.clear()`（reset 时）与图变体缓存（FreeToken variant cache）在多请求序列下的状态残留 | 同一实例按失败顺序重跑，但把回归请求换成 2 个真实 8K——若第二个就分岔，排除回归请求形态的影响 |
| C6 | **跨 SYS 边界拷贝的同步 fallback 判定** | 补丁自述"跨 SYS 的复制仍可能走同步 fallback"；fallback 与异步路径在不同帧序下交替时，事件配对可能错位 | 统计失败 run 日志中 sync fallback 出现次数与分岔 token 位置的相关性 |

建议执行顺序：C1（一次 env 对照，最便宜）→ C3/C6（纯日志）→ C2（sanitizer）→ C4/C5（插桩）。**任何一坐实，DBO 重设计（§3）直接吸收其教训**。

---

## 3. 设计：帧独立的岛内双微批（vLLM 原则的本地化）

### 3.1 原则（从 vLLM 抄什么）

vLLM DBO 三要素（#23693 + #51700）：① 每微批独立 `UBatchContext`（各自 input buffer + attention metadata）；② 两 worker 线程乒乓，微批 A 通信时微批 B 计算；③ **FULL CUDA graph capture 按微批形状固定化捕获**（跨线程 overlap 不在单图内自然可捕获，靠两份整图交替 replay）。

本地化翻译：**两份完整图实例（gf_A/gf_B），各自绑定固定 copy 槽（copy 0 / copy 1），永不重绑指针**——这是与 Codex 失败实现的根本区别（它重绑单图，我们交替两图）。

### 3.2 关键约束：为什么必须双 stream + 双图

同岛 3 成员上，微批 A 的 NCCL AR 自旋 kernel 占住成员的默认 stream——即使它只用 1-2 个 SM，排在其后的 B 计算也会被 stream 顺序卡住。因此：
- 每成员每微批一条 stream（共 2 条/成员）；B 的图在 stream_B 上 replay，与 A 的 AR 物理并行。
- CUDA Graph 每（形状 × 槽）各捕一份：prefill 形状已由 `LLAMA_PREFILL_KV_PAD=8192` 固定化（R18 资产），双微批下形状为 UB/2（如 512→2×256）或保持 512×2 槽——**图数 ×2，图缓存容量需从 4 提到 8**（`GGML_CUDA_GRAPH_VARIANTS_CAP` 已有 env，R16 遗产）。
- KV 写序：微批 A/B 的同层 KV 写必须按 ubatch 序——两 stream 间用 per-layer 事件链（A 的层 k KV 写完成 → B 的层 k 注意力才可发射）。这是 vLLM "attention metadata 独立"在我们因果模型下的等价物：**KV 是共享可变状态，是双微批唯一的序依赖**（residual 不跨 ubatch，prefill-parallel §5.2 已论证）。

### 3.3 伪码（调度循环）

```
# llama-context.cpp process_ubatch 的 prefill 路径（伪码级）
for ubatch u in prefill_queue:            # KV_PAD8192 固定形状
    slot = (u.index % 2)
    other = 1 - slot
    graph[slot].set_inputs(u)              # 写 slot 专属 host 快照 → async H2D
    event_input[slot].record(stream_h2d[slot])
    for layer k in 0..47:
        # 发布本微批层 k
        stream[slot].wait(event_layer_done[other][k])   # KV 因果序
        graph[slot].replay_layers(k)       # 该层段（含 AR）
        event_layer_done[slot][k].record(stream[slot])
    # 无需全局 fence：下一个同槽使用者等 event_input[slot] 的事件链即可
```

要点：层粒度事件（48×2 个/帧）而非整图 fence——这是"每段独立 fence"思想（prefill-parallel §5.2）与帧独立（vLLM）的合体。

### 3.4 改动点与工作量

| 文件 | 改动 | 估计 |
|---|---|---|
| `ggml/src/ggml-backend.cpp` | fifo 结构扩为 per-slot 快照组（去掉指针重绑路径）；split 事件按 (backend, copy) 语义改为 (backend, slot, layer) 可选细粒度 | ~300 行 |
| `ggml/src/ggml-cuda.cu` | 每上下文第二 stream；图缓存容量参数化 ×2 | ~80 行 |
| `src/llama-context.cpp` | 双图结果对象驻留（gf_res_prev 扩为 per-slot；注意 mtp-msgraph 撞过的 meta 2909 断言——本设计不跨槽复用图对象，各自独立 capture，避开该路径） | ~150 行 |
| `ggml/src/ggml-backend-meta.cpp` | 原则上零改动（meta 只见单成员单流）；验证岛内 P2P AR 的图内自旋在双 stream 下的行为（R17 资产：flag 自旋无 host 依赖，理论安全） | ~50 行验证代码 |

### 3.5 最小验证方案（合成微基准 → 整模型门）

1. **通信/计算重叠率微基准（单卡秒级）**：device 上两个 stream——stream_A 跑 3-member AR 的单卡模拟（自旋 kernel：flag 轮询 ~3ms），stream_B 连续跑 MoE grouped kernel（真实形状 256×10×640）。测 B 在 A 自旋窗口内完成的 kernel 数 / 总数 = 重叠率。预期 >70%（AR kernel 占 SM 少）。此基准直接验证 §1.1 的机理前提，**建议 MWE 之后立刻做（半天）**，作为 DBO 立项的 go/no-go 数据。
2. **双帧正确性 fixture**：复用 Codex 的跨 SYS 双目标层框架（`artifacts/rejected-stage-pipeline/` 内 harness），但改用双图双槽（不重绑）；4-copy 轮换 + reset/tail/drain 全过只是入场券。
3. **整模型门（Codex 三层教训）**：① 重复文本 8K/64K 512-parity；② **固定真实 8K 冷/热双跑逐 token 一致**（Codex 失败门的直接复刻）；③ 2×8K 多流 fair-share 回归。性能门：64K PP ≥ +10%。

### 3.6 风险清单

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 双图驻留触碰 meta 图缓存断言（2909 先例） | 不复用图对象、独立 capture；meta 层零改动设计 |
| R2 | 显存：双份 prefill 图 + 双份 compute arena（UB512 工作区已紧张，R20 OOM 先例 4824MiB） | 微批形状 UB/2（256）；arena 复用论证（两图不同时峰值——B 的层 k 在 A 的层 k+1 时运行，arena 若共享则冲突！需双 arena，实测预算后再定 UB） |
| R3 | 层粒度事件的 host 发射开销（48×2 事件/帧 × POWER9 弱单核） | 事件批量记录（每 N 层一 fence，N=4~8 自适应）；微基准先测发射开销 |
| R4 | 与 MTP prefill catch-up 交错（draft 图独立，理论无冲突，需 fixture） | 复用 R17 的 257/15360/65536 三长度 fixture 矩阵加双微批臂 |
| R5 | 收益被 MWE 吃掉后不足 +10% 门 | §1.4-3 的分流：转 C2' 形态（跨 ubatch 段间流水，收益上限 ~2×） |

## 4. 与 Codex 清单的分工建议

- **Codex 侧**：其清单第 5 项（"继续检查失败流水的缓冲及请求重置/缓存交互"）= 本文件 §2 的 C1-C6 排查。**建议 Codex 只做定位不做重实现**——坐实根因后把结论交本设计吸收（重绑路线已判死，修它没有意义；除非根因是 C5 类可独立修的交互 bug）。
- **本侧（ZCode 线）**：§3.5-1 的重叠率微基准（单卡秒级）+ 双图双槽设计实现。两者改动面不重叠（Codex 动 rejected 归档代码的复现，本侧动 fifo 结构）。
- **接口对齐**：若 Codex 坐实 C1（捕获地址），其结论直接决定本设计 §3.2 的"永不重绑"为硬规则并写入 TECH-FACTS；若坐实 C2（中间张量写冲突），本设计需追加 per-slot arena（R2 升级为必做）。
- **时序**：MWE（EPLB-DESIGN）落地 → 重测 nsys → 本设计 go/no-go → 若 go，与 Codex 的失败定位结论合流后开工。
