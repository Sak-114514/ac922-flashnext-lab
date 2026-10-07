# EPLB 热度专家放置 — 可行性论证与设计（2026-10-06）

> 结论先行：**原方案（SGLang 式热专家多副本放置）被判否决**——它瞄准的"专家按 rank 切分导致负载不均"在本 fork 中不存在（代码+实测双重推翻）。
> 实测的失步真根是**全注意力层的成员结构性空转 + 岛间串行**。本文件否决原方案后，给出替代方向
> **MWE（Member Workload Equalization，成员负载均衡化）**——同一显存/放置体系内的低风险改动，直击实测病根，判"值得立刻做（修正内容后）"。
> 树：`/home/debian/ac922-llama-next/sources/llama-v100-v3` @ ac922-flashnext-v100（134 脏文件=最新生产态）。本轮只读调研+已有 nsys 数据复挖，零 GPU 占用，零生产改动。

---

## 1. 论证（占全文 >30%，本节可独立阅读）

### 1.1 原方案前提的代码级推翻：专家不是按 id 切的

原假设（含 prefill-parallel/REPORT.md §1 的归因）："MoE 专家按 rank 切分 → 每层每 ubatch 负载天然不均 → AR 等最慢卡"。

代码事实（行号均为本轮亲核）：

| 证据 | 位置 | 内容 |
|---|---|---|
| 专家张量切轴 | `src/llama-model.cpp:625-640` | `pattern_ffn_up_weight`（regex `blk\.\d*\.ffn_up(_exps)?\.weight`）命中 `ffn_{up,gate}_exps` → `GGML_BACKEND_SPLIT_AXIS_1`（沿 n_ff 切）；`ffn_down_exps` → `AXIS_0`（沿 n_ff=输入维切，部分和进 AR） |
| 切分执行 | `src/llama-model.cpp:972-993`（scan 分配循环）+ `ggml/src/ggml-backend-meta.cpp:1717-1745`（init_tensor 按段建 simple tensor） | 每个岛成员持有**全部 512 个专家**的 n_ff 行切片；专家 id 维（ne2）从不参与切分 |
| 实际形状 | GGUF 头（target-header-preliminary.json） | `ffn_up_exps = {2560, 640, 512}`，down `{640, 2560, 512}`；n_ff=640 |
| 实际每卡份额 | 640 按 scan+粒度（Q2_0 blck=64 / IQ4_NL blck=32，`llama-model.cpp:865-880` FFN 分支 lcm(64,128)→64）三等分 | **{192, 224, 224}**（逐层轮转 rotation，`llama-model.cpp:515-520`）——最重/最轻 = 1.17×，静态不均 5% 量级 |

也就是说：SGLang EPLB 的前提是"专家-id 分区（EP）+ 热专家复制分散"——本 fork 是经典 TP 行/列切，**每卡看到的 (token, expert) 对集合完全相同**，MoE 负载按构造均衡，根本没有"专家放置"这个自由度可用。热画像（`/home/debian/strata-engine/expert-profile*.bin`，STRP 格式：magic+{version,48,512,slots,n_ranked}+uint16 (layer,expert) 对）在本 fork 的放置路径上没有消费点。

### 1.2 实测推翻：MoE 负载不均是 1.8%，不是 2-3ms

对已有 nsys 数据（`profiles/qsa-lc-64k/prefill-slice.sqlite`，64K 单流 prefill 12s 切片，6 卡逐 kernel 时间戳，本轮零 GPU 复挖）做逐 AR 窗口分析：

| 指标（岛内三成员，70 个窗口） | 数值 |
|---|---|
| `mul_mat_q_moe_grouped` 段内 busy 差（最重-最轻成员） | **p50 = 46µs**（段长中位 ~3.5ms → **1.3-1.8% 不均**） |
| 非 MoE（other）段内 busy 差 | **p50 = 3918µs** ← 真正的失步制造者 |
| 逻辑 AR 三成员到达错位 | p50 = 3185µs |
| AR 实例时长 | p50 = 3071µs；错位最低十分位时 ~350µs（线时间地板） |
| 错位与 AR 时长相关性 | **0.83**（AR 自旋 ≈ 到达错位） |

**MoE 专家负载不均理论被数据杀死**：46µs/窗口 vs 3918µs/窗口，差两个数量级。EPLB 即使做到完美，只能拿回 46µs/窗口。

### 1.3 失步真根（实测定性）：注意力结构不对称 + 岛间串行

同一 sqlite 的逐卡 kernel 直方图（12s 切片，6 卡）给出的结构性差异：

1. **全注意力层（12/48 层，full_attention_interval=4）KV 头=2、Q 头=24**（GGUF 头亲核）。3 成员岛按整 GQA 组切 → {0, 12, 12} 头 / {1,1,0} KV 头。实证：`flash_attn_ext_f16` 在 **dev3 = 0 个**、dev4 = 296、dev5 = 295——**dev3 在全部 12 个全注意力层上结构性零注意力工作**，只能提前到 AR 干等；`mul_mat_q` dev3=900 次 vs dev4/5=2090 次；`k_set_rows_quant`（KV 写）dev3=224 vs dev4/5≈820。位置：`src/llama-model.cpp:853-856`（QWEN4EXP qkv 粒度分支，`std::lcm(2*n_embd_q, blck_size_perf)`）。
2. **f32 AR 只在 B 岛出现且自旋极不均**：`AllReduce_Sum_f32_RING_LL` dev3=327ms / dev4=188ms / dev5=106ms（同 302 实例）——B 岛成员间存在系统性的小张量到达差。
3. **岛间串行**：两岛忙闲交替（0.6s 桶粒度可见完整波次），单岛活跃时另一岛空转；活跃桶内每 GPU compute ≈ AR-spin ≈ 150-200ms 各半——**每活跃 GPU 约一半时间在 AR 里等同伴**。

### 1.4 收益上界（诚实算）

口径声明：以上全部是 **64K 单流 prefill** 的 nsys（12s 切片）。多流/decode 形态未重测，AR 失步占比可能不同（decode 步时被 MTP/采样稀释，prefill-parallel 报告同口径）。

- AR 自旋 = 每成员 2.4-2.7s / 12s 切片 ≈ 墙钟的 **20-22%**；错位可归因部分 ≈ 85%（350µs×47 AR/ubatch 是线时间地板，不可消）。
- 修齐成员负载（§2 的 MWE）上界：prefill 墙钟 −15~20% → **64K PP 1003 → ~1180-1250**（与 smlayer 实测 1176 信号同量级，二者机理不同但天花板同源：都消 AR 等待）。
- 原 EPLB 方案上界：46µs/窗口 × ~47 窗口/ubatch ≈ **2ms/ubatch < 1%**——不值得任何工程投入。

### 1.5 成本与风险对比

| 方案 | 改动面 | 风险 | 失败先例 | 判决 |
|---|---|---|---|---|
| 原方案：热专家多副本（EPLB） | 加载器放置表 + MoE grouped kernel 多 slot 路由（同专家多副本查找）+ ids 重映射——**且需先把行切改成 EP 分区，否则无处放置副本** | 大：MoE kernel 是 H 线转正资产；EP 化等于换 MoE 通信范式（AR→all-to-all） | 无直接先例，但前提已被实测推翻 | **否决** |
| MWE：全注意力层成员均衡化 | 加载器切分回调 1 个分支（~40-80 行）+ 可选 KV 镜像 | 小-中：与岛 saga 的 KV/GQA 粒度结论（"整 GQA 组必须匹配、空成员合法"）同区域，须复用既有合法性框架 | 无（正是当年"空成员合法"修复的延伸方向） | **值得立刻做** |
| （对照）DBO 双微批 | 大（见 DBO-DESIGN.md） | Codex 失败在案 | stage pipeline token-37 分岔 | 排后 |

### 1.6 "不做也行吗"的回答

不做原 EPLB：损失 <1%（上界），无感。不做 MWE：64K prefill 每请求多付 ~15-20% 墙钟，且多流场景（2×64K）同样吃这个税（两岛各服务一路时，路内失步照旧）。MWE 是当前所有方向中 **改动/收益比最高** 的一项。

### 1.7 判决汇总

- **原 EPLB（热专家多副本）：否决**。理由：前提不存在（行切非 id 切，§1.1）+ 实测目标张量仅 1.8% 不均（§1.2）+ 工程量大且动 H 线转正资产。
- **替代 MWE（成员负载均衡化）：值得立刻做**，优先级 P0（四设计中最高）。
- **EPLB 机器（画像→放置表）挂起**，作为 DWDP 长期路线的第一步资产保留（§4.3）。

---

## 2. 替代设计：MWE 成员负载均衡化（论证通过后的施工图）

### 2.1 目标与不变量

目标：让 12 个全注意力层在 3 成员岛内每成员都有非零注意力工作，消除 dev3 类结构性空转；顺手排查 B 岛 f32 AR 的 3× 自旋差。

不变量（红线）：512-ID parity 逐位一致；KV/GQA 整组匹配原则（GPT-island saga 的最终结论）不得违反——空成员合法、但禁止拆散 KV 头与 Q 组的对应。

### 2.2 改动点（行号定位）

1. **Q 头填充**：`src/llama-model.cpp:825/854`（QWEN4EXP 的 q_weight/qkv_weight granularity 分支）。现状 `granularity_q = lcm(2*n_embd_q, blck_perf)`（n_embd_q = 12×256 = 3072 → 粒度 6144）导致 Q+gate 段 {0,0,6144}/实际等价 {0,3072,3072}。改为：Q 段按**单 Q 头粒度**（256×2=512，head_dim 整倍数、满足 blck 对齐）切 {8,8,8} 头；KV 段维持 {1,1,0} 不动（GQA 组匹配）。
   - 关键点：Q 头 h 的注意力需要其所属 KV 头的 K/V。成员持有 0 个 KV 头但持有 8 个 Q 头时，该成员必须能读到全部（或所属）KV——**方案：全注意力层 KV 镜像**（12 层 × 2 KV 头 × 256 dim × 2(K+V) × 2B(f16) × 64K ctx ≈ **每成员 +800MB** 满窗；实际 Q8 KV 再减半）。显存账：速度候选配置单卡最小空余 2281MiB → 剩 ~1.4GiB，可行但紧；262K 配置需重算（见风险 R3）。
2. **KV 镜像分支**：`llama_meta_device_get_split_state` 中对 `pattern_kv_cache` 命中的**全注意力层**返回 `AXIS_MIRRORED`（`src/llama-model.cpp:583-585` 一带）。`cache_k_l` 现为 AXIS_0（token 维切）；全注意力层改镜像后写放大 ×3 → 用现有 SET_ROWS 广播语义（meta 已处理镜像写，R19 的 KV/RS 按对象身份隔离逻辑覆盖）。
3. **f32 AR 排查（先诊断后改）**：302 个 f32 AR 实例的三成员自旋差是症状不是原因；疑点=某些小张量的 split-state 在 B 岛不对称（`GGML_META_DEBUG=1` 跑一次 8K 即可拿链路，秒级 GPU）。

### 2.3 伪码（切分回调的修改）

```
# llama-model.cpp, get_split_granularity(), QWEN4EXP 全注意力层分支
if layer is full_attention(il) and n_devices >= 3 and n_kv_head < n_devices:
    # Q 段：整头粒度填充全部成员（Q 头独立于 KV 头可任意归属，读镜像 KV）
    granularity_q  = 2 * head_dim            # 512，与 blck(128) 对齐自动满足
    # KV 段：镜像（见 2.2-2），不再参与 scan 分配
    segments_qkv   = {{n_q_total, 1}}        # K/V 权重与 cache 同改 MIRRORED
# else: 现状不动（GDN 层、IQ4_NL/MoE 段全部保持）
```

### 2.4 验证方案（含单卡微验证草案）

1. **零 GPU 静态验证**：写 40 行 python 重算 12 层的 per-member 工作量份额（用 hparams：24 Q 头、2 KV 头、640 n_ff、flash_attn FLOPs ∝ q头数×kvlen），对比现状 {0,12,12} vs MWE {8,8,8} 的每成员注意力+MoE 总量比。预期：最重/最轻成员从 ~1.9× 降到 ~1.08×。
2. **加载验证（分钟级 GPU）**：`LLAMA_CALLBACK_TRACE=400` 跑加载，grep `ffn_.*exps` 与 `attn_qkv` 的 `sums=[...]`，确认 {192,224,224}（MoE 不动）与全注意力层 Q 段 {4096,4096,4096}。
3. **A/B 门（复用生产矩阵）**：1×64K/512，parity 512-ID 逐位；门：64K PP ≥ +5%（对照门同 prefill-parallel 战役）。加分项：重抓 12s nsys，验证 AR p50 从 ~3.1ms 降到 <1ms、dev3 flash_attn 从 0 → ~296。
4. **守护**：2×8K 多流回归 + 温度 58°C 门（复用 scripts/zcode_probe.py 框架）。

### 2.5 风险清单

| # | 风险 | 缓解 |
|---|---|---|
| R1 | KV 镜像写放大在长上下文拖慢（每层 KV 写 ×3 成员） | KV 写带宽远小于注意力读；实测验证；仅 12/48 层受影响 |
| R2 | 显存：满窗 262K 配置每成员 +~800MB(Q8) 可能挤爆（速度候选余量 2281MiB） | MWE 加 `LLAMA_MWE_ATTN=1` env 门控，默认关；262K 配置单独预算 |
| R3 | 全注意力层 KV 镜像与 QSA/indexer 的 cache 视图（cache_idx_* 已 MIRRORED）交互 | cache_idx 本就镜像，无新交互；fixture 复用 R17 的 257/15360/65536 三长度矩阵 |
| R4 | 岛 saga 断言复活（split-state 一致性） | 不改 GQA 组匹配原则；空成员仍合法（只是不再出现于全注意力层） |

---

## 3. 与 Codex 清单的分工建议

Codex 清单第 2 项（"岛内 DWDP 式专家布局/预取是更大的独立工程"）与第 1 项（多流 MTP）之外，本文件的 MWE 是 Codex 清单未覆盖的新条目，建议：

- **本侧（ZCode 线）**：MWE 的切分回调改动 + A/B 验证（改动集中在 `src/llama-model.cpp` 单文件，与 Codex 的 runtime 侧（llama-context/ggml-backend）改动面不重叠，可并行开工）。
- **Codex 侧**：K8V4 组合验证（Codex 清单第 4 项）跑真实语料/多流时，**顺带收一路 MWE-on 的 64K nsys**（只加 env），用于确认 AR p50 降幅——零额外工程。
- **接口对齐点**：MWE 的 env 名（`LLAMA_MWE_ATTN`）与 preset 归档格式对齐现有 manifests 约定；MWE 落地后 Codex 的 DBO/层流水实验应在 MWE-on 基线上重跑（失步基底变了，重叠收益要重测）。

## 4. 附录

### 4.1 本轮实测数据存档

- 分析脚本：`/tmp/ar_skew.py`、`/tmp/ar_window2.py`、`/tmp/hist_dev.py`、`/tmp/final_check.py`（ac922，可直接复跑）
- 数据源：`/home/debian/ac922-llama-next/profiles/qsa-lc-64k/prefill-slice.sqlite`（2026-10-04 采集，非本轮重抓）

### 4.2 关键数字速查

- MoE 均衡：spread p50 46µs / 段 3.5ms = 1.3-1.8%；hi/lo = 1.018
- 到达错位 p50 3185µs；AR 地板 ~350µs；corr(错位, AR 时长) = 0.83
- 逐卡 AR 自旋：A 岛 ~2.64-2.68s / B 岛 ~2.37-2.39s（12s 切片）
- dev3：flash_attn 0 个（dev4/5 = 296/295）；f32 AR 327ms（dev4 188 / dev5 106）

### 4.3 EPLB 机器的挂起位（DWDP 演进关系）

热度放置（EPLB）与专家行分片（DWDP 变体 7）是同一演进的两步：**DWDP = 把行切改成专家-id 分片 + 按需取页；EPLB = DWDP 世界里再按热度放副本**。本 fork 当前行切布局下两者均无落点；若 DWDP 立项（调研报告优先级 4），第一步就是把 `expert-profile*.bin`（strata-engine 已有 STRP 读取器：`src/core/expert_cache.cpp:20-70`，格式 `STRP`+{1,48,512,slots,n_ranked}+uint16 对）接进加载器作为放置输入。届时本文件的否决结论自动失效，需按 EP 范式重新论证。
