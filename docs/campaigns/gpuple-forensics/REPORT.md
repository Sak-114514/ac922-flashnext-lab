# R20 GPU-PLE 第 71 token 分岔 —— 根因取证报告（只读调查）

日期：2026-10-04。方法：纯只读——服务器 manifest/日志/工件取证 + 本地 ROUND20 数据再分析 + 源码阅读（`/home/debian/ac922-llama-next/sources/llama-v100-v3`）+ CPU 侧纸面推演。未改任何源码，未跑任何 GPU 实验。

---

## 0. 结论速览

1. **"R20 死于 stable top-k 修复前"假说被直接证伪**：R20 全部 7 个 manifest（GPU 裁决轮、CPU 对照轮、审计轮）的 env 里 **`GGML_CUDA_TOPK_STABLE=1` 全部在场**（且该开关是运行时 getenv，`top-k.cu:367`）。时间线也支持：R20 轮次 21:10–22:30，stable top-k 19:40–20:17 已完成。
2. **0.243 的真实含义被当年误读了一半**：`logprob-comparison.json` 显示 max 0.2425 发生在 **index 0（第一个生成 token）**——两边选了同一个 token（id 198 "\n"），top-1 logprob 差 0.0217，top-5 候选整体平移 0.02–0.24，公共前缀 mean-abs-diff **0.1265**。即：**8K prefill 一结束、第一次 forward 的 logits 就已经系统性差 ~1e-2 量级**；第 71 token 只是这个持续分布差首次翻转 top-1 的位置。这不是 tie-flip 噪声，是真实、确定的数值路径差。
3. **分岔是确定性的、模式内禀的**：GPU 模式跨进程逐位复现（21:35 进程 vs 22:00 进程前 128 token 全同），CPU 模式自复现（且 R21 换构建/换天 256 token 与 CPU B29 gold 全同）。两模式互差 → 排除 NCCL autotune / atomic 到达顺序等"运行间随机"族（R19 族）。
4. **PLE 链本身被证明精确**：六卡行分片 gather 的归约走 `exact_rows` F32 路径（`ggml-cuda.cu:1332`），1K 审计 393 万浮点逐位一致（md5 级证据）。但审计 callback 有硬限制：**按名去重、每张量只抓第一个 ne[1]≥128 的 ubatch = 只覆盖了 8K prefill 的第一个 256-token ubatch、三张 PLE 张量**；8192 token 里剩余 31 个 ubatch、PLE 层的 norm/gate/conv 输出、以及全部下游隐状态从未对拍。
5. **判决**：可解——一次便宜的 bisect 轮（复活已归档的诊断 patch 并扩展到逐层隐状态 + 末 ubatch）即可定位首个分岔张量；但**不值得**为生产复活 GPU PLE：其全部收益 = 冷 prefill（966 vs 288），而 dd 页缓存预热 + R21 行缓存（已建好、与 gold 全同）能以零 VRAM、零数值风险拿到暖速（~981 t/s）。"8K 专用模式"技术上可行（显存绰绰有余），但分岔恰好在 8K 出现，8K 专用并不能绕开 parity 问题，且收益被预热方案完全覆盖。

---

## 1. 证据链 A：R20 manifest env 取证（任务 1）

### 1.1 裁决轮与 manifest 对应关系（server.log PP/TG 实测对账）

| manifest | 实际结果 | 判定 |
|---|---|---|
| `r20-ple-six-20261002-2110`（B28/UB512，无 READY） | 加载后首轮 | 早轮 |
| `r20-ple-six-20261002-2120`（B28/UB512，有 READY） | `CUDA error: out of memory ... cuMemCreate` device 3，graph 建图期崩溃 | CURRENT-STATE:128 的 OOM 轮 |
| `r20-ple-six-20261002-2135`（**B29/UB256**+SHARED_DEVICE_COPY） | task 111: TG 72.15；task 199: PP **968.48**/TG 71.67；另有 PP **966.07** | **GPU 真实裁决轮（两波在同一进程）** |
| `r20-cpu-b29-20261002-2145`（PLE_SHARD=0） | PP 288.70 冷/955.11 暖，TG 57.57/75.22 | CPU 对照轮 |
| `r20-ple-check-20261002-2200` | PP 905.57 冷，128 token 输出与 21:35 进程逐位一致 | 跨进程复现性证据 |
| `r20-audit-cpu-20261002-2220` / `r20-audit-gpu-20261002-2230` | +`LLAMA_PLE_AUDIT_DIR`（诊断构建） | 1K 局部审计轮 |

同一二进制（inode 31789354，裁决三轮 mtime 相同）+ 同一 corpus（`assembled_text_sha256=4beb8354…`）+ 同 seed 1234 + temp 0.0 + 8192 prompt + 256/128 输出。两模式差异是**纯运行时 env 切换**。

### 1.2 env 取证原文（关键行，全部 7 个 manifest 一致）

```
GGML_CUDA_TOPK_STABLE=1          ← 所有 R20 轮次均启用（runtime getenv, top-k.cu:367 静态初始化一次）
GGML_META_COMPACT_ARENA=1 / GGML_META_PACKED_CACHE=1
GGML_META_ASYNC_COPY=1 / GGML_META_FIFO_PIPELINE=1
LLAMA_MTP_GPU_CHAIN=1 / LLAMA_MTP_GPU_EMBED=1 / LLAMA_MTP_FIFO=1
LLAMA_PREFILL_KV_PAD=8192, LLAMA_QSA_SPARSE_KERNEL=1, LLAMA_QSA_GATHER_MAX_TOKENS=32
GGML_CUDA_MOE_GROUPED=1, LLAMA_ISLAND_SPLIT=3,3, CUDA_GRAPHS=1
GPU 轮独有: AC922_TEST_PLE_SHARD=1(→LLAMA_PLE_SHARD=1) + AC922_TEST_PLE_ROWS=1(→LLAMA_PLE_SHARD_ROWS=1)
           + LLAMA_PLE_SEPARATE=1 + GGML_META_SHARED_DEVICE_COPY=1
CPU 轮独有: LLAMA_FULL_STEP_TIMING=1（仅计时）
```

（来源：`/home/debian/ac922-llama-next/manifests/r20-ple-six-20261002-{2110,2120,2135}/configuration.json`、`r20-cpu-b29-20261002-2145`、`r20-ple-check-20261002-2200`、`r20-audit-{cpu,gpu}-20261002-2{220,230}`；env→server 映射见 `scripts/run_test_server.sh:16-17`。）

**结论 A**：任务线索里的强假说"R20 上午测、stable top-k 下午才修"不成立——R20 真实裁决轮在 R19 stable top-k 合入之后跑的，且开关是开的。**R20 分岔不是 R19 的 atomic 到达顺序族**。旁证：若真有到达顺序随机，GPU 两波/两进程不可能逐位一致。

---

## 2. 证据链 B：分岔数据再分析（比当年记录更进一步）

数据源（本地 `final-results/ROUND20/`）：

- `logprob-comparison.json`：`first_token_difference=71`，`common_prefix_logprob_max_abs_difference=0.2425427`，`mean=0.126506`，且 max 行 `index=0, shared=5`。
- 四条 256-token 输出序列对比（`ple-real-{first,second}.json`、`cpu-b29-{first,second}.json`）：

| 对比 | 结果 |
|---|---|
| CPU first vs CPU second（同进程两波） | 逐位一致 |
| GPU first vs GPU second（同进程两波） | 逐位一致 |
| **GPU 21:35 进程 vs GPU 22:00 进程**（`ple-b29-probs.json` 前 128） | **逐位一致** |
| CPU vs GPU | 前 71 token 相同，**index 71 分岔**（CPU 多出 25903） |

### 2.1 量化解读（纸面推演）

- index 0：两边都选 id 198，但 logprob −0.19851(CPU) vs −0.17683(GPU) → top-1 logit 差 ≈ 0.022；同 rank 的 "\n\n" 差 0.196、" Be" 差 0.243。**整个 top-5 分布系统性平移**。
- ULP 级（~1e-7 相对）的内核累加差不可能在一次 prefill 里长成 1e-2 的持续 logit 偏移（无离散放大时）；而 BF16 量级（2^-8 ≈ 3.9e-3 相对）的**单次舍入事件**恰好落在观测幅度内。或者：一次离散选择翻转（QSA 块选择/MoE 路由）放大某个更小的固定差。
- 由于差是**确定性、跨进程稳定**的 → 必有一个**固定存在的结构性数值路径差**，只在 GPU-PLE 图里出现。
- QSA 路径 trace（两轮 server.log 各 3 条）显示两模式**选择完全相同的执行路径**（`dense=0, width=2051, width_pad=2304, token_limit=32, chosen=gather/masked` 完全一致）——不是路径分支不同，而是数值本身不同。

### 2.2 当年审计的真实覆盖面（关键新发现）

`ple-audit-diagnostic.patch`（归档于本地 ROUND20/）显示诊断 callback 的限制：

```cpp
const bool selected = name=="ple_lookup" || name.rfind("ple_key_raw-")==0 || name.rfind("ple_value_raw-")==0;
const bool wanted = selected && type==F32 && ne[1]>=128 && 尚未抓取过同名张量;   // ← 每名只抓一次
```

实测抓取（两轮 server.log）：

```
ple-audit name=ple_lookup      shape=160x4096x1   (= 16 heads × 256 tokens → 只是 prefill 第 1 个 ubatch)
ple-audit name=ple_value_raw-1 shape=2560x256x1
ple-audit name=ple_key_raw-1   shape=10240x256x1
```

md5（服务器 `artifacts/r20-ple-audit-{cpu,gpu}/`）：三对文件全部一致 → **第 1 个 ubatch 的 PLE gather+两个 lora_mm 逐位一致**。但：

- tokens 256–8191（其余 31 个 ubatch）的 PLE 张量**从未对拍**；
- PLE 层的 `grouped_norm / ple_gate / ple_conv_out`（norm、sigmoid 门、膨胀卷积）**从未对拍**；
- PLE 层之后所有层的隐状态、以及最终 logits **从未对拍**。

所以"PLE lookup/key/value 全逐位一致 ⇒ 差异在后续图路径"这句话的**证据强度只到第一个 ubatch**——调查当年止步于此，根因从未定位。

---

## 3. 证据链 C：GPU-PLE vs CPU-PLE 图路径差异清单（源码级）

模型：FlashNext/Qwen4Exp，PLE=n-gram 哈希 per-layer 嵌入，**仅 layer 1**（`ple_key_raw-1`）；表 26.82 GiB IQ4_NL（`per_layer_token_embd.weight`，160×~2.6M 行），CPU RAM 只有 node0、SATA SSD、64KiB base page。

两模式（同二进制、纯 env 切换）每个 ubatch 的图差异：

| # | CPU-PLE 模式（PLE_SHARD=0） | GPU-PLE 模式（SEPARATE=1） | 数值性质 |
|---|---|---|---|
| 1 | 表在 CPU；`ggml_get_rows` 走 CPU 后端（POWER9，GET_ROWS_THREADS 实际=1，R21 发现） | 表按行均匀分片到 6 卡专用 PLE Meta（`llama_ple_device_split`，axis=1 六段）；每成员带 `row_offset`+bounded 标记（`ggml-backend-meta.cpp:1771` 置 `op_params[1]=1`）做局部 gather，越界行写 0 | IQ4_NL 反量化整数查表×F32 scale，CPU/CUDA 逐位同（fixture 4 尺寸×12 变索引 + 首 ubatch 审计实证） |
| 2 | — | 六路 partial 经 NCCL allreduce 合并：`exact_rows = op==GET_ROWS && op_params[1]!=0` 强制 **F32**（`ggml-cuda.cu:1330-1345`，R20 修的 BF16 保真 bug）；且 x+0.0 精确 | 精确（设计+审计双重确认）。**注意**：该豁免只认 `op==GET_ROWS`——若某图形状下被 reduce 的边界节点是后继的 ADD 而非 GET_ROWS，且 ne≥262144（≥4 成员阈值），就会落回 BF16 压缩路径。审计形状证明裁决构建里 reduce 落在 GET_ROWS 上，但这是图分块的属性，不是被证明的全形状不变量 |
| 3 | — | 仅 GPU 模式存在 `ple_rows_zero`（cast(rows)×0 repeat）+ `ple_rows_join`（add）"镜像消费者"，被 `llama-context.cpp:3107` 强制钉在 PLE Meta 后端 | ±0 相加精确；作用是迫使六路归约发生在岛边界前 |
| 4 | emb（CPU）→ 拷贝上岛0 | emb（PLE Meta）→ 拷贝到岛0（`GGML_META_SHARED_DEVICE_COPY=1`：共享物理 GPU 直接 D2D） | 均为字节精确拷贝 |
| 5 | ple_key/ple_value 两个 lora_mm：src0 在岛0 → GPU 计算（审计 value_raw/key_raw 逐位同 ⇒ 两模式同后端同内核） | 同左 | 首 ubatch 实证一致 |
| 6 | 其后 norm/gate/conv/全模型/MoE/QSA：两模式同一批 GPU 内核、同形状 | 同左 | 理论逐位同 |
| 7 | 调度器后端集合 = {岛0, 岛1, CPU} | **多一个 6 卡 PLE Meta 后端**（`llama-context.cpp:398` 仅 DEFAULT ctx 注册，MTP 不注册）→ 整图分裂/分块不同（日志：FIFO frames 1048 vs 756、compact-arena 子图数不同） | 数据搬运与重分块本身不改数值，但改变了"哪些节点以何种 split 态执行"的搜索空间——**这是唯一还没被任何审计覆盖的差异面** |

关键排除项（有直接证据）：

- **非 R19 atomic top-k 族**（stable=1 在场 + 双模式跨进程逐位复现）；
- **非 MTP pending_h carry 族**（分岔出现在干净 8K prefill 后的第一个采样位，MTP 尚未介入；且 R19 的 reset 修复已在树上）；
- **非 prompt/seed/构建差异**（corpus SHA、seed、binary inode 全同）；
- **非 QSA 路径分支选择差异**（qsa-path trace 两模式逐条相同）。

---

## 4. 根因排序（带证据权重）

| 排序 | 假说 | 权重 | 支持证据 / 缺口 |
|---|---|---|---|
| **RC1** | **六卡 PLE Meta 特有的结构性数值路径差**（AR/交接/调度重排面里某处一个 BF16 量级舍入事件，或某算子换了执行变体），在 prefill 中后段产生 ~3e-3 相对 perturbation，随后以持续分布平移的形式存在 | **~45%** | ① 幅度匹配（0.02–0.24 logit ≈ 一次 BF16 级舍入的传播）；② 确定性+跨进程稳定 ⇒ 结构性；③ PLE 三张量审计精确 ⇒ 排除 gather 值本身；④ R15-F（GPU PLE 卷积+CPU 表）当年 512-ID parity=True ⇒ 破坏与"六卡分片表"架构强相关；⑤ 同类前科：NCCL 大 tensor 默认 BF16 已在本设计里咬过一次（fixture 期发现并修）。缺口：具体节点未定位——正是审计未覆盖的 97% 区域 |
| RC2 | 索引器 top-k / MoE 路由等**离散选择放大**了某个小得多的固定 ULP 差（如 emb 交接布局变化导致 MMQ tile 累加序差） | ~25% | QSA 稀疏选择全程活跃（dense=0、width=2051/132096），一次块选择翻转即可造成持续宽幅偏移；stable top-k 只保证"给定输入确定"，不保证对输入 ULP 鲁棒。缺口：仍需一个种子差（未定位），且要求索引分数近并列 |
| RC3 | PLE 链**后续 ubatch**（tokens 256–8191 的行集合）里六卡 gather 与 CPU gather 出现未审计的行级差异（fixture 行覆盖是合成的，8K 真实行集多 8 倍） | ~15% | 审计只到首 ubatch；机制上行级反量化应当与行内容无关（整数查表），故权重中等偏低 |
| RC4 | conv history / recurrent cache 在 GPU 模式下经不同 buffer-type/拷贝路径（ple-meta 注册改变 buft 比较序，C14/C15 曾在此区域出过 KV/RS 身份 bug） | ~10% | 历史前科区域；确定性、幅度均可吻合；无直接证据 |
| RC5 | 其它（loader 路径、CUDA graph capture 差异等） | ~5% | 拷贝类均字节精确；timing 类 env 无数值效应 |

**给"R20 是否 R19 同族"的最终回答**：不是同族根因。R19 是**同输入运行间不确定**（atomic 到达顺序），stable top-k 已修且 R20 已启用；R20 是**跨模式确定性数值差**（两个各自完全可复现的图，互不相同）。唯一的"同族"成分是：离散选择点（QSA top-k/MoE 路由）仍可能是把小差放大成 0.1 量级 logprob 差的**放大器**——但种子不是随机性，是结构性差异。

---

## 5. 可解性判决

### 5.1 能不能解决：能，且代价低（一次 bisect 轮）

分岔确定性 + 双方可复现 ⇒ 经典张量级二分可定位。方案（不跑也能写好的重测单）：

1. 复活已归档的 `final-results/ROUND20/ple-audit-diagnostic.patch`（LLAMA_PLE_AUDIT_DIR 机制本身没变，直接重放）并做三点扩展：
   - 抓取点扩到：`ple_conv_out-1`、`ple_gated_value-1`、layer 1 的层输出（`res`/attn out）、以及每层 `norm-*`（或隔层抽样）；去重策略从"每名一次"改为"每名×每 ubatch 前缀"（至少第 1、2、31、32 个 ubatch），外加最终 logits 行；
   - 两模式各跑一次同 8K prompt（132096 ctx、B29/UB256、其余 env 照抄 21:35/21:45 manifest）；
   - 逐张量 md5/逐位对拍 → **首个不一致张量直接点名根因**。
2. 结果分派：
   - 若首个不一致在 PLE 层内且发生在 ubatch>1 → RC3，修 gather（或换行分片边界验证）；
   - 若 PLE 层全同而 layer 2 输入已差 → 差在交接/调度面（RC1）：dump ple_meta 边界被 AR 包裹的节点 op 与 ne，核对是否落入 BF16 阈值（`n_backends>=4 && ne>=262144` 且无 `exact_rows` 豁免）——若是，修法一行：把豁免扩展到 `ple_rows_join`/或对该边界强制 F32；
   - 若逐层二分到某个 GPU 算子在两模式跑在不同后端/内核变体 → 钉住 backend（复用 `graph_get_cb` 的强制机制）。
3. 预计：诊断构建 ~10 分钟编译 + 2×(8K prefill ~9s/12s) + 对拍脚本，**一轮 A/B 即可判**。修完后的验收 = 同 8K prompt 双模式 256 token 逐位一致（且应同时回归 R19 的重复性 fixture）。

### 5.2 值不值得：默认不值得（收益已被更便宜的方案拿走）

GPU PLE 的实测收益边界（ROUND20 数据）：

| 场景 | CPU-PLE | GPU-PLE | 差 |
|---|---|---|---|
| 冷 prefill（页缓存冷） | 288.70 | 966.07/968.48 | **+3.3×（全部收益在此）** |
| 暖 prefill | 955.11（R21 无缓存暖 981.3） | 966–968 | ≈+1%（噪声级） |
| TG | 75.22 暖 | 72.15/71.67 | 持平略低 |
| 成本 | CPU RAM 26.82GiB、SATA 冷读 | **+4.47GiB/卡 VRAM**、132K 仅 B29/UB256 勉强放下、H+E 组合 graph-reserve OOM | |
| 数值 | 与 CPU gold 恒等（整个回归方法学的锚） | 与 gold 第 71 token 起分岔（未解） | |

替代方案已经在手且零数值风险：

- **R21 行缓存**（已建成、fixture 全值通过、输出与 CPU gold 全同）：冷 241.95 → 360.02（+49%）；
- **dd 页缓存预热**（任务书所述零成本方案）：服务启动时把 26.82GiB 表从 SATA 顺序读一遍（SATA ~400-500MB/s ≈ 1 分钟量级、一次性），冷启动即达暖速 ~955–981 t/s，彻底消灭 GPU PLE 的唯一收益场景；
- 叠加后 CPU-PLE ≈ GPU-PLE 的全部性能、无 VRAM 代价、无 parity 破坏、无 132K OOM 风险。

**判决**：除非部署上存在"不允许预热 + 冷启动 TTFT 是硬 SLA + RAM 压力大到页缓存保不住 27GiB"三者同时成立，否则不值得再投入。若团队出于完备性想关掉这个悬案，做 5.1 的一轮 bisect（约一小时级）即可——它同时是"修不修"的决策依据：若真是一行 F32 豁免类小修，保留一个 opt-in 分支的成本也低。

### 5.3 "8K 专用模式"（GPU PLE 只在 8K 开，避开 132K OOM）可行性

- **显存**：可行且宽裕。132K 的 KV/indexer/recurrent 缓存是显存大头；ctx 降到 8192 后每卡释放数 GiB，B29/UB256 的紧张边距（184–192MiB 级）消失，甚至 H+E（FIFO+全图，曾因 981.40MiB graph reserve OOM）也有机会在 8K 放下。
- **数值**：不可行——**分岔恰好在 8K prompt 上出现**，与 ctx 上限无关；8K 专用只是换掉 OOM 问题，换不掉 parity 问题。当前生产方法学以"与 CPU gold 逐 token 一致"为准入门（R14–R21 全部如此），GPU-PLE 模式过不了这道门，除非改立"GPU-PLE 自有 gold"（它确实跨进程逐位稳定，可以立）——但那是方法学让步，不是修复。
- **收益**：8K 场景下 GPU PLE 对"预热后的 CPU-PLE"的暖速优势 ≈ 1%（966-968 vs 955-981），冷启动优势被 dd 预热归零。
- **结论**：8K 专用模式**技术上可行、战略上被 (dd 预热 + R21 行缓存) 支配**。仅当 5.1 的 bisect 落地为小修、且冷启动 SLA 真实存在时，才值得作为 opt-in 保留。

---

## 6. 附录：证据文件索引

服务器（ac922，只读）：
- manifests：`/home/debian/ac922-llama-next/manifests/r20-ple-six-20261002-{2110,2120,2135}/`、`r20-cpu-b29-20261002-2145/`、`r20-ple-check-20261002-2200/`、`r20-audit-{cpu,gpu}-20261002-2{220,230}/`（configuration.json + server.log）
- 审计工件 md5：`/home/debian/ac922-llama-next/artifacts/r20-ple-audit-{cpu,gpu}/`（三对 .bin 全一致）
- 源码：`sources/llama-v100-v3/` — `src/models/qwen4exp.cpp:2714-2870`（build_inp_ple/build_ple）、`src/llama-context.cpp:3100-3121`（ple_rows 强制后端）、`src/llama-model.cpp:228-246,885-925`（PLE split）、`ggml/src/ggml-backend-meta.cpp:883-891,1771-1787`（GET_ROWS partial/bounded）、`ggml/src/ggml-cuda/ggml-cuda.cu:1307-1378`（NCCL F32/BF16 阈值 + exact_rows）、`ggml/src/ggml-cuda/top-k.cu:367`（stable 开关）

本地：
- `final-results/ROUND20/logprob-comparison.json`（0.243@index0 证据）
- `final-results/ROUND20/ple-tensor-comparison.json` + `ple-audit-diagnostic.patch`（审计范围限制证据）
- `final-results/ROUND20/{ple-real-first,ple-real-second,cpu-b29-first,cpu-b29-second,ple-b29-probs,cpu-b29-probs}.json`（四序列+跨进程对拍）
- `final-results/ROUND20/{ple-b29,cpu-b29}-server.log`（qsa-path trace 对比）
