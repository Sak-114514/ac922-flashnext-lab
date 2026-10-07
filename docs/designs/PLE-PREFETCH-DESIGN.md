# PLE 定向预取 + 有限热行缓存 — 可行性论证与设计（2026-10-06）

> 结论先行：**值得做，小工程低风险，P1**。冷启动 TTFT 的可寻址池 = 冷 68.05s − 热 12.08s ≈ 56s（随机 token 探针口径）；arm4 整段 dd 预热已零成本拿走 22s（→45.57s），本设计（按请求 token 定向预取）目标把剩余 ~33.5s 的大部收回，逼近 ~13-18s。R21 行缓存的热税教训（-29.4%）决定了**热行缓存降级为可选项**，主路径是"请求级批量预取 + page cache 当缓存"。
> 依据：本轮代码亲核（qwen4exp.cpp 的 PLE host 侧哈希、ple-row-cache.h）+ TTFT 战役全部实测（本地 ttft/RESULTS.md，2026-10-04）。零 GPU、零生产改动。

---

## 1. 论证（占全文 >30%）

### 1.1 冷启动税的结构（实测账本）

配置背景：模型 73.22GiB，PLE 表 `per_layer_token_embd.weight` 26.82GiB IQ4_NL（GGUF 文件偏移 43.81GiB），主机 RAM 31GB < 模型 → **每次 server 重启必然 PLE 全冷**（加载 46GiB GPU 张量自动挤掉页缓存，TTFT 战役关键发现 A）。PLE 表 mmap 惰性读（`TENSOR_READ_LAZY`，llama-model.cpp:199 一带）。

| 口径 | 冷 TTFT | 热 TTFT | 差 |
|---|---:|---:|---|
| 随机 token 探针（8052 个不同 ID，真实流量形态） | **68.05s** | 12.08s | **56.0s** |
| matrix 固定 prompt（重复文本，仅几十个不同 token） | 13.16s | 12.13s | ~1s（基准掩蔽效应，TTFT 战役关键发现 B） |
| arm1 R21 行缓存开 | 42.91s | **17.08s（热 PP -29.4%）** | — |
| arm4 启动后 dd 预热 PLE 段（63s 一次性） | **45.57s** | 12.30s（热无税） | 33.5s |

冷税的物理形态：每 token 读 `ple_n_heads=16` 行（(ngram−1)×heads_per_ngram = 2×8）× ~90B（IQ4_NL 160 维）→ 8K 随机 prompt ≈ 13 万次散点读，页缓存粒度 4KiB → ~537MB 随机页 IO，SATA SSD（BX500）随机 4K 低 QD 有效带宽 ~10-25MB/s → **数十秒级税，与 56s 观测吻合**。

### 1.2 剩余可寻址池与上界

以最优已验方案 arm4 为基线：45.57 − 12.08 ≈ **33.5s 剩余**（arm4 只是把整表 26.8GiB 顺序读一遍进 page cache——顺序 425MB/s；剩余税来自 arm4 后 26.8GiB>RAM 31GB×0.86 的挤出抖动 + 探针到达时部分页已被其他加载活动逐出 + prompt 特异行未命中）。定向预取只读请求实际需要的 ~537MB 页：
- 预取成本：排序去重后批量 pread，顺序化后 ~425MB/s → **~1.3s**（可与首 ubatch 计算重叠）。
- 收益上界：冷 TTFT 45.6 → **~13-15s**（逼近热地板 12.1s + tokenize/首步开销）；对裸基线 68 → ~13-15s（-78%）。
- 最新 Codex 真实语料口径（183.83 冷 / 949.21 热 PP，8K）：同机理（其冷请求 cache_n=0 全量 prefill），预取后冷 PP 预期 350-500（受其他冷因素——GPU 张量页、图缓存——限制，不会直接到 949）。

### 1.3 与已有方案的对照（为什么不是重复劳动）

| 已有资产 | 关系 |
|---|---|
| arm4 range-warm（dd 26.8GiB） | 保底方案：无请求时后台整段预热。缺点：63s 磁盘占用、与首批请求抢 IO、26.8GiB>可用页缓存必有余出、每 prompt 多读 50×数据。**与本设计互补**：空闲期 dd 保底 + 请求级定向补齐 |
| R21 行缓存（`ggml/src/ggml-cpu/ple-row-cache.h`，O_DIRECT+锁+哈希） | **不采用其形态**：热税 -29.4% 证明逐行锁+哈希在热路径亏；其"行→文件偏移"映射代码可复用 |
| RouterLookahead（strata-engine，decode 侧路由预取线程） | 思想源：本设计把"预取下一层路由命中"迁移为"预取本请求 PLE 行"（prefill 侧）。代码不共用（strata 是独立引擎） |
| `LLAMA_LOAD_PREFETCH`（加载期 WILLNEED） | 实测无关联（TTFT 战役 arm0p），不涉及 |

### 1.4 成本与风险

| 项 | 评估 |
|---|---|
| 改动面 | 小：新文件（ple 预取器 ~200 行）+ `llm_graph_input_ple::set_input` 挂点（qwen4exp.cpp）+ server 请求侧 hook（~30 行）。**不改任何计算路径**（纯预读） |
| 风险 | 低：最坏情况预取读错行（浪费 IO），计算路径读 page cache 照旧按需缺页——正确性零风险；parity 天然不变（预读不改数据） |
| 失败先例 | 无同类失败；R21 行缓存的热税是唯一先例，本设计主路径绕开（无热路径挂点） |
| 与 Codex 树冲突 | 无（Codex 清单第 3 项即本设计，其"按实际请求 token 预取、合并读取、绕开逐行锁、明确缓存预算"四要点与本节一致） |

### 1.5 "不做也行吗"

不做：冷请求 TTFT 停在 45.6s（arm4 已部署时）或 68s（裸）。损失的是真实流量的首请求体验（且这台机器每次重启必冷）。对比：MWE/DBO 是吞吐优化、MTP 多流是并发修复，本设计是**延迟修复**且是四项中改动最小、风险最低的。判决：**值得立刻做**（可与 MWE 并行，改动面零交集）。

---

## 2. 设计

### 2.1 行 id 的可得性（本轮代码亲核，设计成立的前提）

`src/models/qwen4exp.cpp:2566-2660`（`llm_graph_input_ple::set_input`）：PLE 行 id 在 **host 侧**由 n-gram 哈希计算——`mixed_n = (t[p]*m[0]) ^ ... ^ (t[p-n+1]*m[n-1]); row = mixed_n % vocab[h] + offset[h]`，前驱 token 从 KV cells（ext.tok）取。**因此：给定 prompt token 序列，全部所需行 id 可在请求到达时离线重算，无需等模型前向。** 这是定向预取的全部前提，已满足。

### 2.2 组件设计

```
ple_prefetcher（新，ggml/src/ggml-cpu/ 或 src/ 下）:
  # 状态：源信息（复用 ple-row-cache.h 的 source_info：/proc/self/maps 定位 mmap 文件+偏移）
  # API:
  prefetch_tokens(tokens[], n, ngram_ctx):     # 请求级入口（server 收到 tokenize 后调用）
    rows = {}                                   # ① 重算行 id：与 set_input 完全相同的哈希
    for t, window in ngram_windows(tokens):     #    （含 EOS 截断语义，照抄 :2630-2640）
        for h in 0..15: rows.add(hash(window) % vocab[h] + offset[h])
    pages = {row * 90B >> 12 的 4KiB 页号}      # ② 行→页（90B/行 IQ4_NL，页内多行自动合并）
    排序去重 → 批量提交
    # ③ 执行：n_threads=4 的 pread（O_DIRECT 可选）或 readahead() 系统调用
    #    readahead(fd, off, len) 优先——内核异步预读，不占用户缓冲，无需 io_uring
    后台线程执行，与 prefill 计算重叠；完成前不阻塞（预读是提示非承诺）

  # 每 ubatch 增量版（可选 W2）：set_input 挂点处对"下一 ubatch 的行"提前一个 ubatch 发起
  # （attention 元数据已知 token 序列，纯顺序扫描，~50µs 级 host 成本）
```

挂点与触发：
1. **请求级**（主路径）：server `process_prompt` 在 tokenize 完成后、入 prefill 队列前调用 `prefetch_tokens(全部 prompt token)`。8K prompt 的行 id 重算 ~16×8192 次整数哈希 ≈ <1ms（POWER9 单核）。
2. **ubatch 级**（W2 增强）：`llm_graph_input_ple::set_input` 内，为本 ubatch 行集发起 readahead（首 ubatch 会与请求级重复——去重靠页号集合；后续 ubatch 是增量）。

### 2.3 有限热行缓存（可选降级件，默认关）

仅当预取后仍有残余冷税（如重复服务同域 prompt 的场景）才启用：预算式 LRU（默认 **512MB**，env `LLAMA_PLE_HOT_CACHE_MB`），**分片无锁哈希**（16 分片独立 mutex 或 lock-free 开放寻址——直接回应 R21 的锁热点教训），只缓存"预取器读过的行"（预取路径写入，计算路径只读+命中统计），淘汰严格 LRU。热路径查询代价目标 <100ns/行（对照 R21 形态的 ~µs 级锁+哈希）。**首版可以不做**：page cache 本身就是缓存，预取把页带进来后热请求自动命中——自定义缓存的唯一增量价值是抗 page cache 挤出（26.8GiB 表 vs 31GB RAM 的竞争），用 arm4+预取组合的实测决定是否需要。

### 2.4 验证方案

1. **正确性**：parity 天然保持（预读无写入路径）；仍跑 1×8K matrix parity 门（仪式性）。
2. **效果**：复用 TTFT 战役 harness（本地 ttft/，服务器 scripts/launch_ttft_arm.sh + probe_cold.py 随机探针，seed 20261004）：新增两臂——arm5（定向预取 on，无 dd）、arm6（arm4 dd + 定向预取）。主指标：冷 TTFT（目标 arm5 ≤ 20s / arm6 ≤ 16s）、热 TTFT 无退化（12.1±0.5s）、热 PP 无退化（±1%）。
3. **防倒退**：matrix 固定 prompt 口径同步跑（arm4 曾在该口径 -1.8%）。
4. **压力**：连续 3 个不同随机 prompt（页缓存竞争形态），看第 2/3 个请求的 TTFT 稳定性。

### 2.5 预期收益算式（汇总）

```
冷税 = Σ(散点缺页) ≈ N_pages(4KiB) × T_random / 并行度
现状：~131K 行 → ~50-130K 页 × (4K 随机 ~80-250µs) ≈ 40-56s（观测 56s ✓）
预取：同页数但排序后顺序读 → 537MB / 425MB/s ≈ 1.3s，与计算重叠后有效延迟 ~0-1s
预期冷 TTFT = 热地板 12.08 + tokenize/首步 ~0.5-1 + 残余冷因素(GPU 张量页/图缓存) ~1-3
            ≈ 13.5-16s   （arm4 基线 45.6s → -65~70%；裸基线 68.0s → -76~80%）
真实语料口径：冷 PP 183.8 → 预期 350-500（其余冷因素非 PLE）
```

### 2.6 风险清单

| # | 风险 | 缓解 |
|---|---|---|
| R1 | 行 id 重算与 set_input 的哈希漂移（两处实现日后不同步） | 把哈希提炼为共享 inline 函数（qwen4exp.cpp 内 static，两处调用同一份）；W1 加 fixture：随机 8K prompt 的两路行 id 全等断言 |
| R2 | readahead 在 mmap 段上的语义（POSIX readahead 对 MAP_SHARED 文件有效，但内核可能限单次 2MB） | 分段循环调用（128KB×n）；备选 pread 到 /dev/null 式 dummy 读（更笨但确定） |
| R3 | 预取与首批请求抢 IO（arm4 dd 同时期运行时） | 预取器 IO 优先级 ionice idle 类；部署建议：二选一默认开预取、dd 保底仅空闲期 |
| R4 | decode 阶段的 PLE 缺页（生成 token 的新 n-gram 未预取） | W2 的 ubatch 级增量预取覆盖；decode 每 token 仅 16 行，缺页代价 ~2ms/token 可容忍（首版接受，实测后决定是否做 W2） |
| R5 | 多流请求的预取交错（两 prompt 的页集合竞争） | 页号集合按请求独立；readahead 本身无序，天然并发安全 |

## 3. 与 Codex 清单的分工建议

Codex 清单第 3 项即本设计（其表述："按实际请求的 token 预取 PLE 行（不是盲目整表）、合并读取、预取路径绕开逐行锁+哈希、缓存设明确预算"——与本文件 §2.2/2.3 逐条对应）。分工建议：

- **本侧（ZCode 线）**：§2.2 预取器 + 请求级挂点 + arm5/arm6 验证（改动面 src/models/qwen4exp.cpp 挂点 + 新文件，与 Codex 的 runtime 侧工作无冲突）。
- **Codex 侧**：若其 K8V4 组合验证（清单第 4 项）先行，请在其真实语料冷请求臂中预留 `LLAMA_PLE_PREFETCH=1` 的 env 位（实现未就绪时 no-op 即可），避免验证跑完后还要重跑冷场景。
- **接口对齐**：env 命名 `LLAMA_PLE_PREFETCH`（0/1，默认 0）与 `LLAMA_PLE_PREFETCH_THREADS`；arm5/arm6 结果按 TTFT 战役的 manifests 约定归档（`manifests/ple-prefetch-{5,6}/`）。
- **时序**：本设计无前置依赖，可立即与 MWE 并行开工；其验证（arm5/6）复用 TTFT harness，约半天 GPU 窗口。
