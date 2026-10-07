# 多并发叠加实验完整方案（PLAN v1，2026-10-04）

> 定位：**纯准备**。F/G 两路 GPU 战役结束后按本方案直接执行。本文所有预期数字均标注推导来源：
> **[M]** 机器实测锚点（可直接引用）／**[E]** 实测外推（标注外推链）／**[G]** 纯估算（无实测背书）。
> 脚本：`concurrency/multi-concurrency-bench.sh`（本目录，部署到 ac922 `scripts/`）。

---

## 1. 资产盘点

### 1.1 两份关键 manifest 的完整 diff

`island-hb-multi`（B+H 多路，C18/C19 落盘，cases=2:8192，409600 ctx 已启动通过、minfree 1459 MiB）与
`island-h-fifo-single`（当前生产 preset，1×8K PP 966.51/TG 102.31，FIFO 组合 opt-in）的 env 差异全表：

| 键 | island-hb-multi | island-h-fifo-single | 语义 |
|---|---|---|---|
| AC922_TEST_CTX | 409600 | 132096 | 统一 KV 池总量（--ctx-size） |
| AC922_TEST_SLOTS | 4 | 1 | --parallel，即 slot 数 |
| AC922_TEST_SLOT_CTX | 132096 | （缺省=无 cap） | --kv-unified-per-slot，每 slot 上限 |
| AC922_TEST_UBATCH | 256 | 512 | 多路验过的 UB=256；单路生产 UB=512 |
| LLAMA_QSA_POOLED_MULTI | 1 | （缺省=0） | **B 开关**：QSA 多路池化快路径 |
| LLAMA_QSA_MULTI_MIN_CTX | 32768 | （缺省=32768） | B 的聚合长度门（累计活跃上下文） |
| LLAMA_PREFILL_FAIR_SHARE | 1 | （缺省） | 多路 prefill 配额公平分享 |
| GGML_META_ASYNC_COPY | （缺省=0） | 1 | FIFO 组合部件 1：异步跨岛拷贝 |
| GGML_META_FIFO_PIPELINE | （缺省=0） | 1 | FIFO 组合部件 2：输入快照流水 |
| LLAMA_PREFILL_KV_PAD | （缺省=0） | 8192 | FIFO 组合部件 3：定长 prefill KV 视图 |
| AC922_TEST_P2P_AR | （缺省=0） | 0 | P2P 整图归约（R17 验证，收益 ~1-2%） |

其余 30+ 键完全一致（MTP4/PAD6/MTP_UBATCH32、graphs、HC_FUSED、NUMA0、NCCL_AUTO、H=MOE_GROUPED、
island 3,3、DETP=0、ONECAT/MONOMOE=0 等）。**结论：两份 preset 本质是同一底座 + "多路三件套"
（SLOTS/SLOT_CTX/FAIR_SHARE + B）与"FIFO 三件套"（ASYNC_COPY/FIFO_PIPELINE/KV_PAD）的正交叠加，从未合体过。**

### 1.2 B+H 的全部实测记录（TECH-FACTS/GPT-HANDOVER 汇总）

| 记录 | 数字 | 证据 |
|---|---|---|
| H 单独 `GGML_CUDA_MOE_GROUPED=1` | 1×8K PP 532.6→586.2（+10.1%）；2×1K PP +118%；64K 多路 PP 153→175.2（+14%）；decode matvec 沾光 | [M] h-moe-* / bh-2x64k |
| B 单独 `LLAMA_QSA_POOLED_MULTI=1` | 2×64K TG 13.2/17.5→25.5/27.9（+93%/+59%） | [M] b3-2x64k |
| **B+H 合体 2×64K** | **decode 38.7/39.3，聚合 30.7→77.9（+154%），parity=True** | [M] bh-2x64k（六路 TP，无岛） |
| B 的 8K 劣势 | 2×8K B 开 20.9/21.3 vs 不开 34.2/35.4（六路口径），**未解** | [M] B round4/5 |
| round-5 分析 | cell-sel 修复被机器推翻（parity=False + TG 21.45/16.88）；拟合负斜度证明非纯延迟，剩余假设=接受率混淆；B round-4 二进制的 8K parity 从未测过 | [M] 裁决轮2 + [A] |
| F16_GATHER 附带开关 | `LLAMA_QSA_F16_GATHER` 默认**开**（qwen4exp.cpp:2217），随 B round-4 进了 bh-2x64k 二进制，从未单独归因；=0 可回退 F32 gather 做 A/B | [C] 源码 + TECH-FACTS 疑点2 |
| 岛 C+H 2×8K（B 被门挡） | PP 299.72/299.71，TG 45.74/46.39，聚合 92.13（vs 六路 69.64，+32.3%），两路 512-ID parity True | [M] C18 |
| 岛 C+H 单路 1×8K | 749.79/100.04（vs 六路基线 +40.8%/+26.9%） | [M] C17 |
| **长 C+B（岛×B 在 64K 激活）** | **从未验证**（island-hb-multi 只跑了 2×8K，B 被长度门 demote） | 缺口 |
| FIFO 组合 | 单路 1×8K PP +28.90%/TG +2.27% 过门保留；**多路未验**（R18 明确 scope 仅单路） | [M] R18-E |

### 1.3 fork server 并发机制（代码级，本地镜像逐行核对）

- **--parallel/--slots**：`server.cpp:156-172` `n_parallel<0` 自动=4 且 kv_unified；`server-context.cpp:1346`
  每 slot n_ctx=`n_ctx_slot()`。`--kv-unified` 下 `llama-context.cpp:336-337` **n_ctx_seq = 整池**，
  per-slot 上限只由 `--kv-unified-per-slot` 绑定 → `409600 池 + cap 132096` 的真实语义是
  "4 slot 共享 409600 池、每 slot 最多用到 132096"（GPT-ISLAND RESULTS.md "四路不能各自同时占满"即此）。
- **KV 预算实测锚点**（MEMORY-AUDIT.md，直接启动日志读数）：
  - 六路、池 266240、4 slot、MTP3：per-device 缓存保留 = KV 552.50 + indexer 414.38 + draft KV 138.13 +
    draft idx 34.53 = **1139.54 MiB**；telemetry 最低空余 **3120 MiB**。
  - 岛、池 409600、4 slot、MTP4：启动通过，minfree **1459 MiB**（GPT-ISLAND）。
  - 岛、池 132096、1 slot：minfree **4009 MiB**。
  - 线性斜率（两点拟合）：KV ≈ **2.08 MiB / 1K 池 token / 每卡**，indexer ≈ **1.56 MiB / 1K / 每卡**，
    recurrent 状态 ≈ 每 seq 450 MiB 逻辑（n_seq_max 线性）。
  - **权重侧**：GPU 实载 ~46.4 GiB（73.2 GiB GGUF − 26.82 GiB CPU 常驻 PLE 表），岛模式 A/B 岛 25.89+20.24 GiB
    （每卡 8.6/6.8 GiB）。CPU PLE 锁死 node0 31.5 GiB 中的 26.8 GiB → **CPU RAM 没有余量再放第二实例**。
- **cache_prompt / cache_reuse 路径**（`server-context.cpp:3436-3508`）：
  - cache_prompt=true 走公共前缀复用；**hybrid/recurrent 记忆下若 pos_min 无法证明一致，强制全量重算**
    （:3598 "forcing full prompt re-processing ... hybrid/recurrent memory"）——`--ctx-checkpoints 0`
    （当前启动器写死）时没有 checkpoint 可恢复。**纯追加式 agent 轮次（prompt 只增长）是安全的**
    ：n_past=全前缀，pos_min < 阈值，不触发 reset，recurrent tail 保留，KV 增量追加。
  - **--cache-reuse N（KV 块搬移）对本模型有三重隐患**：① `llama_memory_recurrent::seq_rm` 中段删除仅支持
    tail 回滚 ≤ n_rs_seq（MTP4 时 =draft.n_max=4，`common/common.h:396 need_n_rs_seq()`），超界返回 false；
    ② `server-context.cpp:3488` **忽略 seq_rm 返回值**继续 seq_add；③ `seq_add` 对 recurrent 只挪 tail pos、
    **不作废状态**（hybrid-idx 只失效 QSA 块缓存）。can_shift 本身返回 true（hybrid→attn，非 Step35），
    所以 server 不会替我们关掉它。**结论：主矩阵不开 n_cache_reuse；cache_prompt=true 的纯前缀复用（agent
    天然形态）才是场景 A 的正收益来源。**
- **多路专用机制**：
  - `LLAMA_PREFILL_FAIR_SHARE=1`（server-context.cpp:1104）：per-compatible-cohort 配额，防止一路独占 batch。
  - `LLAMA_SPEC_PAD_MULTI=N`（server-context.cpp:80-107）：多路 MTP verify 补齐到固定形状保 CUDA graph 复用。
    **代码级坏消息**：补齐上限受回滚预算约束 `n_pad ≤ n_rs_seq − (n_rows_seq−1)`（:3960 附近），MTP4 满深度
    verify 时 n_rows_seq=5 → n_pad ≤ 4−4 = 0，**满深度下永远补不动**（深度降为 3 时才有 1 行余量）。
    → 不指望它稳形状，改用 `GGML_CUDA_GRAPH_VARIANTS_CAP=8`（fork 已有 env，TECH-FACTS §四）容纳
    6 slot 的 {5,10,15,20,25,30} 行形状族。
  - benchmark 基建：`benchmark_server.py` 按 flow 显式 pin `id_slot`（:396-406 从 /slots 取），
    `matrix_runner.py` 已带 58C 守护、telemetry、parity 比较（512 output_token_ids 逐位，:99-107）。
- **B 的长度门语义**（hybrid-idx.cpp:726-745）：按 **ubatch 活跃序列累计** n_tok 计，默认 32768。
  推论：2×8K（16384）被 demote；**4×8K（32768+）与 6×8K（49152）会自动激活 B**——B 的 8K 劣势是在
  聚合 16384 下测的，聚合 49K 时 B 快路径成本模型（O(sum blocks) vs O(n_kv)）不同，不能直接套用劣化结论。

### 1.4 上游 b11374 "parallel sequences" 修复核查（gh api 亲验）

- b11374 release = **唯一一个 commit `b92761a515`**（PR #29852，2026-10-03）。"fix parallel sequences" 与
  "direct gdn outputs (break parallel sequence)" 是该 commit 里 **ggml-openvino 后端子目录**（Intel Yu Zijun，
  qwen35/OpenVINO）的子提交，**不触碰 CUDA/generic 图路径**。我们 V100 CUDA 管线与之零交集，
  **无需也无法 cherry-pick**；"我们缺 GDN 并行修复" 的担忧对 CUDA 路径不成立。
- 我们自己的并行序列正确性证据：2×8K（C18）与 2×64K（bh-2x64k）**两路各 512-ID 逐位 parity=True**。
  残余暴露面=未测组合（岛×B 激活×64K、FIFO×多路、6 slot），由本矩阵的 parity 门覆盖。
- 基线（#29333, 2026-09-28）之后真正值得看的邻居 commit（已逐个核 file list）：

| commit | 内容 | 冲突面 | 建议 |
|---|---|---|---|
| `889edf43dd` | qwen4exp indexer score 内存减半：pool_mask F32→F16 + lightning_indexer 融合核（n_head=4 vec kernel） | lightning-indexer.cu +20/−1（fork 有此文件，hunk 大概率干净）；**qwen4exp.cpp +7/−16 落在 fork B 路径重写的 build_qsa_sel 区域 → 高冲突，需手工移植**；metal/vulkan 与我们无关 | 战役后可选；对 C 场景是显存利好、速度利好小（S1: QSA 专用核仅占 2.3-2.6%）。不做 git cherry-pick，人工移植图改动并入 B 路径 |
| `bed0a85660` | CUDA: 共享专家融进 MMVQ（mmvq.cu/common.cuh/ggml-cuda.cu +136） | fork 的 MoE matvec 家族（mmq-moe-grouped、monomoe、mul_mat_vec_iq2_s_moe_volta）重度改动同区域 → 中高冲突；且需先确认本模型 GGUF 是否有独立 shared experts | 搁置，并发战役不做 |
| `134b2bb756` | ggml-cuda: cpy 转置路径污染非连续 dst 的正确性修复（cpy.cu **+2/−1**） | 极小；fork 的 cpy.cu 有岛断言补丁但 hunk 很可能干净 | 值得顺手带走（下次统一编译窗口），带 parity 门 |
| `5fdfa62829` | GDN 归一化 max→rsqrt | 已在 fork 内（models.h helper `1.0f/sqrtf(n)` 亲验） | 无需动作 |
| `1ab7e5ad2d` | RMS_NORM+SCALE 融合（2026-09-25，早于基线） | fork 已有该融合且有 GDN 变体（ggml-cuda.cu:4960/5829 亲验） | 无需动作 |
| `1fb7e5ad2d` | spec 概率采样（server-context +57 等） | server-context.cpp 重改区域，中冲突 | 与 greedy 基准无关，不做 |

---

## 2. 叠加实验矩阵（不用控制变量，直接全开；唯二保留的两臂对比是"已知劣化未解"的 B@8K 与 FIFO×多路未验）

通用口径：MTP4/PAD6/MTP_UBATCH32、CUDA graphs、HC_FUSED、H=MOE_GROUPED、NUMA0、NCCL autotune、
PLE=CPU（PLE_SHARD=0）、DETP=0、ONECAT/MONOMOE=0、P2P_AR=0、fair_share（多路）、512 输出/流、
greedy temp=0 seed=1234、逐位 parity 门。启动配置全文见脚本 `scenario_env()`。

| 场景 | 配置（增量语义） | cases | 预期聚合 decode tok/s（口径=sum of per-flow server TG） | 验收门 |
|---|---|---|---|---|
| **A1** 锚点复现 | 六路 TP + B+H（MIN_CTX=32768），slots 2，池 266240/cap 132096，UB 512 | 2:65536 | **≈78**（38.7/39.3）**[M]** bh-2x64k；PP/流 ≈175 **[M]** | ≥70 且 parity（无旧参考则查 512-ID 计数完整）；此为当代二进制是否仍交付锚点的裁判 |
| **A2** 长上下文主线 | 岛 3,3 + H + B（门默认，64K 下必然激活）= **长 C+B 首验**，slots 2，池 266240/cap 132096，UB 512 | 2:65536 | decode 78–103 **[E]**：78 锚点 × 岛多路 decode 因子 1.32（92.13/69.64，2×8K 实测）乘法外推，带宽宽因岛×B 从未合测；PP/流 200–260 **[E]**（153×1.14×1.32≈230） | parity=A1 产物做参考目录；decode ≥78（不达标→生产双 agent 配方回退 A1） |
| **A3** 全开+流水 | A2 + FIFO 三件套（ASYNC_COPY+FIFO_PIPELINE+KV_PAD8192） | 2:65536 | =A2 decode ±2% **[E]**（R18 单路 TG +2.27% [M]）；PP/流再 +10~29% **[E]**（R18 单路 PP +28.9% [M] 的迁移假设） | parity 参考同 A2；PP 增益若 <5% 且 parity 挂 → 回退 A2 |
| **A3-agent** 增量轮 | A3 server 存活，2 流 × 4 轮 cache_prompt=true（历史只增长） | 每轮 n_predict 256 | 首轮 prefill ≈ 65536/PP ≈ 210–330s **[E]**；**第 2 轮起 prompt_ms 应降为 O(1–3s)**（只算 delta ≈ 256+512 token）**[G]**，tokens_cached≈全历史；decode 不变 | 第 2 轮起 prompt_ms < 5s 且 cache_n > 0.9×prompt；若触发全量重算（pos_min reset 日志）→ 记录并按 §1.3 结论处理（预期纯追加不触发） |
| **B1** 短上下文主线 | 岛 C+H，**B 显式关**（POOLED_MULTI=0），slots 4，池 266240/cap 132096，UB 256，fair_share | 2:8192, 4:8192 | 2×8K 聚合 **92.1**（45.74/46.39）**[M]** C18；PP/流 ≈300 **[M]**；4×8K ≈ **83** **[E]**（92.13 × 68.08/75.29：final-mtp-multi-fair 的 4→2 slot 聚合衰减比，跨配置外推） | 2×8K ≥88（锚点 −5% 容差）；4×8K parity 完整 |
| **B2** B@8K 快速两臂 | B1 + B 强制开（MIN_CTX=0） | 2:8192 | 未知，风险带 42–95 **[E]**：下界=六路 B@8K 20.9+21.3=42 **[M]**（不同拓扑），上界=B1 的 92；round-5 接受率混淆假设仍在 | **判定门而非性能门**：若 B2 ≥ B1+10% 且 parity=True → 短上下文配方改带 B 并重跑 B3 变体；否则 B 在 8K 永久关闭，此臂即结案证据 |
| **B3** 全开短上下文 | B1 + FIFO 三件套 | 2:8192, 4:8192 | decode = B1 ±2% **[E]**；PP/流 +10~30% **[E]**（R18 模式迁移） | parity 参考=B1；不达标回退 B1 |
| **C2** 极限参考 | 六路 + H + B（**6×8K 聚合 49152 ≥ 32768 → 门自动放行 B**），slots 6，池 98304/cap 16384，UB 256，VARIANTS_CAP=8，HTTP 12 线程 | 4:8192, 6:8192 | 4×8K ≈ 68–75 **[E]**（六路 MTP3 实测 68.1 **[M]** 为底 + MTP4/UB256 差异带宽）；6×8K ≈ 70–90 **[G]**（聚合随 slot 数平走缓升的通常形态，无本机实测） | 无硬数字门：不 OOM 启动 + parity 完整 + 记录每流 TG |
| **C1** 极限主线 | 岛 + H + B（同上自动放行），slots 6，池 98304/cap 16384，UB 256，VARIANTS_CAP=8 | 4:8192, 6:8192 | 4×8K ≈ 83–95；6×8K ≈ 85–100 **[E]**（C2 带宽 × 岛因子 1.32，低置信，两跳外推） | parity 参考=C2 产物；6×8K 聚合 ≥75 即视为并发扩展成功（[E] 下界）；OOM → SLOT_CTX 降 12288 或 slots 5 重试 |

KV/显存预算核对（C 场景，逐卡 16 GiB）：
权重（岛口径）≈ 8.6 GiB（A 岛卡）／6.8 GiB（B 岛卡）→ 池 98304 的缓存保留 ≈ KV 204 + idx 153 +
recurrent 6 状态 ≈ 450 + compute ≈ 500 → 预测 minfree ≈ **3+ GiB** **[E]**（对照实测锚点：池 266240→3120，
池 409600→1459，池更小只会更宽）。A/B 场景池 266240 < 已启动过的 409600 → 预测 minfree ≈ **2.0 GiB** [E]。
CPU 侧：26.82 GiB PLE 常驻不变，6 slot 增量全在 GPU/池内，CPU RAM 无新增压力。

---

## 3. 执行脚本（concurrency/multi-concurrency-bench.sh）

- 部署：`scp concurrency/multi-concurrency-bench.sh ac922:~/ac922-llama-next/scripts/`。
- 骨架完全复用已验基建：`systemd-run --unit=ac922-flashnext-test` + `run_test_server.sh`（AC922_TEST_* env
  注入，本仓库根目录一份即远端 scripts/ 那份）→ `matrix_runner.py`（健康等待/58C 守护/parity/telemetry）
  → `benchmark_server.py`（flow prompt 唯一性 + id_slot pin + 512 输出校验）。
- 每场景全新 label `conc-<SC>-<时间戳>` + 全新日志（防追加污染，运维红线 #3）；结果落
  `~/ac922-llama-next/concurrency/<label>/`，汇总追加 `RESULTS.tsv`（per-flow PP/TG、聚合 sum、wave 墙钟）。
- parity 参考链（脚本用 REF_A/REF_B/REF_C 环境变量注入，指向先跑场景的产物目录）：
  `A2/A3 → A1`；`B2/B3 → B1`；`C1 → C2`。同 case 同 flow prompt 确定性生成，跨配置逐位可比（bh-2x64k
  正是这样做的）。
- `--agent-turns N`：最后一个场景波次后保活 server，直接接 cache_prompt=true 的 2 流多轮增量测量
  （复用 benchmark_server 的 tokenize/build_flow_prompts 保证 prompt 同源），记录每轮 prompt_ms、cache_n、
  delta_prefill_tok、decode_tps。
- 预检：停 vLLM 两个自启服务 + GPU 显存 <600 MiB + 6 卡在线（红线 #1）；全程不编译不改码。
- 建议执行序列（依赖关系即参考链，总预算 ≈ 2.5–3.5 h 含 8 次载模 ~6 min/次）：
  1. `B1`（~15 min）→ 2. `B2`（~10）→ 3. `B3`（~15）→ 4. `C2`（~20）→ 5. `C1`（~20）→
  6. `A1`（~20，2×64K 波次 prefill 主导 ~13 min）→ 7. `A2`（~20）→ 8. `A3 --agent-turns 4`（~30）。
  命令示例：
  `REF_B=concurrency/conc-B1-<ts> ./multi-concurrency-bench.sh --scenario B1,B2,B3`
- 执行前置条件：F/G 战役收尾、`build-ac922` 二进制为最终态；开跑前先用生产 preset 跑一次
  `1:8192` 单路回归（应 ≈100 tok/s TG），确认树没被动过。

## 4. 风险清单

| # | 风险 | 暴露面 | 缓解 |
|---|---|---|---|
| R1 | **GDN 并行序列正确性** | b11374 的修复是 OpenVINO-only（§1.4 亲验），我们 CUDA 路径的既有 parity 证据限于已测组合；岛×B×64K、FIFO×多路、6 slot 均为组合首验 | 每臂 512-ID 逐位 parity 门（matrix_runner 现成）；R19 已修 QSA radix top-k 非确定性（stable topk），同 prompt 重复性有独立证据；任何 parity=False 即停该臂不修数值、降级回退 |
| R2 | **--cache-reuse 三重隐患**（seq_rm 超界 false 被忽略 / seq_add 不作废 recurrent 状态 / ctx-checkpoints=0 下中段复用无恢复手段） | 仅当启用 n_cache_reuse 或 prompt 非纯增长 | 主矩阵**不开** cache-reuse；agent 模式只做纯追加轮次；若未来要开，先 `--ctx-checkpoints >0`（注意每 checkpoint 常驻 ~450 MiB 逻辑 recurrent 状态的显存代价）并单独过 parity |
| R3 | **B@8K 劣势未解**（接受率混淆假设 + B round-4 8K parity 空白） | 4×/6×8K 时 B 会被长度门**自动激活**（聚合 ≥32768），与 2×8K 的劣化前提不同但同族 | B2 两臂只跑 2:8192 定方向；C 场景如现异常decode劣化，补一臂 `LLAMA_QSA_POOLED_MULTI=0` 对照（脚本改一行）；F16_GATHER=0 可再细分归因 |
| R4 | **KV/显存 OOM 边界** | GPU-PLE 132K 曾 OOM（R16/R20 先例：聚合余量 ≠ 每卡余量）；岛 A 卡权重 8.6 GiB 是最紧卡 | PLE 保持 CPU（PLE_SHARD=0 写死）；A/B 池 266240 均小于已启动过的 409600；C 池 98304 更小；启动日志核对 minfree；OOM 降级路径写入场景表（SLOT_CTX 12288 / slots 5） |
| R5 | **CUDA graph 多形状** | 6 slot 纯 decode 形状族 {5,10,…,30} 行 + prefill 形状，默认 LRU cap=4 会驱逐抖动（B round4 实测 81 次驱逐先例）；首波每形状捕获 ~1–3 s | `GGML_CUDA_GRAPH_VARIANTS_CAP=8` 已入 C 场景 env；首波不计入稳态判读；SPEC_PAD_MULTI 因回滚预算数学在 MTP4 满深度恒为 0（§1.3 代码级结论），不依赖 |
| R6 | **FIFO×多路未验**（R18 明确 scope 仅 1×8K） | A3/B3 | 两臂均以对应基线为 parity 参考 + 明确回退规则；FIFO 输出 view 的 tail 间隙问题（R15 F 轮记录过同类）由 parity 门兜底 |
| R7 | **4×64K 满窗池贴边**（262144+2048 = 266240 整池，历史 6.41 tok/s 疑近池抖动） | 本矩阵不含 4×64K | 如后续要测：池 409600（已验启动）+ 预期低每流，勿复用 266240 |
| R8 | 运维 | vLLM 自启复活、日志追加污染、58C、编译 --parallel 6、32GB RAM 单节点 | 脚本预检 + 全新 label + matrix_runner 58C 守护；本方案零编译 |

## 5. 预期数字总表（口径：每流 512 输出、greedy；"聚合"=sum of per-flow server TG）

| 场景 | 聚合 decode 预期 | 来源 | 单流 PP 预期 | 来源 |
|---|---|---|---|---|
| A1 2×64K 六路 B+H | 78 | [M] bh-2x64k | 175 | [M] |
| A2 2×64K 岛+B+H | 78–103 | [E] 78×1.32（岛因子，2×8K 实测） | 200–260 | [E] 153×1.14×1.32 |
| A3 2×64K 岛+B+H+FIFO | =A2±2% | [E] R18 TG +2.27% 迁移 | 260–330 | [E] A2×(1.1–1.29) |
| A3-agent 增量轮（turn≥2 prefill） | decode 不变 ≈39/流 | [E] | prompt 210–330s → **1–3s** | [G] delta 算术 |
| B1 2×8K 岛 C+H(B关) | 92.1 | [M] C18 | 300 | [M] C18 |
| B1 4×8K | ≈83 | [E] 92.13×(68.08/75.29) | ~90–150 | [E] 线性摊薄 |
| B2 2×8K B 强制 | 42–95（风险带） | [E] 下界 [M] | — | — |
| B3 4×8K 岛+H+FIFO | ≈83±2% | [E] | B1 的 1.1–1.3× | [E] |
| C2 6×8K 六路 H+B | 70–90 | [G] | — | — |
| C1 6×8K 岛 H+B | 85–100 | [E] C2×1.32 两跳 | — | — |

**对用户目标的直接回答**：若 A2/A3 与 C1 达标，生产多并发配方 = **岛 3,3 + H + B(默认门) + FIFO 三件套 +
fair_share + 对应 slots/池**，长上下文双 agent 聚合 ≈ 80–100 tok/s（vs 单路 91 口径，翻倍且两人共用），
短上下文极限并发聚合 ≈ 85–100 tok/s；任何一臂失守按 §2 回退列降级，不影响其余配方落袋。

## 6. 明确不做（本轮边界）

- 不跑任何 GPU 测试、不编译、不动 fork 树、不碰生产服务（本文件成稿时全部完成的是读代码+读记录）。
- 不启用 --cache-reuse / --ctx-checkpoints / SPEC_PAD_MULTI / F16_GATHER=0 / 概率采样（理由见 §1.3/§1.4/§4）。
- 不 cherry-pick 任何上游 commit（889edf43dd 移植列为并发战役后的可选项）。
- 不测 4×64K 满窗（R7）。
