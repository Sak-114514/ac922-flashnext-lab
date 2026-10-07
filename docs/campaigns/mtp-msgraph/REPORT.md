# mtp-multistream-graph 专项报告：多流 MTP draft/verification CUDA Graph shape drift 审计与修复判死

日期：2026-10-04 | 主机：ac922 (6×V100 sm_70, CUDA 12.4) | fork：/home/debian/ac922-llama-next/sources/llama-v100-v3 @ ac922-flashnext-v100 (7fe450e)，分支 mtp-msgraph（隔离，未合入）

## 一句话结论

**P1 前提被测量推翻**：双流联合 decode 的 draft "shape 漂移" 不是漂移，是 catch-up(10 行,0 输出) ↔ draft-step(2 行,2 输出) 的**结构性交替**；且 **CUDA Graph 在交替时本来就走 replay**（capture 537 次 ≈ 512 次 prefill n_kv bump + ~25 次 decode bump，零次来自交替）。全部可回收损失 = llama 层 rebuild 的主机成本 ≈ 0.78s / 联合 decode 窗口 ~11.5s ≈ **+6-7% 上限**，低于 +10% 双门禁。三条修复路线全部实测判死：跨槽复用撞 meta backend 断言（生产敏感层，红线邻区）；单形状 padding 使 draft 逐步成本 ×3（TG -44%）；n_kv 桶粗化对聚合吞吐持平。**不合入，生产树原样复原。**

---

## 1. 阶段1：只读 trace 基线（生产二进制，纯 env，无源码改动）

配置：A2F 配方（岛 3,3 + 2×64K + fair-share）+ `LLAMA_STEP_TIMING_PERIOD=1`（逐步）+ `GGML_FT_HOST_TIMING` + `GGML_FT_GRAPH_CENSUS`。
日志：`logs/msgraph-A2F-t1.log`；结果：`concurrency/msgraph/A2F-t1.json`。
**基线复现**：TG 41.92/36.34，聚合 78.26（P0 锚 74.73，+4.7% 噪声带内）。

### 1.1 shape 直方图（回答：有限桶还是连续漂移？——**有限且极小**）

draft ctx（decode 相，616 步）：

| shape (tokens) | 步数 | rebuilt | 平均 compute (rebuilt/reused) |
|---|---:|---:|---|
| 10（catch-up，双槽满深度恒定） | 103 | **103 (100%)** | 5.19ms / — |
| 2（draft step，双槽） | 414 | 107 (26%) | 3.05 / 1.58ms |
| 5（catch-up，单流尾段） | 19 | 19 | 3.75ms / — |
| 1（draft step，单流尾段） | 80 | 20 | 2.15 / 1.24ms |

- p_min=0.00、n_max=4、无 adaptive → **每槽恒满深度 4**：catch-up 恒 = Σ(n_drafted+1) = 10，draft step 恒 = 2。P0 的 "tokens 2..10 漂移" = 这两个结构形状的交替，不是随机漂移。
- rebuilt 归因：**245/249 发生在 shape 切换上**，same-shape（n_kv 桶跳）仅 4 次。
- prefill catch-up 相恒定 (32,0)，rebuilt 512/4096 = 每 8 步一次（32×8=256 桶跳，符合预期）。

target ctx（verify）：decode 相 (10,10) 恒定 103 步仅 5 次 same-shape rebuild + 尾段转移 3 次；单流参照（qsa-lc-r1.log）(5,5)/(6,6) rebuilt=0。**验证图本来就是稳定的。**

### 1.2 rebuild 计数来源澄清

- `step-timing ... rebuilt=N` = llama-context `process_ubatch` 的 `st.n_build`：**llama 层图重建**（build_graph + sched reset/alloc + evaluate），非 CUDA capture 计数。
- CUDA 层（GGML_FT_HOST_TIMING destroy 统计）：draft 后端 capture=537/device ≈ 512(prefill 256 桶跳) + ~25(decode 桶跳)。**交替 rebuild 不触发 capture**——两个 gf_res_prev 槽各自 arena/galloc 地址确定，node props 稳定，CUDA 图 replay 命中。
- P0 的 "per-rebuild ~60ms" 复核：A2（无 fair-share）串行预填充争占的伪影。A2F 条件下真实成本：draft 交替 rebuild +2.4-2.7ms（主机 build 0.15 + alloc ~1.0 + eval 差额），**验证侧 topology rebuild +115ms**（172.6 vs 57.0ms）。
- 图缓存（FreeToken variant cache）确实在抖（saves=529/evictions=525/hits=1），但那是 llama 层重建产生新 key/props 的下游症状，不是根因。

### 1.3 根因（哪个维度贡献最大）

**`gf_res_prev_active == res` 单活动指针**（src/llama-context.cpp:1773）：`gf_res_prev[2]` 双槽按 n_outputs>0/==0 分居（catch-up→槽0，draft step→槽1），但复用条件要求"上一个计算的 ubatch 就是本槽"——每轮 2 次交替全部拒绝复用 → llama 层全量重建（主机 ~2.7ms/次 × 245 次 ≈ 0.66-0.78s）。n_kv 桶跳（256 对齐，get_n_kv 内置）贡献 ~1/32 步，非主要项。

---

## 2. 方案设计与实测（三条路线，含数学）

### 2.1 路线 A：跨槽图复用（Fix 1'）——**判死：meta backend 断言**

设计：交替时 sched 同步 + reset + 重指向本槽图（跳过 build_graph；张量对象未变 → galloc/CUDA props 稳定 → replay）。
实测：服务端 abort `ggml-backend-meta.cpp:2909 GGML_ASSERT(bcj.nodes[i])`。机理：meta backend 的 simple_tensors 仅在 galloc init_tensor（张量 data==NULL 时）创建一次；uid 键控的 meta rebuild 清空容器后假定张量对象是新建的。跨槽复用保留旧对象 → 查找 miss → 断言。修复需动 meta 层（岛/多 GPU 调度核心，生产敏感）——红线邻区，放弃。已回退。

### 2.2 路线 B：draft 侧单形状 padding（Fix 3, `LLAMA_MTP_PAD_MULTI`）——**判死：成本 ×3 + 序列切分反噬**

数学（先算清，按任务要求）：
- 目标侧 `LLAMA_SPEC_PAD_MULTI` 的已知障碍复核成立：`n_pad ≤ n_rs_seq − (n_rows_seq−1)`，MTP4 满深度验证行=5、n_rs_seq=4 → n_pad ≤ 0，目标侧永远补不动。本路线只动 **draft 侧**（ctx_dft 的 `cparams_dft.n_rs_seq = 0`，无该上限）。
- draft 侧真实障碍在别处：`llama_memory_recurrent::seq_rm` 部分回滚要求 `1 ≤ rollback ≤ n_rs_seq`，draft n_rs_seq=0 → 尾部回滚被拒 → pad 行 KV 残留（靠同位置覆写自愈），且 **hybrid-idx memory 把多序列 batch 按序列切成多个 ubatch**（llama-memory-hybrid-idx.cpp:183 的 trailing-rows 规则）：padding 到 (10,10) 的 draft step 实际被切成 **2+8 两个 ubatch**，每轮出现 4 种形态 (10,0)/(10,10)/(2)/(8)。

实测（A2F-fix3b，MTP_PAD_MULTI=10）：形态序列 10,2,8,2,8,…，**rebuilt 935/1144 = 81.6%（比基线 40% 更糟）**，draft 逐步 compute 4.5ms（10 行注意力×66K KV + 10 行 logits，vs 2 行 1.52ms），acceptance 轻微劣化（槽1 0.737→0.637，pad 行写 KV 于 pos 上方）。**TG 21.15/22.93，聚合 44.08（-44%）**。
结论：padding 统一宽度 = 用 5× 注意力换 replay，净损失；且多序列 batch 无法保持单 ubatch，形态反而碎片化。**与单流 MTP_PAD=6 不可类比（n_seq=1 无切分、无交替）。**

### 2.3 路线 C：decode n_kv 桶粗化（Fix 2, `LLAMA_DECODE_KV_PAD`）——**判死：持平**

`get_n_kv` 已有 256 对齐；新增 env 将其粗化（如 1024，幂等 256-65536）。注意力浪费 ≤ pad/2 每 query 行（131K ctx 下 ≤0.4%），数学无害。
实测（A2F-kvpad，KV_PAD=1024，padding 关）：聚合 **77.38 vs 78.26**——持平（-1.1%，噪声内）。原因：验证侧 rebuild 的主要触发不是 kq-mask 的 n_kv 维度，而是**图拓扑本身随 QSA 池化路径状态变化**（sched-reserve trace：target 图节点数在 7259/7395/7547 间变化）——那是 P2 的领地（QSA 桶取整），且该 run 中 target 后端 capture 数反而升高（5760 vs 177），进一步确认拓扑 churn 与 KV_PAD 无关。

---

## 3. 门禁结果

| 门禁 | 要求 | 实测 | 判定 |
|---|---|---|---|
| 8K 单流 parity（512-ID 逐位） | 不许破 | **matched=true 两次**（fix 后初验 + 终态二进制复验，均逐位一致；TG 93.14/92.91 vs 生产 93.38） | PASS |
| rebuilt/32 | ≤2 | padding 臂实际 935/1144=81.6%（多序列 batch 被切成 2+8 双 ubatch，形态反而碎片化）；不 padding 维持 12-14/32。唯一 rebuilt=0 的观测是 fix3b 单流尾段（单序列无切分），不构成机制 | FAIL（无同时满足两指标的机制） |
| 2×64K 每流或聚合 | ≥+10% | 最好干预 = 持平（77.38 vs 78.26）；理论上限 +6-7%（0.78s/11.5s） | **FAIL** |
| B3 4×8K 守护 | ≥103.43×0.98=101.36 | **101.03（-2.3%）**：flows 0/2 逐点复现（27.07→27.08 tok/s，bit 级一致），偏差集中在 flows 1/3（23.1-23.8 vs 25.2-25.6）= 调度噪声；修复 env 关（休眠）+ parity 逐位 → 非代码路径回归 | 边缘（见注） |

注：B3 复测时修复全部休眠（env 未设），二进制含新增 env 检查分支；同二进制 8K parity 逐位通过。首门禁已 FAIL，B3 不改变裁决。

三组关键数字（2×64K 聚合 TG）：基线（本会话 A2F-t1）**78.26** / P0 锚 74.73；修复最好臂（KV_PAD）**77.38**；padding 臂 **44.08**。

---

## 4. 裁决

1. **mtp-multistream-graph 合入线：判死。** 证据链：多流 decode 图在 CUDA 层已经全 replay（本专项最反直觉的发现）；"shape drift → 40-60ms 重建 → +20-30%/流" 的 P0 假设链在 A2F 条件下每环都不成立（40-60ms 是 A2 串行预填充争占伪影；draft 重建真实成本 2.7ms 主机时间；总池 ≤ +6-7%）。
2. **draft 交替的 llama 层重建（2×2.7ms/轮）**：唯一根治路径是 sched/galloc 支持双图驻留或 meta backend 容忍跨槽复用——均为跨生产核心层手术（红线邻区），收益上限 +6-7% 不值风险。不立项。
3. **验证步 50-62ms vs 单流 24-26**：确认为聚合统一 KV 注意力固有计费（P0 判断成立）。若要再压需 per-slot KV 隔离（memory 布局手术）——超出本专项红线，不立项。
4. **真正的残余低效移交（按优先级）**：
   - **P2'（升级）target 图拓扑 churn**：验证图节点数随 QSA 池化状态在 7259↔7547 间变，每次拓扑转移 ~115ms（2×64K decode 窗口 5-8 次 ≈ 0.6-0.9s ≈ +6-8%）。落点：qwen4exp QSA 池化路径的桶取整（与原 P2 prefill QSA 钳制同族，建议合并立项）。
   - draft host rebuild 池（+6-7% 上限）只在 P2' 落地后值得再看（那时交替重建占比才显著）。
5. 分支 `mtp-msgraph` 保留全部机制代码（`LLAMA_MTP_PAD_MULTI`、`LLAMA_DECODE_KV_PAD`，均默认关、休眠无害）与证据，**不合入**。

## 5. git 序列（在案）

```
# 隔离（会话开始，生产态 132 M 文件 @ ac922-flashnext-v100 7fe450e）
git stash push -m prod-kernels          # stash@{0}
git checkout -b mtp-msgraph
git stash apply                          # 132 M 复现于分支
# 编辑（分支内）：common/speculative.cpp（+LLAMA_MTP_PAD_MULTI 机制+n_outputs_max 提升）、
#                src/llama-kv-cache.cpp（+LLAMA_DECODE_KV_PAD）、
#                src/llama-context.cpp（Fix1' 加入后实测判死回退，净零）、
#                ggml/src/ggml-backend.cpp（临时 reserve trace，已移除，净零）
# 复原（会话结束）
git stash push -m mtp-msgraph-work      # stash@{0}（保留：分支工作全量可恢复）
git checkout ac922-flashnext-v100
git stash pop stash@{1}                  # prod-kernels → 工作树复原 132 M
# 复核：git status = 132 M（与会话开始一致）；stash list 仅剩 mtp-msgraph-work；分支 mtp-msgraph 保留（干净树）
```

产物（远程）：logs/msgraph-{A2F-t1,A2F-fix1pad,A2F-fix3,fix3b,A2F-kvpad,parity8k,parity8k-final}.log、concurrency/msgraph/*.json、scripts/msgraph-trace*.sh、msgraph-parity.sh、concurrency/conc-B3-20261004-212755/。

## 6. GPU 收尾

六卡 0 MiB / 0%（复原后复核；测试 unit 均已停止）。
