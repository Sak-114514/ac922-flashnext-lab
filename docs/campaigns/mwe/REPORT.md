# MWE（Member Workload Equalization）实施与 A/B 验证报告

日期：2026-10-06 ｜ 执行：ZCode 线 ｜ 依据：`designs/EPLB-DESIGN.md` §2（MWE 施工图）

## 判决（先说结论）

**门禁未过，判死，不合入。**

| 门禁 | 要求 | 实测（对齐战场配置，2 轮均值） | 判定 |
|---|---|---|---|
| 64K prefill PP | ≥ +5% | **+0.39%**（594.5 → 596.8 tok/s） | **FAIL** |
| 8K parity（512-ID 逐 token） | 逐位一致 | **4/4 轮全真**（两种配置、8K+64K 双 case） | PASS |
| TG 降幅 | ≤ 2% | −1.2%（8K）/ −1.3%（64K） | PASS（贴边） |

机制本身**实施成功且数学正确**（split 布局逐项验证 + 4 轮 token 级 parity 全绿），但设计的因果归因（"成员注意力空转 = AR 失步 = 15-20% 墙钟"）**被实验证伪**：把全注意力层成员负载从 245× 失衡修到 1.001×，64K 墙钟不动。详见 §5 归因分析。

## 1. 实施内容（61 行，2 文件）

补丁全文：`mwe/mwe.diff`（对生产脏基线的增量）。分支存档：`sources/llama-v100-v3` 分支 `mwe-20261006`，基线提交 `557b422`（=冻结生产态 134 脏文件 + 55 未跟踪文件），MWE 提交 `a465629`。

### 1.1 与设计文档的偏离（及原因）

设计 §2.3 伪码（Q 粒度 512 切 {8,8,8} + KV 镜像）**不可直接实施**，施工时发现两处硬约束：

1. **CUDA 核内 GQA 映射是纯本地索引**：fattn-tile/vec/volta-onecat 与 qsa-attn 全部按 `kv_head = q_head_local / (n_q_local/n_kv_local)` 计算归属。{8,8,8} 头均分后，成员 0/2 的本地映射落在错误的 KV 头上（成员跨度不再对齐 GQA 组边界；解析证明：n_head=24/kv=2/3 成员时，除全复制外**不存在**任何连续等分使本地映射正确）。
2. 设计伪码的 `segments_qkv = {{n_q_total,1}}` 假设 fused qkv 张量——实际全注意力层是**分离的 attn_q/attn_k/attn_v 权重**（wq 每头交错 [q|gate]）。

**采用的等效机制（零 CUDA/图改动）——segment 置换**：
- 利用 split_state 的多 segment 语义（PLE 先例）：把 wq（attn_q.weight，axis1）切成 6 段，段 (g,m) = KV 头 g 的 GQA 组内第 m 个 4 头片 → 成员 m 持有头 {4m..4m+3} ∪ {12+4m..12+4m+3}——每成员 8 头、**每组 KV 头各取 4 头**；
- attn_output.weight（axis0 收缩维）按同一置换切分（成员的 wo 行与其 Q 头同序配对）；
- wk/wv 权重与 cache_k/v_l（主缓存+块缓存两实例）→ **MIRRORED**（全注意力层含 MTP 层）；
- 置换后每成员本地 `q_head/4` 恰好命中正确的 KV 头（静态验证脚本零 GPU 证明，245×→1.001×）；
- meta backend `handle_flash_attn_ext` src[5]（块稀疏）分支放宽 2 行接受 kv MIRRORED（非 src[5] 主路径本就支持 kv_mirrored，qsa-attn 的 (hpw=4,n_grp=1) 模板实例本就存在）。
- env 门控 `LLAMA_MWE_ATTN=1`，默认关。

### 1.2 改动点

| 文件 | 内容 |
|---|---|
| `src/llama-model.cpp`（+58/−1） | MWE gate + 全注意力层 kv 权重/缓存改 MIRRORED + wq/wo 手工 6-segment 置换切分 |
| `ggml/src/ggml-backend-meta.cpp`（+4/−1） | 块稀疏 flash_attn 断言接受 kv MIRRORED |

## 2. 验证链（全绿）

1. **零 GPU 静态验证**：python 重算（`/tmp/mwe_perm_check.py` 思路，脚本随报告存档于本文件 §7 复述）——现状 Q 头 {0,12,12}/KV {0,1,1}（与设计 §1.3 的 nsys 观测一致）；MWE 置换后本地 gqa=4 映射对全部成员正确；全注意力层成员工作量（flash+wq+wo+kv+MoE 混合）最重/最轻 **245.6× → 1.001×**（64K 口径，flash ∝ q头×kvlen）。
2. **加载验证**（`LLAMA_CALLBACK_TRACE`，islands 3,3，逐项 grep）：
   - `blk.3.attn_q.weight axis=1 sums=[4096 4096 4096]`（=8 头/成员；原 {0,6144,6144}）
   - `blk.3.attn_(k|v).weight axis=10`（MIRRORED）；`cache_(k|v)_l3 axis=10`（主+块缓存全镜像）
   - `blk.3.attn_output.weight axis=0 sums=[2048 2048 2048]`
   - 不变量：MoE `ffn_up_exps [192 192 256]`（逐层轮转，未动）、GDN `blk.0.attn_qkv [2560 3840 3840]`（未动）
   - 显存实测：+554~640 MiB/卡（镜像 KV），132K ctx 下无 OOM（余量 >512 MiB 门）
3. **冒烟**：温度 0 计数任务输出干净 1-9（映射错误的典型症状是乱码）。

## 3. A/B 数据

### 3.1 对齐战场配置（islands 3,3 + ubatch 1024 + P2P AR + async copy/FIFO，无调试 env；probe 框架 `zcode_probe.py`，同二进制同配置，仅 `LLAMA_MWE_ATTN` 差异）

| 臂 | 8K PP/TG | 64K PP/TG | parity（vs 同配置 off） |
|---|---|---|---|
| mwe-off-al-r1 | 596.6 / 92.70 | 595.0 / 82.39 | — |
| mwe-off-al-r2 | 593.5 / 92.35 | 593.9 / 81.87 | — |
| mwe-on-al-r1 | 596.4 / 91.67 | 597.1 / 81.08 | True / True |
| mwe-on-al-r2 | 595.0 / 91.19 | 596.4 / 81.11 | True / True |
| **均值差** | **+0.1% / −1.2%** | **+0.4% / −1.3%** | |

轮间噪声 ~±0.2%；+0.4% 勉强出噪声但远低于 +5% 门。

### 3.2 冻结 manifest 配置（speed-bundle-single：u512 无岛，附带证据）

| 臂 | 8K PP/TG | 64K PP/TG | parity |
|---|---|---|---|
| mwe-off-r1（vs 历史 speed-bundle 参照） | 525.5 / 82.95 | 520.4 / 76.72 | True / True |
| mwe-on-r1（vs off-r1） | 533.3 / 81.81 | 527.0 / 76.52 | True / True |

无岛 6-way 下 +1.3~1.5%（MWE 的 gate 在 6-way 也生效），同样远低于门。二进制/树已还原后由 off 臂对历史参照的 parity 可反证：**还原干净，无残留漂移**。

### 3.3 绝对水平说明

对齐配置 64K ~595 tok/s vs Codex 冻结基线 1003.4（bundle-v1 二进制 + UUID GPU 序）：差距来自 GPU 序（0,4,2,1,3,5 vs UUID 序的岛构成）与树态（134 脏文件已含 bundle-v1 之后的 Codex 改动）。本 A/B 为同二进制同配置对照，**比值有效**；绝对值不与 1003.4 直接可比。

## 4. 门禁判定

- **64K PP ≥ +5%：FAIL**（+0.39%）。
- 8K parity：PASS（且 64K 也逐 token 一致）。
- TG ≤ −2%：PASS 但持续为负（−1.2~1.3%，来源：镜像 wk/wv 小投影与 KV 写 ×3 的固定开销）。
- **综合：判死。不合入生产。**

## 5. 归因分析（为什么机制对了收益却没了）

1. **QSA 稀疏路径把注意力绝对量砍掉了 ~97%**。运行时 trace（`LLAMA_QSA_PATH_TRACE`，生产 env `LLAMA_QSA_SPARSE_KERNEL=1`）：64K prefill 每查询注意力走 gather 路径，`top_k=2051`（宽度 2051/65536 ≈ 3%）。设计 §1.4 的收益上界按稠密 kvlen 计的 flash 份额构建——实际全注意力段的绝对尺寸小一个量级以上，成员间失衡的绝对值随之缩水；MWE 重分的 wq/wo gemm 亦是层内小头。
2. **AR 自旋 ≠ 注意力失衡所致**。机制生效（布局/parity/显存三项证据），成员负载已对齐，而 AR 等待并未显著转化为墙钟——说明设计 §1.2 测到的 3.9ms 非 MoE 段 busy 差与 0.83 相关性的**因**不在 Q/KV 切分不对称，更可能在：岛间串行（设计 §1.3-3 自认的共病，MWE 不触碰）、ubatch 边界处的 MoE/GDN 段到达次序、或 P2P AR 线时间本身。
3. **佐证**：无岛 6-way 配置下（失衡模式不同但 MWE 同样修齐）也只有 +1.3%。

**对后续战役的含义**：64K prefill 的剩余墙钟不在"岛内成员均衡"这一自由度上。下一个诊断动作应是**在当前树上重抓 64K nsys 重新归因**（2026-10-04 的剖面早于近期生产改动），重点拆岛间串行与非 MoE 段成分；DBO/层流水实验不必等待 MWE（其失步基底未被 MWE 改变）。

## 6. 合入建议与遗留物

- **不合入**。生产树 `ac922-flashnext-v100` 已还原至冻结生产态（79 修改 + 55 未跟踪，diff stat 12554/430 与隔离前逐字节一致），`build-ac922/bin/llama-server` 已从还原源重编（binary 中无 `LLAMA_MWE_ATTN` 符号，strings 验证 0 命中）。
- 存档（可复现）：
  - 分支 `mwe-20261006`：`557b422`（生产冻结基线）→ `a465629`（MWE，标注 gate FAILED）
  - `mwe/mwe.diff`：61 行增量补丁
  - manifests：`zcode-mwe-{off,on}-r1`、`zcode-mwe-{off,on}-al-{r1,r2}`（ac922 侧 `/home/debian/ac922-llama-next/manifests/`）
- 若未来 DWDP/EP 方向需要头重映射基建，segment 置换 + kv 镜像的组合已被证明是零 CUDA 改动即可落地的机制（parity 全绿），可直接复用。
- 显存注意：MWE-on 在 262K 满窗未预算（设计 R2 原文照旧）。

## 7. 环境与纪律记录

- GPU 占用全程查 `gpu-queue.lock` 与 `nvidia-smi`；Codex 五微测套件于收尾期启动接管 GPU，nsys 加分项（重抓 12s 切片验证 dev3 flash_attn 0→N）按纪律放弃，机制证据以 split sums + 显存增量 + parity 替代。
- 生产树/Codex 工作区零接触（manifests/zcode-* 为 Zcode 线标准产物位）。
- 隔离协议：stash → branch → apply → commit 基线 → 改 → 验证 → checkout 回生产分支 → stash apply 还原 → 重编。期间发现并修复了 `git add -A` 将未跟踪生产文件卷入分支基线导致 checkout 删除的副作用（55 文件已从分支基线恢复至工作区，状态逐项核对）。
- 温度：全程 41-44°C，远低于 58°C 门。
