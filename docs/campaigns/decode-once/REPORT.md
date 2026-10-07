# dense-decode-once 战役报告 — IQ3_S dense nc5 解码去重（阶段 1 判死）

日期：2026-10-04 · 战役：dense-decode-once · 硬件：ac922（6×V100 sm_70，CUDA 12.4）
fork：`/home/debian/ac922-llama-next/sources/llama-v100-v3`（生产分支 `ac922-flashnext-v100` 全程冻结，实验全部在 `dense-decode-once` 分支）
依据：sass-audit REPORT.md §4.3 移交判断——"dense 家族 per-token 重解码是全审计唯一量化的 ≥25% 冗余，就地 decode-once 重构是无格式变更的正解"。

## 裁决（先说结论）

**判死。阶段 1 微基准门禁（≥20% kernel 加速）失败：实测 0.989×（−1.1%，两块 GPU 一致）。按纪律即刻停止，阶段 2（整机 A/B）、阶段 3（Q6_K）不进入。分支保留不合并，生产态已字节级复原。**

死因不是实现失误——正确性重构本身是**完美**的（与 stock kernel 逐位相同，CPU 参考 1.19e-5 双双过 2e-4 门）。死因是**前提不成立**：SASS 审计认定的"×5 权重重解码"在 nvcc 实际产物中不存在——编译器早已把全部 weight 侧解码 CSE 出 token 循环。手动提出 j 循环后热点循环的 SASS 与 stock **逐指令相同**（531 inst / 140 LOP3 / 80 IDP.4A，审计报告里的数字本来就是 CSE 之后的形态；80 IDP = 16 dp4a × 5 token，是不可约的 per-token MAC 本体，不是冗余）。

## 1. SASS 前后动态循环指令数（热点循环对照）

改法（分支 commit 618afef，仅 mmvq.cu 一个文件，+92/−13）：`mul_mat_vec_q` 增加末位模板参数 `decode_once=false`；nc>1 且非融合的 IQ3_S dispatch 传 true；kernel 内 `if constexpr` 走 `vec_dot_iq3_s_q8_1_multi`——iq3s_grid 查表、sign 构造（__vcmpne4/__vsub4）、scale nibble、block d 全部按 (row, k-block) 解码一次进寄存器，5 个 token 的 q8 激活对同一组 grid 寄存器做 dp4a 累加。per-token 运算顺序（dp4a 链、整数 scale 乘、d·ds·sumi）与 `vec_dot_iq3_s_q8_1` 完全一致。无 repack、无 MMA/QPN、无量化格式变更，nc1 路径零改动。

同二进制 A/B（两份 kernel 源文本逐字节提取自分支 mmvq.cu，同一 TU、同数据、生产 nvcc flags：`-O3 -use_fast_math -extended-lambda -std=c++17`，生产 launch 参数 grid(1280,1,1)×block(32,2)）：

| | stock（前） | decode-once（后） |
|---|---|---|
| 热点循环体（BRA 回边实测） | 0x520–0x2640 **531 inst** | 0x520–0x2640 **531 inst**（完全相同） |
| LOP3.LUT | 140 | 140 |
| IDP.4A.S8.S8 | 80 | 80 |
| LDG.E.CONSTANT.SYS | 56 | 56 |
| LDG.E.U16.CONSTANT.SYS | 19 | 19 |
| PRMT / I2F / FMUL+FFMA / HADD2.F32 | 16 / 10 / 20 / 7 | 16 / 10 / 20 / 7 |
| 静态总指令 | 808 | 808 |
| 每线程动态循环执行数（K=2048 单遍） | 1 × 531 | 1 × 531 |

**动态指令数前后零变化**——这是"CSE 早已发生"的直接证据。审计的"×5 结构"（80 IDP、140 LOP3）读反了：那是去重后的形态。kernel 时间的真实约束是 L1 重放：每 iter 56 条 scattered grid 查表 LDG + 19 条窄读，13.7µs 对比 ~2.8µs 纯发射估计，缺口即 L1 32-way 重放（与 iq2s_volta 注释描述的病灶同类）。

## 2. REG / spill / occupancy 前后

| | stock | decode-once |
|---|---|---|
| REG（cuobjdump） | 137 | 145（+8 = 8 个 grid_v 寄存器跨 j 存活） |
| spill stores/loads（ptxas） | 0 / 0 | 0 / 0 |
| STACK / LOCAL | 0 / 0 | 0 / 0 |
| __launch_bounds__ | (64,1) 不变 | (64,1) 不变 |
| 64 线程块理论驻块（64K reg/SM，8-粒度折整） | 144×64→**7 块/SM** | 152×64→**6 块/SM** |

零 spill 达标；REG 145 ≤ 255 达标；但分配粒度折整后 donce 少一个驻块（7→6），与 −1.1% 的轻微倒退量级吻合（未逐项隔离，保守归因）。**"不降 occupancy 档"约束不满足**——即使指令数持平，这次重构在该形状下也是净负。

## 3. kernel µs 前后（阶段 1，µs 口径，同二进制）

真实生产 shape：dense ssm_out **[2560×2048] nc5**（MTP verify 窗口），随机量化权重 + q8 激活（fp16 scale 消毒），300 iters × 7 rounds 交错 A/B，best/median：

| GPU | stock | decode-once | 加速比 |
|---|---|---|---|
| GPU5 | 13.708 / 13.718 µs | 13.858 / 13.872 µs | 0.989× / 0.989× |
| GPU4 | 13.653 / 13.667 µs | 13.804 / 13.810 µs | 0.989× / 0.990× |

**门禁要求 ≥1.20×，实测 0.989×（−1.1%），双卡稳定复现 → 判死。**

## 4. 94.3 配方端到端 TG

**未运行。** 阶段 2 只在阶段 1 过门后进入；判死即停，launch_integration_arm.sh 配方未触碰，94.3 tok/s 生产态不动。阶段 3 Q6_K（[6144×2560] nc5，scale I2F/narrow-load 提出）同样未进入——Q6_K 的 weight 侧重复（vl/vh/scales 窄读 + I2F）与 IQ3_S 同属编译器 CSE 覆盖范围，阶段 1 的证据直接适用，无翻案理由。

## 5. parity 结果

| 检查 | 结果 | 门 | 判 |
|---|---|---|---|
| CPU 双精度参考（解码语义从 vec_dot_iq3_s_q8_1 逐位推导；grid 表从 device 导出保证同表） | relerr(stock,cpu) = **1.19e-05** | ≤ 2e-4 | PASS |
| 同上，decode-once | relerr(donce,cpu) = **1.19e-05**（RMS-floor 口径，rms 7.0e4） | ≤ 2e-4 | PASS |
| stock vs decode-once | **逐位相同**（memcmp 全等） | — | PASS |
| nc1 stock 路径（未改动路径编译/运行健全性，small_k 配置） | relerr = 1.19e-05 | ≤ 2e-4 | PASS |
| 512-token server 级 parity（speed-bundle-single） | **未运行**（阶段 2 未进入） | — | N/A |

## 6. 继续 / 判死裁决

**判死，不合并，不翻案。** 审计移交判断中"唯一 ≥25% 指令冗余"的定性被同二进制实验证伪：冗余在源码层面存在、在 SASS 层面已被 nvcc 消除，"手动 decode-once"没有可移除的指令（动态计数 531→531）。dense 家族就地重构路线（不 repack、不 mma、不动布局）在此编译器/架构组合上**无空间**——后续战役不应再按"j 循环冗余"立项 dense quant。

**给下战役的移交判断**（本轮红线外，仅记录）：IQ3_S nc5 该形状的真瓶颈是 grid 查表的 L1 重放（56 scattered LDG/iter，fork 已有先例 `mul_mat_vec_iq2_s_moe_volta` 用 shared-memory staging 解决同类问题，注释载明 124→75µs）。若要动 IQ3_S，方向是 grid 表 shared 化/查表消除，属结构性改写，需重新立项过全套门禁，且预期收益仍受 dense 家族 ~20.5% 占比 × 单成员权重封顶。

## 7. git 分支操作序列（生产态精确恢复记录）

```
# 战役前（生产态）：branch=ac922-flashnext-v100 @7fe450e，工作树 77 M（四战役未提交改动）+ 55 ??（untracked 新文件）
git status --porcelain > /tmp/prod-state-snapshot-before.txt          # 132 行快照

# 进实验态
git stash push -m prod-kernels            # 77 个 M 入 stash（?? 文件不动，untracked 不受 stash/checkout 影响）
git checkout -b dense-decode-once          # 从同 HEAD 建分支
git stash apply                            # M 恢复到分支工作树（apply 不删 stash，留作双保险）
git add -u && git commit                   # 0d09f48 baseline：生产态快照提交（只 tracked，?? 保持 untracked）
# ... 在分支上改 mmvq.cu ...
git add ggml/src/ggml-cuda/mmvq.cu && git commit   # 618afef 实验归档（含判死结论）

# 回生产态（已执行，双验证通过）
git checkout ac922-flashnext-v100          # 树回 base（?? 文件原地不动）
git stash pop                              # 77 个 M 精确复原，stash 清空
git status --porcelain > /tmp/prod-state-restored.txt
diff <(sort /tmp/prod-state-snapshot-before.txt) <(sort /tmp/prod-state-restored.txt)
# → PRODUCTION STATE RESTORED EXACTLY（132 行全等）
# mmvq.cu md5 = ca4da8013bfbc97b694e653cb0b929e5，与战役前本地存档一致
```

分支拓扑：`dense-decode-once` = 7fe450e → 0d09f48（生产基线）→ 618afef（实验+判决），未合并。两态互切任一时刻可重放：生产态 = 分支名 + stash（或 baseline commit）；实验态 = `git checkout dense-decode-once`。

## 红线合规

- 生产树冻结：全程分支隔离，恢复后 `git status` 与战役前快照全等，mmvq.cu md5 一致；build-ac922 生产构建目录未触碰（基准全部在 /tmp 独立编译）；
- 改动范围：仅 mmvq.cu（decode_once 模板参数 + 辅助函数 + kernel 内 if constexpr + dispatch 一处），nc1 路径与融合路径零改动（模板默认值 + if constexpr 结构保证）；未 repack、未 MMA/QPN、未改量化格式、未动 Q6_K 布局；
- 真实 shape：[2560×2048] nc5 生产形状 + 生产 launch 参数 + 生产编译 flags；
- GPU 使用：CUDA_VISIBLE_DEVICES=5/4 各约 3 分钟微基准，跑前跑后 nvidia-smi 确认 0 MiB / 0%；结束态 6 卡全 0、无残留进程，/tmp 工作目录已清。

## 资产

`~/Desktop/ssh/flashnext-llama-experiment/decode-once/`：
- `artifacts/`：bench_decode_once.cu（同二进制 A/B 微基准：双 namespace 逐字节提取 kernel + CPU 双精度参考 + grid 表导出）、inc1/inc2.inc（从分支 mmvq.cu 标记提取）、census_bench.py（SASS 静态+回边循环 census）、census_final.txt、resource_final.txt、bench_gpu{4,5}.txt（原始输出）、extract_incs.py；
- `work-mmvq.cu`（分支上部署的完整改动版）；`orig/mmvq.cu`、`orig/vecdotq.cuh`（战役前生产态存档，md5 锚点）；
- 远端：分支 `dense-decode-once`（0d09f48 + 618afef）；`/tmp/prod-state-{snapshot-before,restored}.txt`（恢复证据）。
