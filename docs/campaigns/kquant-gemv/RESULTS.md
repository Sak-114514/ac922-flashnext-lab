# K-quant GEMV (dense 量化投影) 阶段一结果 — 诚实收档报告

日期：2026-10-04 · 战役：qmmvq 后续（kquant-gemv）· 硬件：ac922 V100 sm_70（GPU5，协商分卡）
基线：fork mul_mat_vec_q（审计 §2.1）· 对照：copy 870 GB/s

## 判决（先说结论）

**未达标，不进阶段二接线。**

| 目标 | 现状 | 判决 |
|---|---|---|
| Q6_K 大形状 70%+ / 1.5× | **0.40-0.63×（16.7-34.4%）** | 败退（低于 1.3× 线） |
| IQ4_XS（第二优先） | 0.53-0.56× | 败退 |
| IQ3_S（第三优先） | **1.01×（19.4% vs fork 19.2%）** | 打平，无净收益 |
| 正确性 | 全形状 rms-relerr ≤ 1.1e-6（vs fork kernel 输出） | **过门**（K=256 三向 1e-7 级） |

任务书原目标 1.8-2.5×，协调后修订 1.5×——均未达到。GGUF 原生 K-quant 布局下，
"流式宽读 + 就地解码"路线在 V100 上没有跑赢 fork 的 2 字节窄读实现。

## 最终数字（run-final.log，200 iters stream-loop，≥24MB 复制品轮转）

| 形状 | fork µs / % | mine µs / % | speedup |
|---|---|---|---|
| q6k N2560×K2048 nc5 | 13.4 / 36.9% | 22.8 / 21.7% | 0.59× |
| q6k N2304×K2560 nc5 | 15.2 / 36.7% | 28.8 / 19.4% | 0.53× |
| q6k N3840×K2560 nc5 | 24.5 / 37.8% | 46.2 / 20.1% | 0.53× |
| q6k N6144×K2560 nc5 | 35.2 / 42.1% | 88.9 / 16.7% | 0.40× |
| iq4xs N3840×K2560 nc5 | 15.5 / 38.8% | 29.3 / 20.5% | 0.53× |
| iq4xs N2560×K2560 nc5 | 11.8 / 33.8% | 21.3 / 18.8% | 0.56× |
| iq3s N2560×K2048 nc5 | 13.5 / 19.2% | 13.4 / 19.4% | 1.01× |
| q6k N2560×K2048 nc1 | 10.2 / 48.3% | 16.3 / 30.4% | 0.63× |
| q6k N6144×K2560 nc1 | 23.3 / 63.8% | 43.1 / 34.4% | 0.54× |

## 三代架构的实测路径（全部数字实测，剖析方法见"方法资产"）

**隔离测量（决定性）**：stonly.cu / KQ_BPT_LOADSONLY 剖析变体。
- 流读骨架（uint4 对齐 cover + 归一化窗口）单独跑：**78.8%（18.8µs）**——加载模式本身接近最优。
- 完整 kernel：88.9µs——**decode 相吃掉 ~70µs，是 loads 的 3.7 倍**。三代架构都倒在这里：

| 版本 | 结构 | q6k N6144 nc5 | 瓶颈实测 |
|---|---|---|---|
| v1 | shared 原样 staging + lds_u32（u16 对）解码 | 46%（R2T64） | 每 16 元素 18 次窄 LDS × ~2 路银行冲突 ≈ 每 SM 13.7K shared 占用周期 ≈ 9µs |
| v2 | shared 重排（PRMT 桶形移位入库，LDS.128 窗口） | 24% | staging 串行化（shfl 依赖 DRAM 延迟）+ 相位化 barrier |
| v3-v5 | BPT：每线程独占一个 210B 超块，14×LDG.128 全进寄存器，PRMT 归一化 12 个 16B 窗口，dp4a 全寄存器解码 | 15-17% | loads-only 78.8% 但 decode 延迟链（y 的 40 次/线程 L1 读 + 依赖串行）× 低占用（122 寄存器→3-4 块/SM→每 SMSP ~4 warp）压不住 |

BPT 的三个修正历程（SASS 证实）：v[] 数组引用传参→落 local memory（224B stack，2× 回退）；
按需 __ldg 修复后编译器又自限 72 寄存器（`__launch_bounds__(T,1)` 解）；选择链 128 ops/窗口改
uniform switch（LOP3 475 条仍然偏高）。minBlocks=4/6 挤占用率收益 ±10%，无质变。

## 失败的根因判断（给下一战役）

1. **Q6_K 载荷碎片度与 V100 LSU/延迟的乘积**：210B/块 = 8 个 ql 窗口 + 4 个 qh 窗口 + scales/d，
   无论落 shared 还是寄存器，每 16 元素窗口都要 4-8 条重排指令（PRMT/shift），且窗口内 nibble 提取
   依赖 pad 相位。这套 ALU 在 A100/H100 的 LOP3-密集设计里有 Marlin 式解法，在 V100 的调度器
   上跑不出-issue 预算（实测 4440 指令/线程，issue 预算 ~6µs，实测 decode ~70µs = 12× 延迟放大）。
2. **fork 的 2B 窄读并非纯傻**：它的窄 LDS/LDG 由编译器排成 10 迭代 × 小步循环，天然软件流水；
   我们的全展开大块解码反而造成依赖链 + 寄存器高压的恶性循环。
3. **若要打穿此家族，需要离线 repack**（Marlin interleave / 自定义对齐布局）——coordinator 已标注
   为另一工程量；GGUF 原生布局下本轮判定为不可行（在 2+2 小时盒内）。

## 方法资产（可复用，全在 /home/debian/kquant-gemv/ 与本地镜像）

1. **三向对拍正确性方法**：fork CPU dequant（ggml-quants.c 逐字提取）× fork kernel 输出 × 新 kernel，
   K=256 单块形状下三者 1e-7 互证；配合 sim_q6k/sim2/sim3/sim4 主机复刻链（每代 decode 都先过 sim 再上 GPU），
   五个真实 bug 全部由这套方法定位（qh 基址漏 +128、q 错位、scale 漏 8h、header 选择链缺分支、
   staging 拷贝少 1 字、y slot 布局错序）。
2. **RMS 归一误差度量**：随机数据下逐元素相对误差被近零输出污染（同一 kernel 与自身 FP 序的差都能显示 3e-2）；
   bench 已改用 max|diff|/RMS(ref)，此后全形状 5e-7-1.1e-6。
3. **剖析变体**：KQ_BPT_LOADSONLY（纯加载）/ KQ_BPT_NODECODE（staging+sync）/ KQ_PROF_NOY/NOW（v1 相位分离）、
   stonly.cu（staging 模式矩阵）——每代的瓶颈定位都靠它们，全程零 ncu 权限需求。
4. **V100 sm_70 工程教训清单**（每条都有实测复现）：
   - `__syncthreads()` 在分歧分支两侧各放一个 = Volta ITS 下间歇死锁（bpr=1/2 形状挂死，bpr≥4 偶发）→ barrier 移到汇聚点之前；
   - 对寄存器数组引用传参 / 动态索引 → 整个数组落 local memory（ptxas -v 看 stack frame，SASS 里 STL/LDL 计数）；
   - `__launch_bounds__(T)` 不带 minBlocks 时 ptxas 可能自选 ~72 寄存器上限并把热数据压进 local；
   - 索引链对编译期常量的敏感度：m-loop 全展开后选择链退化为 LOP3 475 条，uniform switch 只省一半。
5. **内核文件**：kernel_sm70_kquant.cuh（BPT + raw 双路径、q6k/iq4xs/iq3s 三解码器、dispatch/chooser 完整），
   bench_kq_sm70.cu（fork 提取版对照 + RMS 门 + 轮转计时）——正确性过门，性能未过，留作参考实现。

## 边界与诚实声明

- 所有数字为独占 GPU5 单卡 stream-loop 计时（cudaGraph 在该环境不可用，审计已证）；
  fork 对照为同一二进制同形状。臂间波动 <3%（iters=200 × 3 trials 取最小）。
- IQ3_S 打平但非胜利：fork 该形状 19% 的瓶颈（审计判为查表）在本轮被证实同样压制共享 grid 驻留
  的新 kernel——查表驻 shared（iq2s_volta 先例）不足以打开它，与审计 §4.3 的预期不符，如实记录。
- 阶段二接线未启动（按纪律，未达标不接线）；fork 零改动。
