# 整层融合 launch：fused per-wave BF16 projection group kernel（DESIGN）

日期：2026-10-03。范围：本地设计 + kernel + 微基准 + 接线方案；**未改动 fork 源码树**（接线由另一代理执行）。

产出（本目录）：
- `fused_projection_sm70.cuh` —— 自包含 kernel + host 表构建/launch/持久表（照 `bf16-gemv/kernel_sm70_bf16.cuh` 的注释纪律）
- `bench_fused.cu` —— 微基准：GDN/QSA 整层真实形状组，逐矩阵 launch vs 每 wave 一次融合 launch，含 CPU 正确性参考
- `test_host_emul.cpp` + `syntax_check_prelude.h` —— 本地无 nvcc 的验证 harness（见 §7）
- 本文档

---

## 1. 问题（全部来自已有实测）

- R17 nsys（`bf16-gemv/WIRING.md`、`TECH-FACTS.md`）：decode 5.2s 内 77.8 万 kernel 事件（每 GPU ~15 万，均值 ~12µs/个），GPU busy 仅 ~65%；BF16 matvec 家族占 33% kernel 时间；`[2048×2560]nc5`（= wq 12288 行 6 卡行分片）只有 44% 带宽。
- 微基准已证单 kernel 到头（同文件二战数据）：形状小 → launch/尾巴占比大；**所以融合**。
- 部署形态：6× V100 SXM2、`tensor split 1×6`（`CURRENT-STATE.md:25`）、CUDA graphs 开启、CPU PLE、MTP4（验证窗 nc=5）。

## 2. 一层的 BF16 投影清单（代码证据）

模型常量（`target-header-preliminary.json`）：n_embd=2560，48 层（`full_attention_interval=4` → 层 3,7,…,47 共 12 层 QSA，36 层 GDN），hc=4 → hc_dim=10240、hc_lr=320；GDN：d_state=128、n_group=16 → key_dim=2048，dt_rank=48 → value_dim=6144，conv_dim=2·2048+6144=10240；QSA：head=24×256 → wq 行数 2·256·24=12288（q|gate 交错，`qwen4exp.cpp:2296-2298`），wk/wv=2 KV 头×256=512 行；indexer：4×128=512 / 128 行；FFN：512 专家（IQ3_XXS，不在本方案范围）、router 512 行、共享专家 640。

BF16 证据：`PREFILL-DIAGNOSIS.md:9`"Metadata contains BF16 HC/recurrent-gate/PLE projections. Full-attention Q/K/V are quantized"。
**矛盾待接线时核实**：R17 trace 的 12µs 实例化形状 `[2048×2560]nc5` = 12288/6，恰是 wq 的 6 卡分片——若该形状确属 wq，则 wq 实为 BF16（PREFILL-DIAGNOSIS 的"Q/K/V quantized"至少不覆盖 wq）。本 kernel 对任何 BF16 MUL_MAT 通用，清单按"HC/GDN 门控/PLE = BF16 确定；QSA 投影按 trace 疑似 BF16"标注；接线时用 `GGML_FT_GRAPH_CENSUS=1` 或 GGUF tensor 表逐个确认即可，不影响 kernel 设计。

形状记法 [N_out × K]（N_out 为**总分片前**行数；6 卡行分片后每卡 ≈ N/6）。

### 2.1 GDN 层（36 层，每 token 各 1 次；nc = decode 1 / MTP 验证 5）

| # | 张量 | 形状 [N×K] | 证据（src/models/qwen4exp.cpp） | 读的 y | BF16 |
|---|------|-----------|--------------------------------|--------|------|
| 1 | hc_attn_down | 320×10240 | 225, 386 | xn（normed 宽残差） | 是 |
| 2 | hc_attn_inject | 4×10240 | 227, 422 | xn（与 #1 同 y） | 是 |
| 3 | hc_attn_up | 10240×320 | 226, 388 | lo=silu(down)（依赖 #1） | 是 |
| 4 | wqkv | 10240×2560 | 247, 771 | cur（hc mix 出） | 是 |
| 5 | wqkv_gate | 6144×2560 | 248, 775 | cur（与 #4 同 y） | 是 |
| 6 | ssm_beta | 48×2560 | 252, 2394 | cur（与 #4 同 y） | 是 |
| 7 | ssm_alpha | 48×2560 | 253, 2401 | cur（与 #4 同 y） | 是 |
| 8 | ssm_out | 2560×6144 | 255, 2490 | GDN 核输出 | 是 |
| 9-11 | hc_ffn_down/inject/up | 同 1-3 | 229-231 | 同 1-3 | 是 |
| 12 | ffn_gate_inp（router） | 512×2560 | 267, 2517 | cur2 | 疑似 |
| 13 | ffn_up_shexp | 640×2560 | 273, 2539 | cur2（与 12 同 y） | 疑似 |
| 14 | ffn_gate_shexp | 640×2560 | 272, 2540 | cur2 | 疑似 |
| 15 | ffn_gate_inp_shexp | 1×2560 | 271, 2547 | cur2 | 疑似 |
| 16 | ffn_down_shexp | 2560×640 | 274 | act | 疑似 |

（MoE 专家 IQ3_XXS 不列；PLE 层另加 ple_key 10240×2560、ple_value 2560×2560，见 259-260——部署为 CPU PLE，GPU 无此二项。）

**每 GDN 层 BF16 MUL_MAT ≈ 16 次。**

### 2.2 QSA 层（12 层）

同 GDN 的 #1-3、#9-16（hc 6 个 + ffn 侧 6 个），注意力侧替换为：

| # | 张量 | 形状 [N×K] | 证据 | 读的 y | BF16 |
|---|------|-----------|------|--------|------|
| 4' | wq（q\|gate 交错） | 12288×2560 | 235, 2297 | cur | 疑似（trace 形状吻合） |
| 5' | wk | 512×2560 | 235, 2308 | cur（同 y） | 疑似 |
| 6' | wv | 512×2560 | 235, 2311 | cur（同 y） | 疑似 |
| 7' | index_q_proj | 512×2560 | 242, 1497 | cur（同 y） | 疑似 |
| 8' | index_k_proj | 128×2560 | 243, 1489 | cur（同 y） | 疑似 |
| — | wo | 2560×6144 | 236, 2365 | attn 输出 | 疑似 |

**每 QSA 层 BF16 MUL_MAT ≈ 17 次。** 与任务描述"每层 ~20 次独立小 kernel launch"量级一致（计入 PLE 层的 2 项和 view/reshape 附带小核后即 ~20）。

### 2.3 R17 的两个实例化对应

`mul_mat_vec_f<BF16,float,ncols_dst=5,block_size=160/256>`（`kernel_sm70_bf16.cuh:5-8` 记录）：
- 12µs 级 = `[2048×2560]nc5`（wq 6 卡分片；WIRING.md 二战表 44.5%）
- 59µs 级 = 大 K 矩阵族（wqkv 10240 行 / hc 系 K=10240；分片后 ~14MB 权重流）

## 3. 依赖波次分解（为什么是"每 wave 一次 launch"而不是"整层一次"）

一层的投影是**依赖链**而非集合：hc_up 需要 silu(down(xn))；wqkv 需要 hc mix 完成；ssm_out 需要 GDN 核；shexp down 需要 gate/up。单次 launch 覆盖整层必须引入 device 侧跨 block 依赖（Luce megakernel 方向），见 §4/§6-F2 的拒绝理由。

数据依赖 + **同 y** 两个约束把一层切成 8 个 wave（每 wave 内所有矩阵读同一激活、彼此无生产者/消费者边，且同 K——这是结构使然，非巧合：同 y ⇒ 同 K）：

| wave | 成员（GDN） | 共享 y 的 K | 6 卡分片后行数 | 逐个 launch 数 → 融合后 |
|------|------------|------------|---------------|------------------------|
| W1 | hc_attn_down + hc_attn_inject | 10240 | 54 + 1 | 2 → 1 |
| W2 | hc_attn_up | 320 | 1707 | 1 → 1（= 单 kernel 原样） |
| W3 | wqkv + wqkv_gate + beta + alpha | 2560 | 1707+1024+8+8 | 4 → 1（**主力**，~14MB 权重流） |
| W4 | ssm_out | 6144 | 427 | 1 → 1 |
| W5/W6 | hc_ffn_down+inject / up | 同 W1/W2 | 同 | 2→1, 1→1 |
| W7 | router + up_sh + gate_sh + gate_inp_sh | 2560 | 86+107+107+1 | 4 → 1（全小矩阵，launch/尾巴占比最大） |
| W8 | down_shexp | 640 | 427 | 1 → 1 |

QSA 的 W3' 有 5 个成员（wq+wk+wv+idx_q+idx_k，2048+86+86+86+22 行）。

**每层 16-17 次 launch → 8 次。** y 复用（任务书"广播一次"）在 v1 的体现：同一 wave 内所有 block 经同一 `__ldg` 模式读同一 y，首 block 触发 L1/L2 常驻后，小矩阵 block 全部命中缓存——而不是每矩阵一个冷 launch 各自重读。

## 4. Luce 摘录（读了什么、抄了什么、没抄什么）

1. **`server/deps/llama.cpp/ggml/src/ggml-cuda/moe-fused.cu`**（标量 IQ2_XS/IQ3_XXS 专家点积，一次 launch 处理全部选中专家）：
   - 结构：`moe_fused_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, smem, stream>>>`（:1074）——**grid=1 个 block**，块内 for 循环吃掉 `n_expert_used` 个专家的 gate/up 点积，中间结果 `silu(g)*u` 存 `extern __shared__ gu_smem[]`（:95, :189），阶段二 down 点积直接读 smem（:216 起），最后加权求和写回。
   - 教训采纳：**依赖阶段共用 launch 的手段 = shared 暂存 + 块内顺序执行**；跨专家的"任务表"就是 `expert_ids[] + expert_wts[]` 两个数组，循环认领。
   - 没抄：grid=1 是给小显存 gfx1151 + 小 ff_dim 的选择；对 V100/80 SM 会把带宽打回单 SM（~27GB/s），故本设计的 wave 内仍是数据并行 block 池。
2. **`server/src/common/moe_hybrid_storage.h:43-61` `CachedFfnGraph`**：`{ctx, gf, gallocr, inp, ids, weights, output, n_hot, n_tokens}`——按 (layer, n_hot) 二维缓存整张 FFN 图，"Eliminates per-call graph rebuild + gallocr planning overhead"。采纳其**结构缓存**思想 → 本设计 `fused_proj_table`（每 (层, wave, 图形状) 一份设备任务表，`matches()` 指纹复验，重建仅发生在调度器换了 buffer 地址时）。
3. **`server/src/deepseek4/deepseek4_fused_verify.inc`**（整模型 fused graph）：按 (q, flush, padded-comp) 形状缓存整张前向图 + `LUCE_DS4_FUSED_VERIFY` 门控 + policy 结构体集中开关。采纳：开关集中、按形状缓存、**fused 是配置项而非无条件路径**（`deepseek4_backend.cpp:496`"without it…"的守卫措辞）。

## 5. fork 自带的先例（重要——为什么不直接开 `GGML_CUDA_MMVF_MULTI`）

fork 已有 `ggml-cuda/mmvf-multi.cu`（4 段上限、**单一共享 x**、K≤3076、T≤4、总行数≤2048、块内 warp 切 K 的延迟型体），由 `ggml-cuda.cu:4283 ggml_cuda_try_mmvf_multi` 在图遍历里拼段，环境变量 `GGML_CUDA_MMVF_MULTI` **默认关**，注释给出两个理由：实测与分开 launch 打平（22.2ms）、求和顺序变化会挪动 greedy 文本。结论：小打小闹（≤4 段、只覆盖同 x 的迷你组）不够——本设计把段上限提到 24、复用大形状已调优的 uint4 带宽体、按依赖 wave 组织，并且把 parity 风险写成硬门（§8）。

## 6. kernel 设计决策（`fused_projection_sm70.cuh` 头注释 F1-F10 的摘要）

- **F1 任务表 + block 认领**：`fused_proj_tasks`（SoA：w/dst/n_out/stride_w/stride_dst + 前缀行块偏移 `block0[0..n]`）。`grid.x = Σ ceil(n_out_i/ROWS)`；每 block 对 `block0[]` 二分（≤5 次一致 L1 命中，每 block 一次）定位 (矩阵 t, 行块)。一维展平让小矩阵的 block 回填大矩阵的尾巴——这正是融合收益的物理来源。
- **F2 不做跨 wave / 无 device 依赖**：不引入 spin/atomic（顺序不定 → greedy 文本漂移，mmvf_multi 的教训）；每 wave 一个普通 launch，CUDA Graph 直接捕获。
- **F3 双表形态**：dynamic（设备 buffer 指针，内容可换）与 static（表**按值**进 kernel 参数 = constant bank；CUDA 12.1+ 参数上限 32KB，24 任务表 ~1KB）。二者 body 共用；bench A/B 定夺。
- **F4 每 block 体逐字复用单矩阵 kernel**（uint4 权重流 + __ldg float4 y + ROWS 行 MLP + warp/两级归约）：大形状 61-90% 带宽的流水线不许回退，只消灭外围 launch/尾巴。ROWS/THREADS 与单 kernel chooser 一致时结果**逐位相同**。
- **F5 ROWS/THREADS 按 wave 选**：合法因为一次 launch = 一个 wave = 一个 K。
- **F6 y 共享**（§3 末段）。
- **F7 v1 无 K-split**：hc mix wave（54 行×K10240，~0.7 个 wave）欠填充是已知残留；K-split 要么 atomic（顺序不定，拒）要么两段式（吃掉融合收益），列为后续。
- **F8 零行任务丢弃**（6 卡分片可能分到 0 行的小矩阵）。
- **F9 全部硬性要求 host 侧校验**，任一不过 → 回退逐矩阵路径：k%8、stride_w%8、stride_y%4、w/y 16B 对齐、n_out≥1、全任务同 K、ncols 1..8、n_task≤24。
- **F10 launch 热路径零分配**：持久表 `build()` 在捕获外执行一次；`launch()` 只发 kernel。

## 7. 本地验证（无 nvcc 工作站上的两层证据）

1. **语法/类型**：`clang++ -fsyntax-only -std=c++17` 对 sed 去 launch 配置后的 `fused_projection_sm70.cuh` + `kernel_sm70_bf16.cuh` + `bench_fused.cu` 全量检查——**0 error**（`syntax_check_prelude.h` 提供 CUDA 表面 stub；模板经 host 分发表隐式实例化，kernel body 全被类型检查）。此检查抓到并修复了两个真 bug：宏延续行 `//注释\` 失效、`launch()` 引用 `cfg` 而非 `cfg_`。
2. **功能**：`test_host_emul.cpp` 用宿主线程仿真 CUDA 线程（thread_local threadIdx、真 barrier + shuffle 交换缓冲、`__shared__`→函数局部 static），执行**真实 kernel 体**：二分认领全索引 ×20、{R1,R2,R4}×{nc1,2,3}×{T64,128}、部分行块（n_out=5/ROWS=4）、strided y、多矩阵 wave——对串行 CPU 参考最大 relerr **6.3e-8**，全部通过。

复现命令（两条均已跑通）：
```bash
cd layer-fusion && mkdir -p /tmp/lf_check
sed -e 's/<<<[^>]*>>>/ /' -e 's|#include <cuda_runtime.h>|// stubbed|' \
    fused_projection_sm70.cuh > /tmp/lf_check/fused_projection_sm70.cuh
{ cat syntax_check_prelude.h; echo '#include "fused_projection_sm70.cuh"';
  echo '#include "kernel_sm70_bf16.cuh"';
  sed -e 's/<<<[^>]*>>>/ /' -e 's|#include "fused_projection_sm70.cuh"|// x|' \
      -e 's|#include "kernel_sm70_bf16.cuh"|// x|' bench_fused.cu; } > /tmp/lf_check/check.cpp
clang++ -fsyntax-only -std=c++17 -I/tmp/lf_check /tmp/lf_check/check.cpp
clang++ -O2 -std=c++17 -pthread \
    -DHPE_KERNEL_HEADER='"/tmp/lf_check/fused_projection_sm70.cuh"' \
    test_host_emul.cpp -o /tmp/test_host_emul && /tmp/test_host_emul
```

### 7.1 GPU 微基准（待 V100 主机跑）

```bash
nvcc -O3 -arch=sm_70 -I. -I../bf16-gemv bench_fused.cu -o bench_fused && ./bench_fused
```

对比四档：`single`（逐矩阵生产单 kernel）｜`dyn`（持久表，稳态零 H2D）｜`static`（按值参数表）｜`dyn+H2D`（每次上传表，量化为什么需要持久表）。输出每 wave 的 blocks 数/配置、层总和（µs、GB/s、相对 900GB/s 百分比、倍率）、launch 数（16→8）、CPU 参考 relerr（tol 2e-4）、fused vs single 逐位差。判读要点：W2/W4/W6/W8 单矩阵 wave 应与 single 打平（同体同配置，仅多 ~20 周期认领开销）；W3/W7 应显著赢（launch 数与尾巴同时消灭）；W1/W5 小赢（受 F7 限制）；若 static 慢于 dyn 超过 2% 则弃 static。

## 8. fork 接线路径（给接线代理）

1. **放置**：`fused_projection_sm70.cuh` → `ggml/src/ggml-cuda/fused-projection-sm70.cuh`（namespace 不动，无新依赖）。
2. **挂点**（复用 mmvf_multi 的成熟机制，`ggml-cuda.cu:4541` 图遍历处）：
   - 写 `ggml_cuda_try_fused_proj(cuda_ctx, cgraph, i)`，与 `ggml_cuda_try_mmvf_multi` 同构：从节点 i（BF16 MUL_MAT、非 ids、src[0] 是 OP_NONE 权重）向后扫窗口，收集 `src[1]` 同一 y 的同批 MUL_MAT（≤24 个；窗口内节点不得读写候选输出/y——直接复用现有 `ggml_cuda_ranges_overlap` 检查）。注意图序：`build_hc_mix` 里 inject 排在 down/up 之后（qwen4exp.cpp:421-422），窗口需 ≥ 该间距；W3 四成员天然相邻（771-776、2394、2401）。
   - 命中 → 填 `fused_proj_desc[]`，查每 (层,wave) 的 `fused_proj_table` 缓存（key = 节点指针组；`matches()` 校验地址未变，变了才 `build()`）→ `launch(y, ncols, stride_y, stream)`；后续节点塞进 `cuda_ctx->precomputed`（mmvf_multi 已有的跳过机制）。
   - 不命中 → 原 `ggml_cuda_mul_mat` 路径，无行为变化。
3. **开关**：`GGML_SM70_FUSED_PROJ`（默认 0；parity 过门后改 1），照 `GGML_CUDA_MMVF_MULTI` 模式；另加形状门：仅 `cc==70`、`ne11<=8`、`ids==nullptr`、无 fusion（mmvf 的 GLU 融合路径优先）。
4. **CUDA Graph 共存**：ggml-cuda 先无捕获跑一遍再捕获（现有流程），`fused_proj_table::build()`（cudaMalloc/Memcpy）发生在捕获前；捕获嵌入的 kernel 节点持稳定设备表指针，重放零额外节点（对比 mmvf_multi 也是单 kernel）。图变体驱逐/重建时表缓存随节点指针 key 一起失效，无悬挂。
5. **TP**：y 全卡同读、dst 按卡行分片——每卡各自建表（n_out=本卡行数），零行任务由 F8 自动剔除。
6. **parity 门**（上线前，照 H/B 项纪律）：512-ID 输出一致性（容差 2e-4 相对）；加测 greedy 文本不漂移（mmvf_multi 注释点名的风险：ROWS 归约序不同于 fork mmvf——W3 里 beta/alpha 从 R1 变 R4 分块）。挂了 → 开关一键回退。
7. **二期**（不在本范围）：MUL_MAT_ID 专家组融合；W1/W5 的两段式确定性 K-split；Luce 式跨 wave megakernel（需 cooperative launch，sm_70 无 PDL）。

## 9. 预期收益算术与风险

**Launch 开销**：每 token 每卡 BF16 launch 780（36×16 + 12×17）→ 384，省 ~396 个。
- 无图模式：ppc64le cudaLaunchKernel ~1.5-3µs → 省 0.6-1.2ms/token；decode 23.2ms/步 → **~2.5-5%**。
- 有图模式（现状）：图节点 GPU 侧调度 ~0.3-0.5µs → 省 0.12-0.2ms → **~0.5-1%**。
**尾巴回填**（融合的本体收益）：小矩阵 kernel（W7 全组、W3 的 beta/alpha、W1）各 2-5µs 且 <45% 带宽；并入大 wave 后其 block 填充 SM 空洞。以 BF16 家族 33% kernel 时间 × 小形状占比 ~40% × 改善 30-50% ≈ kernel 时间省 4-6.6%；按 GPU busy 65% 折算步时 → **~2.5-4%**。
**合计诚实估计**：图模式 +3-6%，无图模式 +5-9%（叠加在单 kernel 优化之上；两项独立可乘）。与 WIRING.md 的单 kernel +5-8% 合计仍远离 launch/SNCCL 主瓶颈，符合"多刀累积"的既定路线。

风险与对策：
- W1/W5 欠填充（F7）→ 至多不赚，不会回退（launch 只减不增）；
- 求和顺序变化挪 greedy 文本 → parity 门 + 默认关；
- 图缓存驱逐频繁 → 表缓存 key 随节点指针，驱逐即失效重建，无正确性风险；
- static 表 local-spill 意外 → bench A/B 数据裁决（§7.1 判读）。

---

## 10. V100 实测（2026-10-03 深夜，修复后首跑）

**修复记录**：`fused_proj_find_task` 的 `__ldg` 用于 static 表（kernel 参数空间）触发
"operation not supported on global/shared address space" sticky error，毒死后续所有
build（bench 首跑 8 wave 中 7 个报 FAILED 的根因）。已改普通读（该处 __ldg 本无收益）。
**教训入档：kernel 参数空间指针禁用 __ldg。**

### 层级总结（single → dyn/static，µs/层）

| 层型 | ncols | single | dyn | static | 加速 |
|---|---|---|---|---|---|
| GDN | 1 | 85.8 | 64.2 | **60.9** | **1.4×** |
| GDN | 5 | 132.7 | 95.2 | **90.1** | **1.5×** |
| QSA | 1 | 81.9 | 57.2 | **54.7** | **1.5×** |
| QSA | 5 | ~同比例 | — | — | 1.4-1.5× |

### 关键 wave（GDN nc=1）
- **W7（router+shexp 4 小矩阵）：15.4 → 5.2µs = -66%**（nc5：-64%）——小碎片吸收的标本 case
- W3（主力 4 矩阵 14MB 流）：30.9 → 23.2µs = -25%（nc5：-39%）
- W1/W5（K=10240 双矩阵）：9.5 → 5.4µs = -43%
- W2/4/6/8（单矩阵 wave）：+7~13% 微亏——**接线策略：仅融合 n_task≥2 的 wave**（单矩阵
  融合只付认领开销无收益，直接走单 kernel 原路径），即每层 16-17 launch → 6 而非 8
- static 恒优于 dyn 1-4% → 生产选 static 形态
- 正确性：relerr 0.0（vs CPU）、fused vs single 逐位差 0.0（同配置）

### 折算全模型（诚实版）
BF16 投影组占 kernel 时间 33%、kernel 占步时 ~65%：0.33 × (1 - 1/1.45) × 0.65 ≈ **+7~10% 步时**
（叠加在已落袋的单 kernel +8.9% 之上，两项近似独立可乘）。
