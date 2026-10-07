# 1Cat-vLLM 上游深读侦察报告

- 日期：2026-10-06（纯只读侦察：GitHub `gh api` 读 PR/issue + 本地 clone fetch 上游 main 到 FETCH_HEAD，零服务器访问、零文件修改）
- 上游快照：`1CatAI/1Cat-vLLM` main（fetch 到本地 FETCH_HEAD，HEAD 含 PR #1001，上游 10-06 14:07 仍在推送）
- 本地 clone 状态：`Sak-114514/super-ac922`（= 上游 fork）分支 `ac922-ppc64le` = 上游 `db292f9a`（PR #703，**2026-09-27**）+ 我们 5 个 AC922 commit。即**缺 09-27 → 10-06 整整一波（#704–#1001，含全部 PLE cascade / GGUF 专家 kernel / GDN copy 融合 / HC LL 交换）**
- 仓库动向：README 首段宣布**计划改名 1Cat-LLM**；headline 已换成 **4× Tesla V100 16GB · Qwen3.8-27B-NVFP4 + DFlash2 ≈260 tok/s**（Bilibili 实机 demo），并强调 v1.5.1 release（10-03）数据是 32GB 机器、与 16GB demo 分属不同契约
- 重要更正：README 里 GGUF Flash-Next 的 92.3/280.4 只是 #940 时点（10-04）的数字；#1000/#1001（10-05 晚）后 **C1 单流已推进到 ~18.5ms/round（≈148 tok/s 契约）**，graph parity 文档另有 C1 24.1ms / C4 49.7ms 的自然负载口径。所有数字必须带契约引用

---

## 1. 多流聚合 280 的调度架构（PR #940 一族）

### 1.1 PR/文件级证据

| 证据 | 位置 |
|---|---|
| C4=280.441 的定义与复现命令 | PR #940（merged 10-04）；`docs/design/sm70_flashnext_gguf_mtp4_operator_routes.md`（FETCH_HEAD 版 L9-L60、L246-L278 复现命令块） |
| width 队列与 FULL 图校验 | `benchmarks/benchmark_gguf_model.py` L206-207：`query_len = 5 if args.mtp_draft else 1; required = {width * query_len ...}`；L268-291 每宽度构造 `width` 个独立 prompt 的 cohort，`generate_cohort(..., atomic=True)` |
| 引擎配置 | 同文档 L23-L31：max len 8704、**4 个 sequence slot**、batch budget 512、util 0.9、ring auto；C1=I8192/O256，**C4=每请求 I128/O1024 ×4，ignore_eos** |
| MTP draft 模型 | benchmark L120-126：`speculative_config method=mtp`、FP16 MTP4、`draft_sample_method=greedy` |
| per-slot GDN 状态 | `vllm/model_executor/models/qwen3_next.py` L1196-1198（强制 `--mamba-cache-mode=align`）、L1255-1272（state shape 计算含 `num_spec`）；`vllm/v1/worker/gpu/mamba_align.py` L57 注释 "Temporal state chooses the accepted speculative column"（SD-layout block table，验证后按接受列提交状态） |
| TP 拓扑与通信 | operator routes L18-L21：direct NVLink 0-1/0-2/1-3/2-3，对角 SYS；PR #992（HC 分片 + 两 launch NVLink LL 直连交换）、PR #957（融合 allreduce norm）、PR #1001 trace：rank0 每轮 21.04ms 中 target 图 17.13ms，draft 恒 210 kernels/round |
| 显存账本 | operator routes L193-L196：**模型 24.18 GiB/rank**（32GB 卡、util 0.9）→ KV 2.75 GiB（69,360 tokens，FP16 KV + FP32 SSM，page 816）+ 图 0.38 GiB |
| 并发扩展旁证（NVFP4 路线） | `docs/design/sm70_qwen38_concurrency40.md`：no-MTP C1 97.4 / C2 149.2 / C4 265.9 / C8 424.4 / C16 604.6 tok/s（batch fastpath +1.03GiB/rank 打包） |

### 1.2 技术机制拆解

**C4 不是"四个实例"，是一个引擎的四路批处理投机解码：**

1. **MTP4 共享方式**：单一 FP16 MTP4 draft 权重（0.79 GiB 量级，单层），对整个 running batch 一次前向（C4 时 draft 每步 M=4 序列并行；C1 时 step0 M5、后续 M1——见 `sm70_flash_next_mtp4_batch_gdn.md` L18-19）。draft 与 target 共享 packed 词表头（operator routes L88-91）。
2. **verify 窗口批法**：target verifier 是**一张 FULL CUDA graph，token 数 = width×5**（M=5/10/15/20 对应 width 1/2/3/4）。四条序列的 4 个 draft token + 1 bonus 在一次 M20 前向里并行验证（含 GDN 递推和 QSA attention 都按 20 token 批处理）。
3. **PLE/KV 显存**：权重 TP4 分掉大头（24.18 GiB/rank）；KV+SSM 仅 2.75 GiB/rank（4 条 ≤8.7K 上下文足够）；PLE 表完全不在 GPU 上（见第 3 节——pinned host 分片 7.2GB/rank + 磁盘）。**32GB×4 vs 16GB×6 的差别全部落在权重侧**：他们的 dual-bank（原字节块 + canonical 块并存）在 16GB 卡不可复制。
4. **continuous batching 与 GDN/QSA per-slot state**：标准 vLLM V1 调度 + `mamba-cache-mode=align`。混合状态存储为 SD-layout block table（每请求每 spec 位置一列 temporal state），verify 后由 `mamba_align.py` 的 Triton kernel 把**接受位置对应的状态列**提交为该 slot 的新状态（temporal 16 tile 切块拷贝、conv 状态块内 memmove）。QSA 的 KV 走常规 paged KV，E4M3 缩放是另一套 calibrated 通道（issue #746 时间线）。
5. **效率数字**：C4 52.06ms vs C1 26.07ms——2× 时间发 4× token；每流 70.1 tok/s（相对单流 92.3 损失 24%，来自 M20 verify 的 expert 激活集放大 + KV 扫描 + HC/QSA 批开销）。**280.4 是高接受合成负载**（重复文本 prompt，3.65 token/请求/轮）；自然/新负载 C4 聚合回落到 ~217-227 tok/s（#1000：C4 43.85ms × 9.43 token/轮；graph parity：49.7ms × ~11.3）。

### 1.3 对我们 75 tok/s 瓶颈的对照判决

我们的 75 = **两个独立 llama.cpp 单流进程互相踩**（125→聚合 75，每流 37.5）。他们既没绕过也没"解决 GPU MTP 链单序列限制——**他们的架构里这个问题不存在**：draft/verify/state 全部按 slot 批化，MTP 链的 GPU 利用率随并发上升而变好（C4 每流仍有单流的 76%）。

**可搬性判决：**
- **llama.cpp 线：架构不可搬，结论可搬。** 把 llama.cpp 改成 continuous batching + per-slot GDN state + 批量 verify 图 ≈ 重写调度器和图执行器（大型工程，不值）。但结论"多流必须共享一次前向而不是多实例共享 GPU"直接适用。我们 6×16GB 有第三条路他们没有：**实例分卡**——3×TP2 或 2×TP3 独占部署，零干扰，聚合 = N×单流。改造量：0 代码，1-2 天基准测试（TP2/TP3 的 llama.cpp V100 单流吞吐是关键未知数）。
- **我们 vLLM fork 线（super-ac922）：直接抄。** 我们的 27B AWQ TP3+native MTP4 生产（4×50k 并发 116.3 tok/s）用的就是同一套 align/mamba 状态机制（fork base #703 已含）。上游 10 月这波对多路并发的增量（HC LL 交换 #992、融合 allreduce norm #957、GDN copy 融合 #927/928/929/936、QPN 联合投影 #998）**全部直接 merge 即得**。注意 TP3 ≠ TP4：#992 的 LL 直连交换按 4 卡 ring 设计，6 卡/3 卡拓扑需要适配（我们对角 SYS 拓扑同他们）。

---

## 2. IQ4/IQ3 专家 integer dots 校准法（PR #998-1001）

### 2.1 PR/文件级证据

| PR | 实质 | 关键文件 |
|---|---|---|
| #1000（核心，10-05 22:00 merged） | Flash-Next 最后一个 expert 层（IQ4_XS gate/up + IQ4_NL down）改 integer dots | `csrc/sm70_turbomind/ops/gguf_dp4a.cuh`、`gguf_dp4a_sm70.cu`、`docs/design/gguf_lut4_q8_experts.md`、`vllm/model_executor/layers/quantization/gguf_turbomind_moe.py` |
| #998 | 27B 的 Q/K/V 联合 QPN 单 launch（14 种逻辑类型组合，M8/N3584/K5120） | `csrc/sm70_turbomind/ops/gguf_qkvz_sm70.cu`、`docs/design/gguf_qpn_attention_inputs.md` |
| #999 | **负结果记录**：IQ4_XS/Q4_K canonical-stream 筛选全输给 canonical 执行，不 admit | `docs/design/gguf_qpn_canonical_streams.md` |
| #1001 | 集成报告 + ledger（C1 19.5/20.6ms、6.7-6.9ms/output token） | `docs/design/gguf_qpn_output_batch.md` |

kernel 代码位置：`gguf_dp4a.cuh` L38 `LatticeDot<Type>`（type21 IQ3_S 等）、L131 `CanonicalIntegerDot<Type>`（type20 IQ4_NL / type42 Q2_0）；`gguf_dp4a_sm70.cu` L16 `quantize_q8`、L24 `gate_up`（template Type/Activated/Index/Lanes）、L198 `down_unroute`、L102-136 lanes 4/8/16 派发。

### 2.2 技术机制拆解

- **canonical weights 生成**：不是再量化。IQ1/IQ2/IQ3 族转成 TurboMind 2-bit operand carrier 包（index+sign，嵌套 scale 展开成 FP16；`gguf_turbomind_lattice.md` 存储表——代价是**比原 GGUF 块更占显存**）；IQ4_NL 复用共享 TurboMind LUT（`Transform_HMIA_SM70_Lut4<0>::iq_values`，`^0x80808080` 转有符号）；IQ4_XS 经 "canonical scale expansion" 复用同一 reader；Q2_0(type42) 直接 2-bit 展开 + `__vsub4` 居中。
- **integer dots**：激活先 warp 级量化成标准 **Q8_1**（32×int8 + FP16 d + FP16 sum，36 字节/块——与 llama.cpp `vecdotq.cuh`/`quantize.cu` 公式完全一致，文件头 L3 明注 "Integer-dot formulas and Q8_1 layout follow ggml-org/llama.cpp"）。`__dp4a` 整数点积、int 累加，最后 `float(sum) × (weight_scale × x.ds)`（`gguf_dp4a.cuh` L160-167）。无 split-K——小 M 下单 CTA 内完成，FP32 MMA 的 split-K 只在 canonical pair 路线（`gguf_iq3_gated_pair_dispatch.md` L8-10）。
- **"calibrated" 的真实含义**：**运营流程词，不是精度补偿**。指三重准入：①数值 oracle 校准（对 official GGUF 反量化 + 独立 oracle，relL2 门槛）；②同钟（1290/877MHz 锁频）cold-L2 ABBA 计时筛选；③只 admit 实测过的 `(格式, M, TP 几何)` 组合，其余 M（如自然生成遇到的 M25）bitwise 回退 canonical GEMM。能力框架字段 `sm70_gguf.lut4_expert_dp4a` 隔离 A/B。
- **性能**（真实末层 bank，512 experts/hidden 2560/local N160/top-k10，含激活编码→gate/up+SiLU→routed Q8→down/unroute 全链）：M5 223-229→**70.7us（3.2×）**，M20 308-328→**199.7us（1.6×）**；有效唯一权重带宽 479/564 GB/s；relL2 vs canonical 0.0106/0.0110（Q8 激活舍入是显式授权的）。质量门：64 前缀 teacher KL 0.000716、top1 63/64、接受率 +0.5pp（CI [-1.4,+2.7]）。整模型 C1 均值 18.996→18.534ms（中位增益 0.09-0.18ms——单层而已）。
- **lane 筛选**：4 lane 默认（全模型+质量双验证），8 lane 快 ~4us 但未晋级，16 lane 回归（`gguf_lut4_q8_experts.md` 分屏表）。

### 2.3 与我们 llama.cpp 反量化路径对比 + "18/40 层 48%" 真相

- llama.cpp 的 `GGML_CUDA_MOE_GROUPED`（MMQ/MMVQ）**本来就是 dp4a 整数点积**——他们引用的公式源头就是我们已有的路径。真正差在：①融合度（他们 quantize→gate/up→SiLU→routed Q8 编码→down/unroute 一条链，读路由 ID 免 gather）；②CUDA graph 消 launch；③per-M lane schedule；④共享激活（gate/up 读一次 A）。这些是 PR #29184（shared-expert fusion）同方向的深化。
- **更正背景数据**："18/40 层 48.463%" 出自 `gguf_qwen38_iq3s_source_inventory.md` L70，是 **27B dense（64 层、40 个 mixed gate/up 对）的阶段性快照**（前两格式集 IQ3_S+IQ4_XS / IQ3_S+IQ3_XXS）。最新状态（`gguf_qwen38_mixed_pair_coverage.md`）：19 个 orientation 全部 admit，**40/40 mixed 已覆盖 + 8 纯 IQ3_S = 48/64 层**；剩 12 纯 IQ3_XXS + 4 纯 IQ4_XS 仍 canonical。**Flash-Next 侧**：IQ4 integer dots 只覆盖**最后 1 个 expert 层**；其余 17 层 IQ3_XXS(type18) gate/up 仍走 canonical 向量（每 target 轮 34 次 Launch），IQ2_S gate/up 和 down M20 点位因更慢被排除。
- **剩余难点本质**：小 M 下 per-format integer reader 干不过已经高度优化的 canonical MMVQ/lattice 路线（#999 整个 PR 就是记录这个负结果：IQ4_XS down 40.96 vs 38.91-39.94us canonical）。格式越多、每格式层数越少，逐格式 kernel 的边际收益越薄——这是他们"48% 之后推进变慢"的原因。

### 2.4 可搬性判决

- **llama.cpp 线：理念可搬、代码不可直接抄。** 我们已在 integer dots 上；可搬增量是 (a) gate/up 共享激活 + 融合 SiLU×up + 中间 Q8 重编码（省一次反量化往返），(b) routed 直接读 ID 免 gather。改造量：中（1-2 个 CUDA kernel 改造 + MMQ 派发点），Codex 可做。预期收益参考他们 M5 链 3.2×——但我们 llama.cpp 单流瓶颈未必在专家（先 profile 再动）。
- **vLLM fork 线：直接 merge 上游。** #998-1001 全在 main。
- **16GB 红线**：canonical 存储 > 原 GGUF 块（lattice 文档明说 "occupy more memory than original GGUF blocks"），他们的 dual-bank（24.18 GiB/rank）在 16GB 卡必然砍成单 bank——上游代码有 capability 框架和显式 fallback 路径，但 16GB 配置他们从没测过（所有 GGUF 验证都在 32GB SXM2 上），**这是我们要自己踩的部分**。

---

## 3. PLE 三级放置（PR #806 / #877 / #885）

### 3.1 PR/文件级证据

| PR | 内容 | 
|---|---|
| #806（10-03） | cascade 能力准入框架（整合 #646/#717 的 shard planner + 量化字节 gather）：device → pinned host → file-backed 三层，每行只属一层；`KernelConfig.ple_disk_cascade` 默认 true（`vllm/config/kernel.py` L544） |
| #877（10-04） | GGUF packed IQ4_NL 行表过 offloader：mmap 保留 n-gram 存储、只解码选中行、修复 disk-only placement 的 row-zero 指针 bug |
| #885（10-05） | CPU 端磁盘行读取优化：批量 madvise 预取 + 批拷贝，单 token 查找省 0.0726ms；cold-page 每查找 4.26% major fault |

代码位置：`vllm/model_executor/kernels/ple/disk_rows.py`（`MappedRowGatherKernel` → `torch.ops._C.ple_disk_gather_u8`，`csrc/ple_disk_rows.cpp`）、`ple/gguf_pinned.py`（`pinned_table_capability` L41+，TP 分片 + NUMA 亲和分配 L26-59）、`ple/host_result.py`、`vllm/model_executor/layers/ple_offload_layer.py`（跨进程 transport：`CpuGpuSemaphore` 用 `cuStreamWaitValue32/WriteValue32` 让 **CUDA graph 内原地等待 offloader 结果**，L5-20 时间线图）、offload worker 进程注册在 `vllm/models/qwen4_exp/nvidia/ple_layer.py`。环境变量 `VLLM_QWEN4EXP_PLE_HOST_GIB`（`vllm/envs.py` L952）。

### 3.2 技术机制拆解

- **表的真实尺寸（我们背景里 "26.8GB" 的出处）**：GGUF IQ4_NL packed 行表 320,001,536 行 × 160 宽 = **28,800,138,240 字节 ≈ 26.8 GiB**；全量反量化会膨胀到 ~102.4GB，所以永不全解（`gguf_packed_ple.md` L5-7）。NVFP4 检查点版本是 FP8 E4M3 全表 47.68 GiB（TP4 = 11.92 GiB/rank pinned，`sm70_qwen38_flash_next_nvfp4.md` L105-106）。
- **分层判定**：加载期（构建模型前）解析元数据与 host 容量：可用 host 内存 − reserve 按 **fair share 分给本机 local workers**；显式 `VLLM_QWEN4EXP_PLE_HOST_GIB` 优先；已有的显式 hybrid/disk-only/全表放置再优先。GGUF pinned decode 路线额外要求：SM70、FP16 输出、**local TP4**、单 IQ4_NL 表、**≤4 个调度请求**、Model Runner V2 + 双编译 + FULL decode 图（`gguf_pinned_ple_decode.md` L17-30）。
- **per-rank 放法**：不是 per-head 分卡，是**按行分片**——TP4 每卡 pinned 原始 packed 字节的 1/4 = 7,200,034,560 B/rank，无第二 bank（同文档 L3-5）。compute rank 直读本地下步行，等 offloader 的 raw 字节在图内合并；两路用同一 E4M3/IQ4_NL 反量化 + scale 后再做 TP 归约。
- **GGUF 路线的最新演进（关键）**：decode 时**不再走 offloader**——`ple_pinned_decode` 默认 true（`kernel.py` L559），一个 Triton launch 经 pinned UVA 直接 gather+解码（IQ4_NL 码本 + FP16 块 scale，数值与官方反量化逐值相等），M20 查找 ~25.4us、NUMA 本地化收益微小。**模型级收益：C1 22.50→18.62ms，C4 53.93→46.07ms**（同文档 L92-100 表）——是 10 月这波单项最大的端到端收益之一。prefill 仍走 offload worker（动态 prefill 需要全词表行）。
- **传输同步**：offloader 是独立 OS 进程；图内 `WaitValue32(flag==1)` 等待 pinned→GPU 异步 H2D 完成，请求绑定的输出 buffer + event 保并发归属；`ple_disk_release_pages` 可选 madvise 释放已读文件页。

### 3.3 对我们 16GB×6 / 32GB RAM 的可搬性判决

- **GPU 侧零成本**：表完全不住 GPU——16GB 显存不是障碍，这点对我们极友好。
- **真正的硬约束是 host RAM**：他们 host 动辄 160 GiB（issue #746 那台），全表 pinned 毫无压力。我们 AC922 **32GB RAM**：表 26.8 GiB，TP4 分片也要 4×6.7 GiB pinned = 26.8 GiB 全部 RAM——不可能。实际可 pin 总量 ~8-14 GiB（留 OS/KV CPU tier/页缓存），即 **1/3~1/2 表 pinned，其余留 mmap/disk**——这正是 cascade 三级设计的场景，#885 的批量 madvise 预取（4.26% major fault/查找 → 已缓解）就是为缺页代价准备的。
- **POWER9/NVLink2**：pinned UVA over NVLink2 理论可用（`cudaHostRegister` + UVA），POWER9 的对称 NVLink 反而是优势；但 upstream 从未在 ppc64le 测过，NUMA 亲和代码（`sched_setaffinity` + `/sys/devices/system/node`）在 POWER9 NUMA 拓扑上需要验证。
- **判决**：vLLM fork 线 = **直接抄（merge 上游）+ 参数改造**（TP6/TP3 下 `tp_size != 4` 的准入要放开、pin 份额调小）；llama.cpp 线 = **思想可搬**（"PLE 表 pinned 分片 + kernel 内 IQ4_NL 解码 gather，免 CPU 往返"），改造量中（一个 gather kernel + 表按 TP 分片加载），但受 32GB RAM 同样约束，且我们 llama.cpp 当前 PLE 查找成本未量化——**先 profile 再决定**。

---

## 4. 边缘侦察

### 4.1 "15-17ms round goal" 还开着——路线图在哪

- **目标定义**：`docs/design/sm70_flash_next_mtp4_15ms.md` L38-44——"optimization target remains **15 ms per complete round**"（含 draft+verify+target head/采样+交接+准备；分母是完整投机轮数）。这是 **NVFP4 路线**的契约；GGUF 路线在 `sm70_flashnext_gguf_mtp4.md` L4-5 定的是更激进的 **10-11ms 客观目标**（未达成、未测）。
- **没有跟踪 issue**——目标只活在 PR 正文和设计文档里（PR #940 "The 15–17 ms goal remains open"；`gguf_lut4_q8_experts.md` "18.5 ms phase target...ultimate 15 ms target"）。issue 列表里最近的性能讨论是 #852（PP=2 异步调度 -55% 单流）和 #880（mixed-prefill 250ms 默认卡长 prompt）。
- **现状与差距**：NVFP4 路线 fixed8k 21.2ms（15ms 文档实测表）；GGUF 路线最新 C1 均值 **18.53ms**（#1000）——**距 15ms 还差 ~3.5ms**。
- **剩余路线（从 trace 分类表推断，operator routes L150-172 + graph parity 文档）**：三大头是 HC combine+gates 3.58ms、TP ring（含就绪等待）3.42ms、bitplane 投影 3.03ms；#998 QPN 融合已砍 0.6ms；下一步候选：copy/cast 0.93ms、index/reduce/scatter 0.73ms、rank0 异步输出物化 skew（620us vs 其他 rank 246us）。52ms 的出处：C4 round 52.059ms（#940）——"52ms→15ms" 实为 **C4 52ms / C1 26ms→18.5ms（仍向 15ms 爬）** 的两条曲线。

### 4.2 Flash-Next NVFP4 路线与 16GB

- **80.7 / 138.3 的定位**：README L125-126——no-MTP 8K/512 纯 decode 80.732（#415）、MTP4 冷 JIT gate 138.26（#389）。这是**官方 NVFP4 检查点的早期（9 月中）数字**；v1.5.1 现役 NVFP4+DFlash2 生产线已到 32K 单流 201 / C4 445 聚合。NVFP4 是 **release 路线**（v1.5.1 wheel 可装），GGUF 是 **dev main 路线**（要源码构建）。
- **16GB 能放吗**：README 新 headline **4×V100-16GB 跑 Qwen3.8-27B-NVFP4+DFlash2 ≈260 tok/s**（实机 demo 视频）——27B NVFP4 权重总量 ~13-14GB，TP4 每卡 ~3.5GB，16GB 卡放得下且有余量给 KV。但 **Flash-Next NVFP4/GGUF 在 16GB 放不下**：NVFP4 版 target+MTP 23.16 GiB/rank（`sm70_qwen38_flash_next_nvfp4.md` L392）、GGUF 版 24.18 GiB/rank——都是 32GB 卡的账本。旧文档 `sm70_memory_defaults.md` L167 "No physical 16GB GPUs are available" 已过时，但 16GB 上**从未跑过 Flash-Next**。
- **对我们**：Flash-Next IQ3_XXS GGUF 全模型 ~11-12GB（llama.cpp 单卡 16GB 就能跑），TP 分片后每卡 <3GB——**权重从来不是我们的瓶颈，KV+SSM state 才是**；他们的 KV 建置（E4M3 KV + DCP2 + calibrated QSA scales，issue #746 时间线：FP16 KV 310K→E4M3 570K→DCP2 1.08M tokens）是我们 6 卡长上下文路线的最直接参考。
- NVFP4/GGUF 接受率 parity 已验证（graph parity 文档：45.47% vs 45.20%，长度 2.819 vs 2.808，20k 自助 bootstrap CI 含 0）——GGUF 路线的质量叙事成立。
- 相关开放 issue：#731（local-inference-lab 血统的 NVFP4 PLE 表不支持）、#865（SM70 MTP decode ≥2 并发死锁——**注意：这是 NVFP4+TP4 的 bug，说明 C4 并发路径仍有雷**）、#1003（检索正确性依赖 chunked-prefill 边界——GGUF 路线质量开放 bug）。

---

## 5. 综合建议（优先级排序）

### P0 —— vLLM fork 线：立即 sync 上游（改造成本最低、收益最大）
我们 fork 停在 #703（09-27），缺整波 #704-#1001。按端到端收益排序应带回：
1. **pinned PLE decode**（C1 -3.9ms / C4 -7.9ms，单项最大）+ PLE cascade 三级（#806/#877/#885）——正好命中我们 32GB RAM 约束场景；
2. **GDN copy 融合链**（#927/#928/#929/#936：projection-tail 单 launch、行跨步 recurrent input、直接 recurrent output，M5 共省 ~0.8ms）；
3. **HC 分片 + NVLink LL 直连交换**（#992）——注意 TP3/TP6 拓扑适配；
4. **IQ4/IQ3 专家 integer dots + QPN 联合投影**（#998-#1001 及其前置 #897/#960-#986 系列）。
改造量估计：merge + ppc64le 编译修复（我们已有 reproducible build 记录）+ TP3 适配（`tp_size != 4` 的 PLE/capability 准入放开，参考我们 TP3 head-replication 的既有 patch 手法）≈ **1-2 周量级**。风险：上游 10 月波次全部只在 32GB SXM2 验证，16GB 显存预算要逐项重审（dual-bank 必砍）。

### P1 —— llama.cpp 线（Codex 实验）：两个可直接落地的抄法
1. **多实例分卡部署**（第 1 节结论的应用）：6×16GB = 3×TP2 或 2×TP3 独占实例，聚合 = N×单流。零代码，先测 TP2/TP3 的 llama.cpp 单流吞吐。若 TP2 单流 ≥100 tok/s，3 实例聚合 ≥300——**直接超过他们 32GB×4 的 C4 280**（不同契约，仅作量级参照）。
2. **PLE pinned 直读 kernel**（第 3 节思想搬移）：先 profile 我们当前 PLE 查找成本（llama.cpp mmap 全表在 CPU 的每次查找开销），若 >0.5ms/轮再做：表 TP 分片 pinned + kernel 内 IQ4_NL 解码 gather。改造量中；32GB RAM 下只 pin 1/3-1/2 表 + madvise 预热。
3. （可选）专家 gate/up 共享激活融合（第 2 节）：在 profile 确认专家链是瓶颈后再投入。

### P2 —— 观察项（不建议现在跟进）
- 15ms round 目标的剩余 3.5ms：他们自己的下一步（copy/cast、rank0 输出物化）都是细活，等他们趟；
- NVFP4 27B 16GB 路线：与我们 Flash-Next 主线无关，且 27B dense 我们已有 AWQ 生产；
- GGUF 路线的 27B dense mixed-pair（48/64 层）：仅当我们切 27B GGUF 时才有意义。

### 明确不建议
- 在 llama.cpp 上复刻 continuous batching + per-slot GDN state + 批量 verify：等于重写调度器，且我们 vLLM fork 已有同等能力——双线重复。
- 16GB 卡上照搬 dual-bank canonical 存储：显存预算不支持，走单 bank + fallback。

### 关键不确定性（下一步侦察/实验点）
1. llama.cpp TP2/TP3 在 V100 的单流吞吐（决定 P1-1 上限）；
2. 上游 10 月波次在 ppc64le 的可编译性（`gguf_dp4a_sm70.cu` 等 SM70 扩展用 `__dp4a`/PTX，ppc64le 交叉编译应无碍但未验证）；
3. POWER9 NUMA 上 pinned UVA gather 的实际带宽（他们 x86 NUMA 数据 ~25us/M20 只能作下界参考）；
4. 我们 32GB RAM 里 OS+KV CPU tier+页缓存的实际可用 pin 份额（决定 PLE pin 比例）。
