# 设计 1 裁决报告：fattn workspace 减容流式化（UB1024/2048 解锁）——前置侦察推翻前提，判死

- 日期：2026-10-06（夜）· 执行：ZCode（fattn-stream 线）
- 任务：按《onecat-prefill-pr-deep-dive.md》B 部分设计 1（P0），先零代码前置侦察（UB 斜率外推），仅当斜率为正才实施流式化
- 领地遵守：生产树零修改（本任务全程未改任何源码）；Codex 五微测/upstream 领地只读；GPU 使用经 wait-drain 礼让协议
- 远端工件：`ac922:~/ac922-llama-next/experiment/zcode-fattn-stream-20261006/`（驱动脚本 + 隔离 runner 副本）；`manifests/fattn-stream-ub{256,512,1024}/`

## TL;DR（裁决）

1. **设计 1 判死，不实施**。三个前提逐一被证据推翻：
   - **前提推翻 A**：UB1024 并没有被"挡死"——**132K ctx + Q8K/Q8V 下 UB1024 早已可跑且就是 Codex 现役速度候选**（8K 920.56 / 64K 1003.40，Codex REPORT.md 同轮组合表）。4824.31 MiB OOM 只发生在 **262K ctx + Q8K/Q4V** 容量配置。
   - **前提推翻 B**：4824.31 MiB **不是 fattn 的 split-K workspace（dst_tmp）**，而是 Q4V 使 QSA block-sparse kernel 失效后，**gather 回退路径的 K/V F16 gather 张量**（每个 2304.0 MiB @UB1024）撞爆 compute buffer（证据链见 §2）。
   - **前提推翻 C**：批深度斜率为**负且倒 U**。本轮新鲜三点同链 A/B（§4.2）：64K PP = UB256 687.37 / **UB512 745.21（峰）** / UB1024 706.50——UB1024 **-5.2%**、UB256 **-7.8%**；且与 prefill-parallel 会话昨晚独立测得的 -5.2%（648.84→615.74 / 746.45→707.86，parity ✅，其 REPORT 已给 ub1024 判死）**精确互证**。1Cat #754 的"批深度一阶变量"结论在我们栈上反向：**现役 UB512 已在最优批深度上**。
2. 唯一值得留下的 redirected 跟进项（P2，非本设计）：**qsa-attn.cu 支持 q4_0 V**（~60 行），消除 262K 容量配置的 gather 路径，回收 k8v4 配置 -4~5% 的 prefill 税。与 UB 无关，不构成本设计的救赎。

## 1. 阶段 0-1：UB1024 OOM 的确切条件（日志级澄清）

| 配置（全部 IQ3_XXS-MTP-Q8、岛 3,3、132K/262K） | UB | 结果 | 证据 |
|---|---|---|---|
| 132096 ctx + Q8K/Q8V | 1024 | **通过**（单卡最小空余 2281 MiB） | Codex REPORT "速度候选"行；mwe-off-al-r1/r2 今晚复跑通过（64K PP 595.04，probe 链） |
| 262144 ctx + Q8K/Q4V | 512 | 通过（采样最小空余 1783 MiB） | Codex REPORT 容量行 + kv-262k 臂 1（PP 624.6/707.6，生产链） |
| **262144 ctx + Q8K/Q4V** | **1024** | **首次 prefill 扩展工作区时 device3 cudaMalloc 4824.31 MiB OOM** | codex-upstream REPORT.md 容量与失败记录节；codex-five-micro 04-k8v4 PLAN.md："UB1024 failed during runtime workspace allocation (cudaMalloc request 4,824.31 MiB on CUDA device 3)" |

- 现役 UB 澄清：**生产基线 = UB512**（speed-bundle / kvtest 脚本默认），**Codex 速度候选组合 = UB1024@132K**（bundle-r19-u1024 名字里的 u1024 是真的；TECH-FACTS "UB512 现役（64K PP 1003）"表述有误——1003.40 那轮就是 UB1024）。
- 262K+UB1024 的 4824.31 MiB 请求发生在**首次请求扩展工作区**（启动加载与 graph 预留都成功，READY 后第一发 prefill 撞）。

## 2. 4824.31 MiB 的解剖（读码 + 显存账本闭环）

### 2.1 路径决策树（`src/models/qwen4exp.cpp`）

单序列 prefill（top_blk 路径，`build_attn_qsa_blk` L1882-1976）：

```
sparse_ok = LLAMA_QSA_SPARSE_KERNEL(=1) && n_tok > 16
         && K.type==Q8_0 && V.type==Q8_0 && D==256 && ratio==4
  ├─ true  (Q8/Q8 生产配置) → qsa-attn.cu block-sparse kernel（prefill 时 n_split=1，workspace=0）
  └─ false (V=q4_0，262K 容量配置) → gather 回退：
       K_gather = F16 [256, Hkv_local=2, 2304, n_tok]   ← GET_ROWS 整块 gather + cast F16
       V_gather = F16 [256, 2, 2304, n_tok]
       之后 build_attn_mha 按 n_tok 个 stream（每 stream 1 query）做 FA
```

- width=2304 cells 的来源：`indexer_top_k=2048`（GGUF `qwen4exp.attention.indexer.top_k u32 = 2048`），n_sel=2048/4=512 块，+tail 后 `GGML_PAD(2052,256)=2304`。
- 几何：GGUF `head_count=24 / head_count_kv=2`；岛 TP3 → 每卡 8 Q 头 + **2 KV 头全量复制**（K cache 实测 544 B/cell/layer = 2 头 × 256 D × q8_0 34/32 B），gqa=4。深挖报告 B1 表"岛内 8/4，gqa=2"有误，实为 8/2、gqa=4。

### 2.2 尺寸账本（每卡）

| 张量 | 公式 | @UB512 | @UB1024 |
|---|---|---:|---:|
| K gather F16 | 256×2×2304×UB×2B | 1152.0 MiB | 2304.0 MiB |
| V gather F16 | 同上 | 1152.0 MiB | 2304.0 MiB |
| 其余 UB 比例项（Q/out/mask/ids/索引器/投影激活等） | ≈0.21 MiB/token×UB | ~108 MiB | ~216 MiB |
| **首 prefill 工作区增长合计** | | **~2412 MiB** | **~4824 MiB** |

**闭环验证**：
- 4824.31 = 2412.155 × 2.0000（UB 翻倍、恰好全线性——失败的那笔分配 = UB1024 尺寸的整块 meta compute buffer）。
- 分配语义（ggml-backend-meta.cpp `ggml_backend_meta_buffer_type_alloc_buffer`）：meta buffer 的 `size` 会在岛内**每张成员卡上全额分配**（逻辑切分在 buffer 内部进行），所以"device3 申请 4824.31 MiB"= 整个岛 compute buffer 的全额请求，不是单卡份额。
- 增长方式 = 先释放旧 buffer 再申请新的更大 buffer。UB512@262K 首发时的可用空间 ≈ 16150 − 9377(模型) − 3594(KV) − ~600(上下文) + 1789(释放的旧 compute) ≈ 2579 ≥ 2412 → 通过（采样最小空余 1783 MiB 与账本自洽：实际驻留 ≈ 9377+3594+~800(触页部分)+ctx ≈ 14367 用量）。UB1024：同路径需 4824 > ~2579 → **device3 OOM，误差为零参数拟合**。
- 对照组：132K+Q8/Q8+UB1024 初始 compute 预留实测仅 1922.89/1930.14 MiB（mwe-off-al-r1 log），UB512 为 962.33/965.96（kvtest-arm0 log）——1.879 MiB/token 完美线性，**无任何 fattn 专属大头**。

### 2.3 fattn workspace 现状核对（对设计 1 原设想的逐项否证）

| 设计 1 的假设 | 实况 |
|---|---|
| "workspace 大头是 score 累加缓冲 dst_tmp ∝ parallel_blocks×M" | 生产 prefill 路径（qsa-attn sparse kernel）`n_split = max(1, min(64, n_sm*8/(n_tok*n_kvh)))` → UB≥256 时 blocks0=UB×2 ≥512 > 640/… → **n_split=1，dst/part 缓冲完全不分配**（qsa-attn.cu `if (n_split > 1)` 块） |
| "fattn-wmma case<256,256,32> 的 split-K 物化" | fork 中 D=256 prefill 的 fattn-common `launch_fattn` 路径仅在 **Q4V gather 回退后的 per-stream FA**（Q rows=1，TILE kernel）与 decode 小批量出现；dst_tmp = parallel_blocks×nelements(KQV) 在 Q rows=1 时 ≤ 几 MiB，不是大头 |
| "score workspace ∝ UB×ctx 物化" | 不存在此物化。真正的 UB×KV 尺寸项是 **Q4V 回退的 gather 张量（UB×2304 cells）与 masked 路径的全长 F16 mask（UB×n_cells）**，都属于 QSA 调度层产物，不是 fattn kernel workspace |
| "132K 下 UB1024 需要流式化才能装下" | 已装下且在生产组合中运行（§1）；UB2048 的障碍是**普通激活 compute buffer 全线性涨到 ~3.8 GiB**（962→1923→3847 外推），与 fattn 无关 |

## 3. UB2048@132K 的可行性外推（顺带回答设计 1 的"2048"半边）

- 激活 compute buffer 斜率 1.879 MiB/token → UB2048 ≈ 3847 MiB。
- island-B 卡：9377（权重）+ 2198（KV@132K q8/q8）+ 3847 + ~700（上下文/杂项）≈ 16122 MiB vs 16150 可用 → **空闲 ~28 MiB，必炸**（还有 CUDA graph 等未计）。
- 结论：UB2048 需要的是**全模型激活层减容**（F32→F16 中间态/图调度分块），那是另一个量级的工程；即便做出来，§4 的斜率说明它也没有收益可兑。

## 4. 阶段 0-3：UB 斜率实测（零代码，现役二进制）

### 4.1 已有同链 A/B（prefill-parallel 会话 2026-10-06，生产链，parity ✅）

| UB | 8K PP | Δ | 64K PP | Δ | 64K TG |
|---|---:|---|---:|---|---:|
| 512 | 648.84 | — | 746.45 | — | 83.07 |
| 1024 | 615.74 | **-5.1%** | 707.86 | **-5.2%** | 83.07 |

该会话 REPORT.md 结论原文："ub1024 计算侧损失大于通信侧收益。均判死。"

### 4.2 本轮补测（三点斜率；生产 env、kvtest 协议、当前二进制 557b422+mwe 惰性断言、q8/q8、132K ctx、matrix 遥测 + 512-ID parity）

时间 2026-10-06 23:35-23:48，各 1 轮，串行，臂间 wait-drain（远端 `experiment/zcode-fattn-stream-20261006/ub_slope2.sh` + 隔离单元 ac922-fattnstream-test）：

| UB | 8K PP | 64K PP | TG 8K / 64K | compute buffer (MiB) | parity |
|---|---:|---:|---:|---:|---|
| 256 | **667.45** | 687.37 | 93.80 / 83.33 | 552.6 / 554.4 | ✅ 逐位 |
| 512 | 632.51 | **745.21** | 92.34 / 83.10 | 962.3 / 966.0 | ✅ 逐位 |
| 1024 | 607.98 | 706.50 | 92.88 / 83.18 | 1922.9 / 1930.1 | ✅ 逐位 |

- **64K（设计 1 的主战场）：峰值在 UB512，两侧均降**——UB256 **-7.8%**、UB1024 **-5.2%**（与 prefill-parallel 的 -5.2% 精确复现，两条独立会话×两晚互证）。批深度曲线是**倒 U 形，峰在 ~512**，不是 1Cat 式的单调上升。
- **8K：随 UB 单调下降**——UB256 比 UB512 **+5.5%**（次要发现：短 prompt 偏好小 ubatch；混合负载仍应取 512，按 prompt 长度切 UB 策略是零代码 P3 观察项）。
- TG 与 UB 无关（83.1-83.3 / 92-94 持平），UB 只影响 prefill。
- 噪声口径：本session 8K 绝对值比同链早间运行低 ~2.5%（648.8→632.5，Codex 五微测并行 CPU 编译的宿主争用），64K 与历史吻合 ±0.2%（745.2 vs 746.4）——臂间相对比较（同条件串行）不受影响。
- compute buffer 同链斜率：552.6 → 962.3 → 1922.9 MiB（≥512 段 1.88 MiB/token 全线性，与 §2.2/§3 外推一致）。

### 4.3 交叉参考

- mwe-off（UB1024，probe 链，今晚）：r1 596.60/595.04、r2 593.5/593.9 —— 与生产链 UB1024 同向偏低。
- Codex 岛链 UB1024（真实岛/P2P+R19）：913.12-920.56 / 1003-1006；**该链从未测过 UB512**（graph_compute host 窗口 13.434ms@UB512 < 14.554ms@UB1024 暗示 UB512 或更快）。若 Codex 后续回补 UB512 A/B 且重现 -5% 量级，其速度候选还有免费上行空间——此为给 Codex 线的备忘，不在本任务范围。

## 5. 裁决与后续

- **设计 1（fattn workspace 减容流式化解锁 UB1024/2048）：判死。**
  - 死因 1（收益侧）：批深度斜率为负且倒 U（§4）——64K 峰值就在现役 UB512，UB1024 -5.2%（双会话互证）、UB256 -7.8%；解锁更大的 UB 无收益、有实亏。UB2048 即便装得下也只会更慢。
  - 死因 2（问题侧）：132K 生产配置不存在 fattn workspace 瓶颈（qsa-attn sparse kernel prefill 零 workspace；UB1024 compute buffer 仅 1923 MiB）；262K+Q4V 的 4824 MiB 是 QSA 调度层 gather 回退产物，最小修法是 q4_0-V sparse kernel 支持而非流式化。
  - 死因 3（UB2048）：障碍是全模型激活缓冲（~3.8 GiB 外推，island-B 卡空闲仅 ~28 MiB），非 fattn，且无收益可兑。
- **Redirected P2 建议**（记录，不实施）：`qsa-attn.cu` 增加 q4_0 V 反量化模板（qsa_load_raw 换 q4_0 解码 + supported() 放宽 V 类型 + 环境变量门控），262K 容量配置 prefill 可从 624.6 回到 ~650 量级（回收 k8v4 的 -4~5% 税）。parity 门槛：q4 解码路径 vs CPU 参照逐位（同族于 kv-262k 已验证的 q4 V 无损结论）。
- **P3 观察（零代码）**：8K prefill 在 UB256 快 +5.5%——若业务以短 prompt 为主，`AC922_TEST_UBATCH=256` 是免费的短窗提速旋钮；按 prompt 长度自适应 UB 的调度属后续课题。
- 深挖报告勘误建议：B1 表"岛内 TP3 8/4 gqa=2"应为"8/2、gqa=4（KV 岛内复制）"；B3 设计 1 的问题定义与前提 A/B/C 按本报告 §1-§3 修正。

## 6. 过程记录（复现用）

- 远端：`experiment/zcode-fattn-stream-20261006/`（`ub_slope2.sh` 驱动 + `matrix_runner_iso.py`/`benchmark_server.py` 隔离副本）；manifests `fattn-stream-ub{256,512,1024}/`（state.json + c1-p8192/c1-p65536 + parity + gpu-telemetry）；日志 `logs/fattn-stream-ub*.log`。
- **操作性发现 1（给后续会话）**：共享脚本 `scripts/matrix_runner.py` 与多会话共用的单元名 `ac922-flashnext-test.service` 是踩踏点——Codex five-micro 执行器/遥测会 stop 该标准单元（2026-10-06 23:13:32/37 两次击杀本实验初版驱动起的同单元名服务器）。后续并行实验应使用私有单元名 + 私有 runner 副本（v2 驱动的做法，实测与 Codex 两车道整夜和平共处）。
- **操作性发现 2**：`pgrep -af X | ... | grep -q X` 管道会匹配到下游 `grep -q X` 自己的命令行（管道各进程同时创建，pgrep 枚举得到）→ 谓词永真。检测 GPU 占用用 nvidia-smi 显存阈值即可，别用 pgrep 文本匹配。
- GPU 礼让：全程 wait-drain（min_free≥14000 MiB + 共享单元 inactive 才起服），臂间串行；实验结束卡已全部释放（0 MiB，unit inactive）。Codex five-micro（01-multi-mtp 编译、04-k8v4 集成）与 upstream 车道在实验窗口内多次占用 GPU，本实验自动让行。
- 复跑：`nohup bash experiment/zcode-fattn-stream-20261006/ub_slope2.sh >> experiment/zcode-fattn-stream-20261006/ub_slope2.out 2>&1 &`（臂序 512→1024→256，各含 8K/64K + 512-ID parity）。
