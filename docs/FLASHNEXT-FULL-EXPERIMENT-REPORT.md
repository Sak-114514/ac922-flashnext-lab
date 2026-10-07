# AC922 Qwen3.8-Flash-Next 全实验总报告（2026-10-01 ~ 10-04）

> 版本 rev3（终版）：经两轮独立子代理核查（第一轮 11 错误+7 遗漏全修；第二轮 4 必须+3 建议全修）。
> 所有数字可回源到文内引用的工件文件；无源可查的会话推断均已标注。

> 整机：IBM AC922（POWER9 ppc64le 176 线程、32GB RAM、CUDA 12.4 + 525 driver）
> GPU：6× Tesla V100-SXM2-16GB（sm_70、NVLink 一代 mesh、每卡 ECC 后 15.47GB 可用）
> 模型：Qwen3.8-Flash-Next GSQ-RCO（qwen4exp 架构：48 层 = 36 GDN + 12 QSA、512 专家选 10 MoE + 共享专家、
> PLE 稀疏嵌入表 16 头 × 2000 万行 ≈ 28.8GB IQ4_NL、1 层 MTP 草稿、~167B 总参/6B 激活）
> 模型文件：IQ3_XXS-MTP-Q8.gguf 73.2GB（ISTA-DASLab 量化；GPU 实载 ~46.4GB，**PLE 表 26.8GB 常驻 CPU 侧**）
> 执行者：Codex（10-01~10-03）与 ZCode+子代理编队（10-03~10-04）

---

## 0. 三条战线命运总览

| 战线 | 仓库 | 最终 decode | 结局 |
|---|---|---|---|
| **llama.cpp fork（主线）** | llama.cpp v0.5.0 (#29333) + 深度补丁 | 单路 94.3（合成口径）/ 4 路聚合 103.4 tok/s | ✅ 生产（本报告主体；MOE_REWORK 经启动器显式开）|
| 1Cat-vLLM 线 | vLLM fork（SM70 特化）| **29.19**（run l：V 排列实证版，集成验证跑通、前 5 token 与 oracle 一致、第 6 起分岔未定位；run k 30.94 已被 revert 不作依据）| ❌ Codex 额度耗尽弃线 |
| Strata 线 | Niko1221/Strata (MIT) | 66.1（层分片+MTP）| 📦 收档（架构不适合：prefill 磁盘流式）|

---

## 1. llama.cpp 主线（生产战线）

### 1.1 仓库与技术底座

- 源码：服务器 `/home/debian/ac922-llama-next/sources/llama-v100-v3`（本地镜像 `flashnext-llama-experiment/sources/`）
- 上游：https://github.com/ggml-org/llama.cpp （基线 **v0.5.0 = commit 7fe450e / #29333, 2026-09-28**） + qwen4exp 全架构支持（GDN/QSA/PLE/MTP）+ 3+3 岛结构 + meta backend
- 部署形态：`--tensor-split 1,1,1,1,1,1` 六卡行分片（attn 走 N 切、ssm_out/down 走 K 切、专家每岛全量复制）
- 生产 preset：`manifests/island-h-fifo-single`（岛 3,3 + H + FIFO 三件套 + MTP4）
- 两套基准口径（严禁混比）：**合成口径** = matrix_runner `--cases 1:8192`（重复文本 8K prompt / **512** 输出 / greedy，MTP 接受率虚高 100%）——六战役 A/B 全用此口径；**真实语料口径** = real 8K/256（R19 锚点 PP 1038.65 / **TG 83.03**，接受率 63%）。两口径数字不可直接相除

### 1.2 Codex 时期战役（R1-R21，10-01~10-03）

**转正落袋（生产在用）**

| 优化 | 开关 | 实测 | 证据 |
|---|---|---|---|
| H：MoE 死块消除 | `GGML_CUDA_MOE_GROUPED=1` | PP 532.6→586.2（+10.1%）；2×1K PP +118%；64K 多路 PP +14% | h-moe-1x8k 等 |
| B：QSA 池化多路（限长上下文）| `LLAMA_QSA_POOLED_MULTI=1` | 2×64K TG 13.2/17.5→25.5/27.9 | b3-2x64k |
| **B+H 合体** | 两者同开 | 2×64K 聚合 30.7→**77.9（+154%）**，parity=True | bh-2x64k |
| C：岛内 3+3 分组 | `LLAMA_ISLAND_SPLIT=3,3` | 单路 532→749（+40.8%）；2×8K 聚合 92.1 | C17/C18 |
| E：草稿不对称去 TP | `LLAMA_DRAFT_DETP=1` | TG 78.85→84.52（+7.2%），draft 3.4→2.2ms | （后被岛 preset 显式关闭，与岛互斥）|
| R18：FIFO prefill 组合 | ASYNC_COPY+FIFO_PIPELINE+KV_PAD8192 | **PP 749→966.51（+28.9%）**，TG +2.27% | island-h-fifo preset |
| R19：正确性修复三连 | — | chain 同步缺口/stale 携带/确定性 top-k → **83.03 锚点** | ROUND19 |
| R21：PLE O_DIRECT 行缓存 | opt-in 未转正 | 冷 prefill **241.95→360.02**（vs 32/8 无缓存基线）| ROWCACHE fixture |

**失败/判死路径（全记录）**

| 路径 | 死因（实测证据）|
|---|---|
| 岛间合并整步图（常驻执行器）| R17 nsys：图隙 3.557%+通信 9.879%=13.4% 候选上界 < 50% 准入门槛 |
| R14 岛步时门 | t_compute 22.88ms 未达 <12ms；"23→9-11ms" 预测被直接计时推翻，性能承诺冻结（结构代码保留）|
| R15-F Onecat PLE 卷积 kernel | +0.39%/−0.78% 无收益默认关 |
| R15-G Onecat XQA（fattn-volta-onecat）| parity=False、TG 22.40、MTP 0/2034——数值语义不兼容打崩 MTP 链，弃用 |
| R18 四子项未过门 | 锁页 argmax +0.45%、GPU 隐藏接力 +3.78%、GPU token 链（接受率跌 26% 判回退）、异步跨岛桥 +7.47%<10% 门（仅 FIFO 组合胜出）|
| R16-B MonoMoE 单 launch | +2.235% < 8% 门槛，默认关（代码保留）|
| TECH-FACTS 已证伪清单 | NCCL 调参全灭、图缓存 TTL/容量 cap 双证伪、B cell-sel 8K 修复被机器推翻（parity=False）、FP8 KV、KV sidecar、1Cat HC·GDN kernel 移植、A 定长步 |
| 连续批处理（短上下文）| 2×8K 聚合 92.13 < 单路 100.04（C17/C18 实测；根因归 MoE 专家读取放大属会话推断无文档）|
| D：BF16→F16 权重转换 | TG -4~5%（V100 无 BF16 tensor core，cuBLAS 快路径要 ne[1]>8，decode matvec 不受益）|
| wrapper 13.65ms/步解剖 | R17 nsys + R16-B cooperative 实测双重判明：可回收上界仅 13.4%，直接实验只值 +2.2% |
| R20：六卡 PLE 分片（冷 prefill 终极方案）| fixture 全过、真实 prefill 207→966（4.7×），**但 CPU/GPU 第 71 token 分岔（logprob 0.243）+ 132K OOM**——暂缓转正归档 |
| GPU PLE 图路径数值差异 | 1K 局部审计 PLE lookup 全逐位一致 → 差异在后续图/数值路径，未解 |
| 132K 单路 OOM（R20 UB512 首次 prefill GPU3 OOM；R16 GPU-PLE graph reserve 981.4MiB OOM）| R20 降 UB256/boundary29 继续但 GPU PLE 判缓；R21 是独立的 CPU 行缓存线 |

**诊断方法论沉淀**：R16 墙钟拆账（50.11ms/步七段分解）、R17 nsys 77.8 万 kernel 事件 trace（BF16 matvec 33.1%、量化 matvec 31.2%、MoE 12.3%、通信核 9.879%）、`graph census`（全图节点 dump）——本报告后续所有 kernel 战役的靶子全部来自这套测量。

### 1.3 ZCode 编队时期（10-03~10-04，六战役 + 并发矩阵）

**战役一：BF16 GEMV 单 kernel（✅ 生产，+8.9%）**
- 技术：uint4(128-bit) 流读 + bfloat16 位移转换（免 cuda_bf16.h）+ ROWS 行间 MLP + warp/两级归约；CUDA Graph 可捕获
- 微基准三轮：TP 分片形状假设被运行时修正（hc 系不切分；真实命中 320×10240 实测 **2.3×**）；staged(shared 预取) 判死
- 接线：42 行最小 diff，白名单 k≥10240 || (nc≥5 ∧ N≥8192)，开关 `GGML_SM70_BF16_GEMV` **默认开**
- 实测：TG 111.24/110.50 vs 100.17/103.35（**+8.9%**），parity 4×512 逐位一致
- 工件：`bf16-gemv/`（kernel_sm70_bf16.cuh + INTEGRATION.md）

**战役二：整层融合 launch（✅ 落袋中性，技术储备）**
- 技术：依赖 wave 分解（一层 16-17 launch → 8 wave 设计值；**实际接线只融合 n_task≥2 的 wave → 6**）+ SoA 任务表 + block 二分认领 + static 参数空间表
- 微基准：GDN/QSA × nc1/5 四组 **1.41-1.50×**（W7 四小矩阵 -66%）；修复教科书 bug：`__ldg` 用于 kernel 参数空间指针 → sticky error 毒死全部后续 CUDA 调用
- 接线：parity 4/4 过，但**端到端中性**（-0.05%）——DESIGN 的 BF16 清单错误：wqkv/wq/wk/wv/ssm_out/shexp 全是量化格式非 BF16，主力 wave 不存在，可融合组仅 ~6 组/eval
- 教训入档：**目标清单必须 GGUF 运行时验证，不能代码推断**（此教训直接改进了后续战役任务书）
- 开关 `GGML_SM70_FUSED_PROJ` 过门后默认开（无害）

**战役三：MoE Q2_0 kernel 重组（✅ +1.27% 落袋）**
- 技术：grid 12800→**1000** blocks（调度地板 22.2→**~3.5µs** 实测，kernel 头注释与 qmmvq-audit/REPORT.md §2.2 一致）+ 对齐窗口 uint4 流式 staging + smem 行打包
- 微基准：Q2_0 down 3.0×（uniform）/1.9×（生产 distinct）；**流量勘误：审计低估 5×**——MUL_MAT_ID 的 per-token distinct 路由使 nc5 实读 ~50 专家/launch（9.2MB）
- 接线：白名单 K∈{192,256}，parity 4/4，+1.27%（93.06 vs 91.90 臂均值；INTEGRATION.md TL;DR 记 +1.16%）——与流量勘误重估区间（0.9-1.4%）精确一致
- IQ4_NL distinct 倒退、IQ3_S 解码 ALU 病——均按裁决不追加
- 开关 `GGML_SM70_MOE_REWORK`（默认 0；启用即吃 +1.27%）

**战役四：dense K-quant 流式化（❌ 报败，最有价值的失败）**
- 三代架构全试（v1 shared 窄读 46% / v2 shared 重排 24% / v3-5 BPT 寄存器 15-17%）全部倒退
- 归因实验：**流读骨架本身 78.8% 带宽（对的），decode 相吃 70µs（12× issue 预算）——GGUF 原生 Q6_K 布局（210B/块、12 个 16B 窗口、pad 相位重排）在 V100 上无法就地解码打穿**
- 结论：唯一出路是**离线 repack/行交织**（R4）——此失败直接定义了下一个方向
- 资产：三向对拍方法链、零权限剖析变体（stonly/LOADSONLY/NODECODE）、V100 教训清单（SASS 级）

**战役五：ik_llama 移植评估 + fused MoE 移植（评估✅/移植❌）**
- ik_llama.cpp（IQ 格式作者本人 fork，https://github.com/ikawrakow/ik_llama.cpp ；本地克隆 `opensource/ik-llama/`）三候选评估：fused gu+gate 微基准大胜（IQ3_S 2.27×）→ 移植立项
- 移植判决：**正确性 40/40 逐位一致，但收益是编译器 CSE 幻觉**——ik 原核 channel-uniform 假设（nc==1 语义）下 x 解码被 CSE 消掉；改成生产 distinct 语义后收益蒸发，全模型 **-3.6%**（88.76 vs 91.98）
- faithful 错误语义当场复现 2.33× 铁证归因；kernel 留树默认关
- **跨战役教训：MoE ids 的 per-token distinct 语义两次让 uniform 微基准收益蒸发**（战役三勘误 + 此役）
- 附产：launch_bounds/rows=2 候选无货（fork 已有同款且快 1.04-1.07×）；ik MMQ 判死（比 fork MMVQ 慢 1.6-2.4×）；格式侦察独有格式全景（ik _R4 行交织后缀 7 种格式 = 最大可反哺技巧；注意 "R4" 一名两用：报告中的 R4 路线指离线 repack 方向，ik _R4 是其格式后缀）

**战役六：多并发叠加矩阵（✅ 8 场景全跑完）**
- 方案：concurrency/PLAN.md（资产盘点 + KV 显存账 + 风险清单；排除 b11374 假警报、`--cache-reuse` 正确性雷区）
- 结果（RESULTS.tsv，聚合 decode）：

| 场景 | 配置 | 聚合 | 备注 |
|---|---|---|---|
| **B3 4×8K** | 岛+H+FIFO 全开 | **103.43** 🏆 | 超预期 24% |
| B1 4×8K | 岛+H | 94.94 | |
| B1 2×8K | 岛+H | 91.15 | 复现 C18 |
| B3 2×8K | +FIFO | 92.50 | |
| C1 6×8K | 岛 H+B 极限 | 85.52 | |
| C2 6×8K | 六路 H+B | 81.62 | |
| B2 2×8K | B 强制 | 66.28 | 证实 B@8K 劣势 |
| A1-A3 2×64K | 长上下文 | 36.6-40.8 ⚠️ | 双流不均衡（2.7/38），历史 77.9 锚点未复现，视为下界 |

- 意外收获：**FIFO 对长上下文 prefill 增效 +81%**（PP 478→864，A1→A3）

**单路整合实测（两开关全开）**：`launch_integration_arm.sh` → matrix_runner（合成口径 8K/**512**）两轮
**93.46 / 95.20 tok/s（均值 94.3）**，parity matched=true。
⚠️ 口径说明：合成口径内同臂对照的增量 = BF16 +8.9% + MoE +1.16%（各自的 A/B 实测）；
94.3 对真实语料锚点 83.03 不能直接算百分比（跨口径）。真实语料口径的整合实测未单独跑

### 1.4 判死/边界清单（两时期合并，防止重蹈）

岛间合并整步图（13.4%<50%）、连续批处理短上下文（聚合倒退）、cooperative 单 launch（+2.2%）、
wrapper 13.65ms（双重判明）、D 线 BF16→F16（TG -4~5%；归因细节为会话推断）、fused proj 端到端（BF16 人口不足）、
ik fused MoE（CSE 幻觉 -3.6%）、IQ3_S 就地解码（布局病）、IQ4_NL distinct（倒退）、
staged y 预取（sync 税）、GPU PLE 数值分叉（未解悬案）、2×64K UB512 OOM、
PLE→SSD 卸载（前提错误：会话核查发现 PLE 本在 CPU 侧，无实验记录）、`--cache-reuse`（正确性雷区）、b11374 并行修复（假警报：openvino 子目录）。

---


## 1.5 收官增补战役（10-04 晚：复活实验 / TTFT 专项 / SASS 审计）

**复活组合实验**（revival/RESULTS.md）：P2P_AR 历史值 +2.46% 在 94.3 配方不复现（+0.29%/+0.02% 判噪声）；MonoMoE 与 MOE_REWORK 结构性互斥（MONO=1 时 rework 命中 90→0），顶替后净不优——两者维持归档，生产配方不动。

**TTFT 专项**（ttft/RESULTS.md）：R21 行缓存（GGML_CPU_PLE_ROW_CACHE）冷 prefill +59~63%/TTFT 68→41.8s 复现且 parity 过，但热 prefill 稳态税 -26~29%——**不进生产**。两个机制发现：①重启必冷（server 加载自读 46GB 挤掉 PLE 页缓存，重启后首请求 TTFT ~68s）；②基准重复文本掩盖冷问题（冷热差仅 8%，高多样性 prompt 才暴露）。生产建议：启动后 dd 区间预热 PLE（TTFT 68→46s，零稳态税零改动）。

**SM70 SASS 审计**（sass-audit/REPORT.md，五阶段门禁）：11 kernel census（全部零 HMMA）；自建 QPN 参照核（fragment-native mma.m8n8k4，parity 过）实测 40.8µs 比我方生产核慢 2×——**fragment-native 路线三家族全判死**：BF16 已 92% 带宽上限零冗余；MoE Q2_0 带宽地板封顶 1.47× 达不到 2.2× 需求；dense quant 唯一发现 ≥25% 冗余（nc5 解码按 token ×5 重执行）但 QPN 路线更慢，理性修法是就地 decode-once 重构（冻结红线外，移交记录：家族理论空间 ~4-5%，优先 ALU-bound IQ3_S）。勘误：fork 的 MonoMoE 非 mma 路线（weight-streaming FFMA）；1Cat marlin 是 sm_75+ 编不过 Volta。工具沉淀：IDP.4A 助记符、census 脚本、fragment 布局探针。

**dense-decode-once 分支战役**（decode-once/REPORT.md，10-04 深夜）：SASS 审计移交的唯一活口（nc5 ×5 重执行）被证伪——**nvcc 编译器早已把 weight 侧解码全部 CSE 出 token 循环**（热点循环 SASS 531 条指令逐类完全相同，0.989× 判死，双卡一致，parity relerr 1.19e-05）。审计的 ×5 发现实际是激活侧 dp4a 本体（80 IDP=16×5 token 不可约）。该形状真瓶颈=grid 查表 L1 重放（13.7µs vs 2.8µs 发射估计），IQ3_S 若再立项方向应为 grid 表 shared staging（iq2s_volta 先例 124→75µs）。分支 dense-decode-once 保留不合并，生产态 stash 序列双验证复原。**至此单路 kernel 战线全部收官：94.3 配方经受住 SASS 级+微架构级双重检验，无可再榨。**

**64K 长上下文 QSA 审计**（qsa-longctx/REPORT.md）：三嫌疑全清——①indexer 空壳不适用（qwen4exp 从不调用 lightning indexer，Volta 有 SIMT 兜底）；②单路 64K decode 已内置增量池化+固定预算+radix（nsys 实证 QSA 全链仅 ~0.54ms/步/卡≈1.7% 墙钟，外部 2-4× 修复的等价物全在）；③radix 已启用（CUB 生效，14604 实例）。**A 组 2.7/38 悬案真凶=预填充串行**（A 组缺 LLAMA_PREFILL_FAIR_SHARE=1）——配置补一个 env 复验：聚合 **40.8→74.73（+83%，达历史锚点 96%）**。移交 P1：多流 draft 图形状漂移致 40-60ms 重建（修后联合 decode 预计再 +20-30%）；P2：prefill QSA kvlen 钳制（2-5%）。

**GPU-PLE 分岔根因研究**（gpuple-forensics/REPORT.md，只读）：R20 悬案深挖——①top-k 假说证伪（R20 全部 7 manifest 的 GGML_CUDA_TOPK_STABLE=1 在场，跑在 R19 修复后，且 GPU 模式跨进程逐位复现排除运行间随机）；②**0.243 被当年误读**：max diff 在 index 0 且两边同 token——是 prefill 首次 forward 的 logits 系统性差 ~1e-2（BF16 舍入传播），第 71 token 只是首次翻转 top-1；③当年审计只覆盖第一个 256-token ubatch 的三张量（归档 patch 按名去重），31 个 ubatch+下游全未对拍。判决：**可解**（诊断 patch 扩逐层逐 ubatch，1 小时 bisect）但**默认不值得修**（dd 预热+R21 行缓存零成本拿走 GPU PLE 的全部收益；8K 专用模式被预热方案支配）。根因排序：六卡 PLE Meta 结构数值差 ~45% > 离散选择放大 ~25%。

**mtp-multistream-graph 专项**（mtp-msgraph/REPORT.md）：双门禁 FAIL 判死，但 P0 前提被三项推翻——①双流 draft 非"漂移"是 **catch-up↔draft-step 结构交替**（245/249 rebuild 由交替触发）；②CUDA 图本就在 replay（A2F 零 capture 来自交替），"40-60ms 重建"是 A2 串行预填充伪影；③根因=llama-context.cpp:1773 单活动指针使双槽图复用失效。三条修复路线全实测判死：跨槽复用撞 meta 层断言（红线邻区）、PAD_MULTI=10 得 -44%（padding 数学障碍确认：目标侧 n_pad≤0、draft 侧 recurrent 回滚被拒）、KV_PAD 桶粗化持平（真触发是 QSA 池化拓扑 churn 7259↔7547 节点）。可回收池合计仅 +6-7% < +10% 门禁。**移交升级 P2'**：QSA 池化桶取整（拓扑 churn 0.6-0.9s/窗口 = +6-8%）与原 P2 合并立项。8K parity PASS、B3 守护 101.03（-2.3% 调度噪声非回归）。生产树原样复原（stash 序列在案）。

**Strata 上游吸收战役**（strata-port/{REPORT,INTEGRATION}.md，10-06）：核实"prefill 已等效"说法不成立——fork 的 BF16 权重在 sm_70 batch>8 时走 F32 SIMT（ggml-cuda.cu:2000-2005），正是 Strata #655 修的病。**BF16→F16 prefill 补丁转正落生产**（默认开 `GGML_SM70_BF16_PREFILL_F16`，含 #540 ±65504 饱和钳制）：8K/64K prefill **+6.2%/+5.8%**，TG 持平，parity 逐位（512×2+64K+2×8K 2047/2048）；2×64K 墙钟 -2.8%。v70 QSA prompt attention kernel 判死（TP6 下 gqa=2 使其 G=12 硬编码吞吐塌 1/8，且仅占 prefill 3%）。**sglang-sxm2 深挖**（ASSESSMENT.md）：换引擎硬件否决（NVFP4 88GB 占 16GB 卡 92%+PLE 51GB>31GB RAM）；Triton 路线 ppc64le 实测可行（前人 ~/ac922-vllm 已编 torch2.10/triton3.6/flash_attn_v100，GDN kernel V100 跑通）；MTP recipe 可搬——FP16 KV accept 优于低精度（decision of record），直指我们 2.09 vs 2.47 差距，两个零代码实验（KV dtype A/B、draft 深度扫描）待做。

**K8V4/262K 扩容实测**（kv-262k/RESULTS.md，10-06）：四臂对照（Q8/Q8、K8V4×2ctx、F16）——**K8V4 parity 零翻转、decode 无损（TG 持平）、262K 池零 decode 罚**，代价 prefill -4~5%（V-q4 写路径格式代价）。262K 全家桶 3.51GiB/卡（+63% vs 现役，预估 2.2 被实测修正），峰值 85.8% 单流放得下。F16 臂 TG 反而 +1.6%（V100 f16 V 读快于 q8 反量化）但 262K 上不了。真实文本 accept 因续写族混杂无法精确测（合成基准饱和 1.000 无区分度），以 parity 零翻转为主证据。定价建议：**花 4-5% prefill 换 2× 上下文**——3×262K 并发需先重算显存（3.51×3>预算）。

**prefill 通信税攻坚**（prefill-parallel/REPORT.md，10-06，七臂全实测）：①`-sm row` 判死（CUDA 后端无 split buffer，加载即抛）；②**smlayer（`--split-mode layer`）8K PP +14.6%/TG +5.5%**（64K 持平——失注意力头并行+无跨 ubatch 流水），记强候选待多流联测；③P2P AR 复活判噪声（64K +0.0%）；④**病根再定位：64K 通信税 ~90% 是岛内失步**（MoE 专家按 rank 切分负载不均→AR 等最慢者，nsys 实测 AR 线时间 0.2ms vs 实例 2.2-3.2ms）而非协议/带宽；⑤六段流水 meta 路径双负（岛间层分配 2 岛硬编码需补丁且补丁后仍差）；⑥**C2' 跨 ubatch 段间流水立项**（唯一 ~2× 上界的 64K 路径，decode 兼容性已被 smlayer TG 背书）。

**Codex 上游回移战役**（experiment/codex-upstream-20261006/REPORT.md，10-06）：cherry-pick #29825/#29901/#30020 + **关键纠错：旧 GPU 顺序 0,4,2,1,3,5 把两岛混编**（PCI 总线交错）——UUID 固定正确 3+3 岛后图内 P2P AR 才真正生效。**最终组合（正确岛+P2P+新 indexer×2+R19 全家）实测：8K 920.6/125.1、64K 1003.4/108.8（vs 同轮冻结基线 648.9/93.7、738.8/83.0 = **PP+42%/TG+34%**），512-token 逐位一致，真实语料 949.2/98.5 热**。层流水 1176 prefill 因真实语料重复失败被诚实回退（重复文本 parity 通过但真实语料 token 37 分岔）。多流实测（final-mtp-multi）：2×8K 聚合仅 51/75（fair），4×8K 41/68——**GPU MTP 链/carry/FIFO 限单序列**，多 slot 快路径停用（Codex 列为下一轮首要适配项）。262K UB1024 首次 prefill OOM。Codex 下一步清单：多序列 GPU MTP 链+固定 verify 图、双微批重叠（vLLM DBO 参考）、PLE 定向预取、DWDP 式专家布局。

**MWE 成员负载均衡战役**（mwe/REPORT.md，10-06，判死）：EPLB 论证线的 P0 落地——设计伪码 {8,8,8} Q 头均分**不可实施**（CUDA 核内 GQA 映射纯本地索引，24 头/kv2/3 成员无连续等分解，且全注意力层实为分离 wq/wk/wv 非假设的 fused qkv），改用 **segment 置换**（61 行零 CUDA/图改动）：wq 6 段置换（每成员 8 头、每 KV 组各 4 头）+ wo 同序配对 + wk/wv/KV 缓存（主+块+MTP 层）MIRRORED，env `LLAMA_MWE_ATTN` 门控。验证链全绿：静态负载 245.6×→1.001×、加载 sums 逐项符、+554-640MiB/卡、**4 轮 512-ID parity 全真**。但 A/B 门禁 FAIL：64K PP 仅 **+0.39%**（594.5→596.8，门 ≥+5%），8K +0.1%，TG −1.2~1.3%（镜像开销）——**归因被证伪**：生产 64K prefill 注意力走 QSA gather 稀疏路径（top_k=2051/65536≈3%），全注意力段绝对量比设计假设小一个量级，修齐成员负载 AR 等待不转化；无岛 6-way 臂仅 +1.3% 佐证。**通信税归因再收窄**：排除注意力头失衡（本役）与 MoE 专家不均（1.8%，前置论证）后，剩余嫌疑=岛间串行/段到达次序/P2P AR 线时间本身——下一步应在当前树重抓 64K nsys 重新归因（现有剖面早于近期生产改动）。遗留资产：segment 置换+kv 镜像=parity 证明可复用的零 CUDA 头重映射基建（DWDP/EP 可直接复用）；分支 `mwe-20261006`（557b422→a465629 标注 FAILED）+ mwe.diff 存档；生产树逐字节还原、二进制无 MWE 符号复验。

**Codex 五方向微测**（experiment/codex-five-micro-20261006/RESULTS.md，10-06，微证据全部落盘、未接线）：①**多流 MTP**——GPU carry/chain/FIFO 仍限单 seq（启动标记=0，源码 guard）；CPU 布局参考 7/7 过、真实 2/4 路跑通，Phase 2 隐式 carry 实测无增益且 parity FAIL → 默认拒收。真实多流微波（64-token 冷启动，非稳态）：2 路每流 26/11、4 路每流 18/21/10/6，含 prefill 聚合仅 13-17 tok/s。②**双微批/图 UID**——复现具体失效机制：同 UID 原地切 src 仍读旧地址（违背捕获图地址稳定性）；Phase 2 修复后新 UID 校验通过、独立双 frame 重叠观测为真，但完整双微批流水未集成。③**PLE 缓存**——字节工作集很小（微样本 232 unique/416 gather）、页 I/O 放大明显；`GGML_CPU_PLE_ROW_CACHE=1` 已实接 lazy IQ4_NL GET_ROWS 且多流实机真实命中（Q8 hits 61640/direct 46014），记有界候选。④**K8V4**——Q8K/Q4V 原生向量 dispatch 编译通过，但 QSA sparse Q4V 不支持，且两格式 6 条输出 hash 全不同（不能沿用重复文本"Q4 无翻转"结论），未验收为无损生产。⑤**DWDP**——真实专家 P2P 后 gate/up/down 计算逐位一致（2 种量化证明），整模型布局/账本未做。**新红旗：多流确定性**——冻结 V1 短控臂自身重复也有分岔（首差 60/26、12/1/36/none），任何多流配置升生产前必须先查确定性。GPU 队列已清空、零部署、零长压测。

**fattn workspace 流式化前置侦察**（fattn-stream/REPORT.md，10-06，设计 1 判死、零源码改动）：三前提全被证据推翻——①**UB1024 从未被"挡死"**：132K+Q8K/Q8V 下早已可跑（8K 920.56/64K 1003.40 就是它）；4824.31 MiB OOM 仅发生于 262K+Q8K/Q4V 容量配置（device3 首次 prefill）。②**4824 MiB 不是 fattn workspace**：Q4V 令 qsa-attn.cu block-sparse kernel（要求 q8_0+q8_0）失效后，`build_attn_qsa_blk` 的 gather 回退物化 K/V F16 `[256,2,2304,UB]`（各 2304 MiB@UB1024）；账本零参数闭环（4824.31=UB512 尺寸 2412.155×2.0000，island-B 9377 权重+3594 KV+2412 buffer 精确吻合）。生产 q8/q8 下 QSA prefill 走 sparse kernel 且 n_split=1、workspace=0；UB 撑大的 compute buffer 全是普通激活（1.88 MiB/token 全线性）。③**批深度斜率为负（倒 U）**：三点同链 A/B（生产 env、512-ID parity 全过）64K PP UB256 687.37 / UB512 **745.21（峰）** / UB1024 706.50（**-5.2%**，与 prefill-parallel 会话独立实测精确互证）；8K PP 小 UB 更优（UB256 vs UB512 +5.5%）；TG 与 UB 无关。**现役 UB512 已在最优批深度；1Cat #754 批深度结论在本栈反向；"prefill 差距=批深度"假设作废**（与 MWE 收窄后的归因一致，剩余嫌疑=岛间串行/段到达次序）。副产物：**P2 改向=qsa-attn.cu 支持 q4_0 V**（~60 行）消 262K gather 回退、回收 k8v4 -4~5% prefill 税并根除该 OOM；P3=短 prompt 负载 UB256 免费 +5.5%。操作性教训：共享单元名 `ac922-flashnext-test.service` 会被 Codex 执行器 stop 击杀并行实验（后续改私有单元名+runner 副本）。勘误：深挖报告 B1 岛内头数应为 8/2、gqa=4。

**qsa-attn q4_0 V 支持战役**（qsa-q4v/REPORT.md，10-07，parity FAIL 判死、方向未死）：fattn-stream 的 P2 改向落地——block-sparse kernel 加 q4_0 V 反量化（55 行 2 文件、编译 339/339、env `LLAMA_QSA_Q4V` 默认关）。阶段 0 侦察交付：Q4V 两处失效点、gather 回退换算式 `((float)nib-8.0f)*d`+RN、cache 步长 144/288 与显存账目吻合。A/B：**8K PP +3.4%（654.15 vs 632.49）——读路径回收量级真实存在**；但 **parity token 0 即分岔**（on=3343 vs off=264）、TG -19%（accept 1.000→0.894 次生）——kernel prefill 数值 bug，判死不合入。行级定位未果（lane→nibble/镜像算术复核无果），独立单测 `test_qsa_v4.cu` 已备，复活裁决约 10 分钟（嫌疑：fork 布局变体/meta GET_ROWS 执行路径/-use_fast_math）。**事故：服务器 00:50 整机失联，生产树还原被打断**——后经 BMC 查明为**干净软关机**（HostState=Off、全天零 checkstop/热/电源故障事件，非崩溃；00:50 关机来源待开机后查 `last -x`/journal）。树停在分支 `qsa-q4v-20261006`（3 commit），生产 79 文件脏态完整保存在 `stash@{0}: qsa-q4v-freeze-20261006`，`qsa-attn.cu` 原版备份（md5 d6e43a61…）在 `experiment/zcode-qsa-q4v-20261006/`；REPORT §6 六步复原清单（checkout→备份回写→stash apply→diff stat 对账 12554/430→重编→符号验证 0 命中）约 5 分钟，**下次开机最优先执行，且先于 Codex 任何操作**。

## 2. 1Cat-vLLM 线（Codex，已弃）

- 上游：https://github.com/vllm-project/vllm （本地克隆 `opensource/super-ac922-clone/`，SM70 特化：qwen3_5.py 含 GDN+SM70 Triton split kernel、
  qwen3_next/mtp/dflash2、gguf.py 支持 IQ3_XXS、AWQ Marlin/turbomind 后端）
- 进度：eager 4.89 → 图模式 26.7/27.2（run h/i）→ 窄候选 30.94（run k，**已被 revert**）→ 终点 run l **29.19 tok/s**；prefill 稳态 819.2（TP4 线 run r，decode 仅 5.182，跨阶段不可比；首块 wall 55.3s 含冷 JIT）
- parity 追踪树：PLE 哈希✓/卷积✓/查表行号✓/输出头✓ → 收敛到 **GGUF 布局逆转**（148 个 norm 预加 1、
  36 个 ssm_norm 不能减、ssm_a 负指数→A_log、V 头 tiled 重排、in_proj_qkvz 拆分例外）
- 终点（run l，pp3x3-hf-verified-inverse-20261003l）：V 排列（QKV/Z tiled）实证修正版**完整跑了集成验证**——真实 8K/512 请求 29.19 tok/s，前 5 token 与 llama.cpp oracle 一致，第 6 token 起分岔（原因未定位，文档注明"进一步正确性工作待做"）；此前 run k（30.94）已被 revert。Codex 额度耗尽，用户裁决弃线。fixture：TP3 QSA 7/7、GDN 7/7、MoE 6/6
- 遗产：onecat-kernels/ 参照库（glm53_gemv 骨架、HMMA 汇编、marlin——上游 https://github.com/IST-DASLab/marlin ）、GGUF↔vLLM 转换知识（norm ±1 惯例）、评估文档


## 2.5 Luce 评估线（ZCode，纯评估未移植）

- 上游：https://github.com/Luce-Org/luce-megakernel （消费级专用引擎，clone HEAD 2d9d4c0，2026-10-03；本地克隆 `opensource/luce-megakernel/`）
- 判决（reports/luce-port-assessment.md，292 行带行号证据）：**megakernel 是 0.8B 硬编码研究 PoC 不可用**（"官方支持 V100"指 server 引擎）；Route A 移植 75-135 人日太贵；**Route C 借技术回 fork 15-30 人日**（判决执行中）
- Route C 四件套落地情况：①moe-fused/CachedFfnGraph 思想 → 输入了战役二（layer-fusion/DESIGN §4 整节摘录）；②delta_net_chunked verify（未做）；③Volta fattn-wmma-f16（未做）；④moe_hybrid L+1 预取（未做，长上下文备用）
- 影响：本报告的战役二、三设计直接受其输入；"wrapper 13.65ms" 的核销也引用了其 megakernel 章节结论

## 3. Strata 线（ZCode，一天，收档）

- 上游：https://github.com/Niko1221/Strata （MIT，Qwen3.8-Flash-Next 专用消费级引擎，ggml 底座；本地克隆 `opensource/strata-engine/`，服务器 `/home/debian/strata-engine/`）
- 移植（1 天）：ppc64le 编译（cpuid 守卫/x86 SIMD stub/ggml GNU 扩展）+ 5 专家卡扩展（remote array 3→5）+
  `--remote-mirror-all`（堵自适应层内容变化产生的"无人持有洞"）
- 战果链：7.42 →（adapt/lookahead 关）9.64 →（**sm_70 零拷贝陷阱修复**：kernel 读 host-mapped 内存每次 ~280µs，
  关掉后 wait 67.6→2.3ms）25.87 →（热画像+大 cache 图内命中）32.75 →（MTP 转换器落地，接受率 0.376）**40.48** →（层分片 serve）**66.1 tok/s**（接受率 0.662）
- MTP 转换器：blk.48 → checkpoint 布局（eh_proj 沿输入维切分、shared_expert_gate 2-D、norm 减 1），
  接受率 0.376（Q8→Q2 双重量化 lineage），flip-eh 归零验证映射正确；**输出 64 token 与非 MTP 逐位一致**
- 弃因：prefill 1.19（首跑实测） tok/s（磁盘流式架构 vs 我们 GPU+CPU PLE 分工——Strata 全部权重流式、我们的 PLE 在 CPU 其余 GPU）、每层主机门铃握手——架构基因不合
- BF16 MTP 源下载 100%（镜像断流两次），chain 未续（收档尾巴）

## 4. 最终数字汇总

| 口径 | 数字 | 配置 |
|---|---|---|
| R19 基线锚点 | 83.03 tok/s（真实语料 8K/256）| 岛+H+FIFO+MTP4；合成口径同代锚点 ~100 |
| 合成口径对照 | BF16 A/B 100.17/103.35→111.24/110.50；MoE A/B 91.98/91.83→93.15/92.98（BF16 钉开）| 增量以各战役自身 A/B 为准；跨批次基线漂移 17-20%，不可横比 |
| **单路整合实测** | **94.3 tok/s**（合成口径 8K/512）| + BF16_GEMV + MOE_REWORK，parity true；合成口径内增量 +8.9%/+1.27% |
| **4 路并发聚合** | **103.43 tok/s** | B3（岛+H+FIFO+fair_share，slots=4）|
| 6 路极限聚合 | 85.52 | C1 |
| 2×64K 长上下文 | 36.6-40.8（下界，待修不均衡）| A 组 |
| prefill（热）| 532→966（+82%，R19 warm 1038）| 岛+H+FIFO 全在生产 |
| prefill（冷启动）| R21 行缓存 241.95→360.02（+49%，opt-in）；R20 六卡分片 207→966 已暂缓（数值分岔）| R20/R21 归档 |
| 预估单路上限 | ~97-99 | R4/repack（+3-5%）+ 小池 |
| 预估并发上限 | ~120-140 | B3 配方 + A 组修复 + 调优 |

## 5. 生产资产索引

- fork 开关四枚：`GGML_SM70_BF16_GEMV`（默认 1）/ `GGML_SM70_FUSED_PROJ`（默认 1，中性）/
  `GGML_SM70_MOE_REWORK`（默认 0，**启用即 +1.16%**）/ `GGML_IK_FUSED_MOE`（默认 0，判死留档）
- 启动器：`scripts/launch_integration_arm.sh`（单路全开）/ `scripts/multi-concurrency-bench.sh`（并发矩阵）
- 报告链：reports/ 下 FINAL-REPORT、sm70-kernel-campaign、strata/luce/1cat 三评估
- Codex 交接三件套：GPT-HANDOVER / TECH-FACTS / CURRENT-STATE（改动前必读）
- 外部参考源：上游 llama.cpp（https://github.com/ggml-org/llama.cpp）、ik_llama.cpp（https://github.com/ikawrakow/ik_llama.cpp）、Strata（https://github.com/Niko1221/Strata）、Luce（https://github.com/Luce-Org/luce-megakernel）、Marlin（https://github.com/IST-DASLab/marlin）、vLLM（https://github.com/vllm-project/vllm）；模型 ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF（https://huggingface.co/ISTA-DASLab ）
- 悬案（可重启）：GPU PLE 数值分叉、2×64K 不均衡、冷 prefill 根治（R20 半成品）、R4 行交织、A 组修复

## 6. 方法论遗产

1. 两阶段闸门：微基准达标（真实形状+正确语义）才准接线——两次防止无效接线
2. MoE kernel 评估必须 distinct ids 对拍（同一坑两次教训）
3. 目标清单运行时验证（GGUF tensor dump/graph census），不信代码推断
4. 微基准 L2 污染（>20MB 才信 copy 上界）、ids 语义、CSE 幻觉——三大数据陷阱各有实测案例
5. 子代理编队：并行战役 + GPU 分卡 + 编译排队 + 时间盒 + 诚实报败纪律
