# GPT 交接文档：AC922 推理优化战役现状（2026-10-02 11:00）

> **R19/R20 free-optimization update:** R19 repaired GPU-input producer ordering, stale MTP request carry, and deterministic QSA top-k. Fresh regression960.79/111.07; warm regression1055.16/113.13,512 IDs exact. Real8K256 now repeats exactly at206.71/59.38 cold and1038.65/83.03 warm (182/288 drafts both). See `final-results/ROUND19/RESULTS.md`. R20 independent six-GPU PLE keeps the trunk3+3; six-way gather48 replays exact. First132K/UB512 reachedREADY but firstprefillOOM. Current newcandidate usesboundary29/UB256, directshared-devicePLEbridge andprefetch0; still under test, no performance success yet. Goal remains active.


> **2026-10-02 GPT最新结果：C已跑通并与H整合。1×8K 749.79/100.04；2×8K 299.72/299.71、45.74/46.39，512-ID parity均True。下文v13是历史死路，不能继续按“均分三路Q-gate”修。真实修复为整GQA分组、KV/RS同名buffer map身份隔离、深split-state显式栈。详见 `final-results/GPT-ISLAND/RESULTS.md` 和 `CURRENT-STATE.md` 末尾。长C+B未验证，E与C互斥。**


> 若你接手，从"第 1 节现状"开始读；技术细节的证据链全部在 `TECH-FACTS.md`（含证据分级与全部推翻记录），各补丁设计在 `patches/*-NOTES.md`。

## 1. 已验证落袋的（直接可用，勿重测）

| 优化 | 开关 | 实测（parity 门禁全过）|
|---|---|---|
| H：MoE 死块消除 | `GGML_CUDA_MOE_GROUPED=1` | 单路 PP 532→586；2×1K PP +118%；64K 多路 PP +14%；decode MoE matvec 也提速 |
| B：QSA 快路径（限多路长上下文）| `LLAMA_QSA_POOLED_MULTI=1` | 2×64K decode +74% |
| B+H 合体 | 两者同开 | **2×64K decode 13.2/17.5 → 38.7/39.3（2.9×/2.2×），总吞吐 30.7→77.9（+154%），parity=True** |
| E：不对称草稿去TP | `LLAMA_DRAFT_DETP=1 LLAMA_DRAFT_DETP_TRUNK_W=0.5` | 单路 TG 78.85→84.52（+7.2%），parity=True，draft forward 3.4→2.2ms |

基线锚点（speed-bundle 冻结）：1×8K 532.6/78.85；2×8K 226/34.2+35.4；2×64K 153/13.2+17.5。

## 2. 岛内 3+3 分组（C 线）：结构通过，R14目标步时门失败

**实际测试的就是本节 C 方案**：同一个模型按层放置到两个 meta device；GPU0/1/2 在第一岛做 TP3（加载日志 blocks 0..24），GPU3/4/5 在第二岛做 TP3（后续层及输出/MTP）。跨岛保留层边界 activation copy，AllReduce 限定在岛内。启动参数为 `LLAMA_ISLAND_SPLIT=3,3`、GPU 顺序 `0,1,2,3,4,5`、`CUDA_DEVICE_ORDER=PCI_BUS_ID`。

### 已实测（2026-10-02）

| 场景 | PP tok/s | TG tok/s | 输出门禁 |
|---|---:|---:|---|
| C，1×8K | 633.70 | 99.93 | 512 IDs 完全一致 |
| C+H，1×8K | 749.79 | 100.04 | 512 IDs 完全一致 |
| C+H+B长度门，2×8K | 299.72 / 299.71 | 45.74 / 46.39 | 两路各512 IDs完全一致 |

- NCCL日志确认通信组只包含[0,1,2]或[3,4,5]，AllReduce仅nranks=3。
- 132096单路ctx和409600/4-slot ctx配置启动成功；上表只验证8K真实输入，没有长prefill或四路满窗验收。
- B的32768累计真实长度门在2×8K时未触发；长C+B仍待验。E的`LLAMA_DRAFT_DETP=1`会禁用岛模式，已从岛preset显式关闭。
- **“步时23→9–11ms”已由R14直接计时推翻；通信−92%、同步96→0、prefill通信÷13仍无新增nsys验证。100 tok/s约10ms/token不能代替ms/step。**

### R14直接步时裁决（2026-10-02）

同一C+H参数仅岛开关不同，新标签各一次1×8K/512，parity均通过：

| 直接计时 | 岛开 | 岛关 |
|---|---:|---:|
| target t_compute，纯decode64采样调用 | 22.8795ms/步 | 22.7580ms/步 |
| target reused compute | 22.0119ms/调用 | 21.0712ms/调用 |
| draft decode，300-call窗口 | 2.437ms/次 | 3.382ms/次 |

**目标步时不满足<12ms且未优于本轮基线；“23→9–11ms”预测推翻，性能承诺冻结，结构代码保留。** 端到端TG仍较快（101.56 vs78.76），draft变快已确认，但其余等待/调度收益未单独剥离；不能与target步时混同。按本轮用户判据，不继续nsys和2×64K三合一。证据`final-results/ROUND14-15/A-verdict.json`和两份timing JSON。

### R15 F/G收尾

749.79配置原来不含F/G；live源缺F接线/G部署，旧“已编译待测”状态纠正。补现成适配后各唯一一次1×8K/512：F真实CUDA调用，748.74/100.77，parity=True，对本轮基线+0.39%/−0.78%，无明确速度收益，默认关闭；G有Onecat激活日志，但747.79/22.40、parity=False、MTP0/2034，按用户弃用不修。普通preset显式F=0/G=0，可选`island-hf-single`只验单序列。没有推生产。完整证据`final-results/ROUND14-15/RESULTS.md`。

### R16后续：墙钟拆账、MonoMoE与GPU PLE

A的103步平均50.11ms：target graph24.95、draft graph6.34、显式sync1.80、sampling2.78、投机簿记0.44、模型包装/未细分等待13.65、server0.16。采样+簿记3.21<10ms，没有触发该优先分支；不能把图外27ms全叫CPU。

主代理实现SM70七格式单launch MoE：40+20 fixture通过，最终1×8K/512 **751.66/102.28，parity=True**，但只+2.235%未达≥8%，默认关闭，不推生产。不要把早先scale=0错误版本的速度计入证据。C没有新的graph内launch/sync>50%证明，未立项。

GPU PLE按H+E、TRUNK_W0.5、ctx132096重试：E实际激活，但graph reserve CUDA device1申请981.40MiB OOM，未到实例化和8K请求。仍不能使用该132K组合。

完整证据：final-results/ROUND16-A、ROUND16-B、GPU-PLE-RETRY；B源码experiment/monomoe-sm70/和ggml/src/ggml-cuda/monomoe-*。

### R17：岛内完整GPU图已验证，常驻执行器门未过

启动器新增AC922_TEST_P2P_AR=1会unset P2P_AR_DISABLE（原设0也禁用）。两岛12个数值/graph replay fixture通过。1×8K/512：chunk512为758.58/102.70；chunk0为758.40/102.50，均parityTrue。chunk0 target compute19.4505ms（R14为22.8795），draft2.413ms。对100.04锚点TG+2.46%，无统计重复性结论。

nsys真实抓到每GPU97次主图重放：第一岛每图50个P2P归约，第二岛47个。残余NCCL606次集中图切换窗口，CPU PLE/跨岛边界保留。图内间隙3.557%+通信核9.879%=13.436%候选上界，未满足整步常驻执行器>50%门；CPU API57.039%与GPU重叠，不可混用。按原门槛不立项，不能把未写的常驻执行器称为完成。

可复用实验配置manifests/island-h-p2p-whole-single；原始证据final-results/ROUND17、remote profiles/gpt-r17-trace-20261002-1603-a3。没有推生产或修改硬件，测试服务已停。

### R18：顺序实验完成，保留prefill组合

按单项闭环完成：锁页argmax752.54/100.49（未过门）；GPU隐藏接力749.40/103.82（未过门）；GPU token链752.18/43.00（接受率26%，回退）；异步跨岛桥805.82/103.39（独立未过门）；**FIFO输入快照+跨岛桥+prefill KV_PAD8192为966.51/102.31，512-ID parityTrue，prefill+28.90%，保留opt-in**。compute22.5495ms，仍是host graph_compute调用时间。

新preset `manifests/island-h-fifo-single`；原preset不变，未推生产。B/C实现已从runtime完整移除，仅留归档patch。runtime只保留ggml-backend-meta.cpp、ggml-cuda.cu、ggml-backend.cpp、llama-context.cpp的默认关闭异步/FIFO功能。仅1×8K已验，真实长输入/多路未验；不得把组合收益归因成已量出的GPU重叠比例。失败原因、原始数据、回退、hash在 `final-results/ROUND18/RESULTS.md`。

### 修复和证据

1. 撤销v13错误的Q-gate三路均分：两个KV头按整GQA组保持Q/K比例，允许合法空成员；本模型KV粒度256，旧文所写512及“[0,256,256]是污染”已推翻。
2. 修KV/RS/DSV4/loader按同名Meta()合并buffer context的遗漏，按对象身份隔离两个岛的缓存。
3. 修Power深split-state递归栈溢出，改显式后序栈；快照清cache提前，host view不误入meta分析，调试日志只读cache。
4. 最终构建通过；900层深图旧快照失效后的数值=901，在1MiB进程栈且debug开启时也通过。

完整实测、源文件、回滚和preset：`final-results/GPT-ISLAND/RESULTS.md`；remote manifests为`zcode-gpt-c16-island`、`zcode-gpt-c17-island-h-single`、`zcode-gpt-c18-island-hb-multi`。最终C19只修debug日志分支，未重复GPU矩阵。

### v13及以前的历史现场（已解决，保留死路地图）

**最终状态（11:07 终审）**：v13 的粒度修复（6144→4096 岛内均分）**已生效**——v12 污染断言不再触发、{0,·,·} 家族签名消失；但暴露了下一层：`meta.cpp:1381 GGML_ASSERT(split_state.ne[j]*nr[0]*src->ne[axis] == sum*tensor->ne[axis])`（比例守恒断言，裸断言无名字）。**给你的第一步建议**：给 1381 加自报家门（本战役的成熟模式，参考 v11/v12 的做法），一次运行拿到张量名——大概率是另一个量化/粒度类的同类舍入问题（v13 只改良了 Q-gate 类，KV 类 512 粒度被有意保持不动）。

- **价值**：跨岛通信 decode −92%、同步 96→0，预估步时 23→9-11ms、prefill 通信 ÷13（全部未验证）。
- **现状**：加载/布局/预检/显存分配全通，卡在**权重张量的 split-state 构造**。
- **精确现场**（v12 产生点断言，meta.cpp 污染检测器）：`blk.3.attn_q.weight [NONE] axis=1 ne=[2560 12288 1 1] sums=[0 6144 6144]`——成员 0 得 0 行。数学线索：12288/2=6144，疑似某处按 2 而非 3 分割、或 scan 首条目清零后归一化。
- **已排除（勿走弯路）**：split 回调交互（插桩实证 tensor_split=[1,1,1]、detp_scale=0 干净）；C 的惰性化（结构论证不可能改写状态）；"第 0 槽未填"类循环错误（静态排查无）；上游 llama.cpp 无此机制可参考（100% fork 私有）。
- **洋葱全链**：v1 成员分布→v2 边界+预检→v3 ctx 同名合并→v4 诊断化→v5 空张量→v6 CPY 轴传播→v7 GDN 镜像态→v8/v10 真实段+6路分流→v11 命名化→v12 产生点断言→v13（进行中，止损线最后一轮）。
- 全部改动 env 门控（`LLAMA_ISLAND_SPLIT=3,3`），不设即原行为；备份 `.zcode-orig/` 分层。

## 3. 已证伪（勿重试）

NCCL 调参全灭（autotune 最优；强制 LL −22%；Tree 全败）；图缓存 TTL（时间驱动假设错）与容量 cap（实测无差）；B 的 cell-sel 8K 修复（parity/速度双未兑现）；A 定长步（hybrid 递归目标数学无效）；FP8 KV / KV sidecar / 1Cat HC·GDN 内核移植。B 的 8K 差距未解但 64K 价值已锁（可用长度感知门 `LLAMA_QSA_MULTI_MIN_CTX`，默认 32768，B round-5 已实现）。

## 4. 运维红线（重要！今早的实战教训）

1. **测试前必查 vLLM 复活**：`ac922-coding-api.service`（TP3+FLASH_V100）开机自启，会占 GPU0-2 九成显存并诱发主机冻结（PHB4/SATA 病史）。先 `systemctl --user stop ac922-coding-api.service ac922-acestep.service`，`nvidia-smi` 确认 0 MiB。apigateway 的 uvicorn 会重生，需正式 disable（留给用户决定）。
2. BMC 的 HostState 在 degraded 态误报（Off 读数不可信）；开主机用 `RequestedHostTransition=On`（Chassis 同值重置无边沿）。
3. 同标签重跑必须清旧 log（追加污染今早造成一次误读）。
4. 编译 `--parallel 6`（32GB 内存约束，幽灵 DIMM 修复后只有 32G 在线）。
5. boot-guard 曾在岛测试崩溃同秒死亡（mask 超时后安全回退 4000 转）——主机重崩后记得查它。

## 5. 工具与数据

- 测试框架：ac922 `scripts/zcode_probe.py`（--frozen/--ref/--cases/--set，带 parity 门禁+58°C 守护）；序列脚本 `scripts/zcode_series*.sh`。
- 全部实测：ac922 `manifests/zcode-*`；日志 `logs/zcode-*`。
- F16 转换模型（备用）：`models/.../IQ3_XXS-MTP-Q8.bf16-f16.gguf`（收益边缘已冻结）。
- F已测parity通过但无明显提速，默认关；G已测parity失败并弃用；MonoMoE已测parity通过但只有+2.235%，未过8%门，默认关。

## 6. 后续切入点

C线短输入结构已通过，但R14目标步时门失败，性能承诺冻结；实验配置复用`manifests/island-h-single`和`manifests/island-hb-multi`，不要再按旧v13均分Q-gate或污染检测结论重修。本轮不继续原通信预测的profiling；如后续用户重新授权，应区分岛内AllReduce、边界copy、总同步次数和ms/step。长C+B、真实语料、长prefill与四路满窗仍待后续选择。本次没有推生产。


### 2026-10-06 — 上游回移与真实岛组合（Codex）

- [C/M] 只在 `experiment/codex-upstream-20261006` 隔离副本工作；3674 个原源码文件 SHA 核对无变化，生产/固件/风扇/驱动/内核/电源未改。
- [C/M] 本轮 `prefill-parallel` 旧 PROD_ENV 的 PCI_BUS_ID + visible `0,4,2,1,3,5` 混合两岛；明确 UUID 排序修正为 0004:04/05/06 与 0035:03/04/05。新日志两组真实图中 P2P AR 激活。历史测试须按各自设备顺序判断，不能统称以前所有岛实验都错误。
- [C/M] 选择回移 #30020 NextN reserve、#29825 最终 fused scorer/CUDA四头、#29901 64-key×8-token tile；9/9 CUDA indexer数值fixture与实际CPU NextN/cache fixture通过。#29994不同标准kpool布局不强套，现有partial seq_cp等语义保持。
- [M] 冻结本轮基准8K648.86/93.67、64K738.80/83.02。真实岛+P2P+新indexer+R19 GPU chain/embed/FIFO/stable top-k/compact/cache：8K920.56/125.06、64K1003.40/108.81，两档512-token回归一致。target graph_compute host调用窗口中位14.603/14.371ms；不是GPU完成时间或全MTP步时。
- [M] 固定真实8K两波输出512/512一致、367/573接受；冷183.83/71.29、热949.21/98.48。重复文本408/408不可作为真实接受率。速度候选132096ctx、MTP4、UB1024、Q8 K/V；单卡最小空余2281MiB、最高GPU/主机传感器46C（5秒采样）。
- [M] 262144ctx+Q8K/Q4V+UB512：8K881.43/113.92、64K987.49/99.20回归通过；只验证分配及64K实际输入，不是262K满窗。UB1024首次prefill workspace4824MiB OOM，已退出。
- [M/X] 新层流水小模型跨SYS双目标层、4-copy、reset/tail/自动输出drain通过8×128logits（max1.164e-10）；整模型64K1176.37/88.46、重复文本512parity，但真实重复token37分岔。同layer/R19/UB512关闭新流水时真实512/512一致。新增流水未过整模型门，源码3文件与runtime已回退V1；完整失败实现归档，不能将1176作为合格候选。
- 工件：`codex-upstream-20261006/REPORT.md`、`results.json`、`artifacts/upstream-bundle-v1.patch`、`artifacts/rejected-stage-pipeline/`。可复跑 `python3 experiment/codex-upstream-20261006/scripts/replay.py fast132`。测试unit全部退出，GPU已释放，未推生产。
