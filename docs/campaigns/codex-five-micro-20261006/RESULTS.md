# 五方向微测试准备与 leader 实机结果

日期：2026-10-06。五个原生 executor 子代理已交付，leader 串行执行 GPU 队列。原源码、冻结 bundle-v1、模型、生产和硬件配置未修改；所有微测试服务已退出。

## 交接状态

|方向|已交付与验证|仍待集成的内容|
|---|---|---|
|01 多流 MTP|CPU 独立布局参考7/7；真实2/4路1K/64请求、slot并发/结束idle检查通过|GPU carry/chain/FIFO仍限单seq；未完成多流GPU链移植|
|02 双微批|CPU延迟/快照/reset、两GPU CUDA Graph事件与输出通过；真实GGML同UID切src回归已复现旧地址|把不可变frame图/UID及工作区依赖接入真实模型，保持数值与缓存契约|
|03 PLE缓存|真实IQ4_NL行/hash/GET_ROWS bit一致；原序/页合并/预取/受限LRU/fallback字节一致；实际多流运行中已开启已有CPU cache|预算化缓存/预取的端到端收益与真实负载稳定性，GPU行缓存未实现|
|04 K8V4|实际GGML CPU Hadamard与Q8/Q4 cache写读通过；Q8K/Q4V2/4路短请求成功|真实多流质量、同格式可重复性、注意力dispatch和大上下文峰值|
|05 DWDP|七种主专家格式扫描；真实4专家12片7,475,200B按owner重组一致；岛内2/3GPU peer双缓冲checksum通过|完整expert计算/布局与稠密/状态/预取账，未实现整模型DWDP|

每个lane的 `PLAN.md`、`MICROTESTS.json`、fixtures、CPU结果和rollback已保留。`execute_micro.py` 是 leader 独占队列；它使用文件锁、空闲GPU检查、58°C/512MiB余量、独占transient user unit、超时与finally释放。

## 1. 多流实机：确实同时处理2/4路

配置：bundle-v1、真实UUID两岛、图内P2P、CUDA Graph、四头fused indexer、MTP4；ctx65536、UB256、slots4、QSA pooled multi、CPU PLE row-cache，`LLAMA_SPEC_PAD_MULTI=24`。不同流取冻结真实语料的非重叠1,024-token窗口，每流64输出token。

GPU MTP carry/chain/embedding启动标记为0，符合单seq源码guard；这不是“多流也125 tok/s”。MTP普通投机实际有draft/accept数。请求长度及 `/slots` 2/2、4/4活跃采样均通过，结束均idle。

|KV格式/波次|每流server decode tok/s|每流TTFT s|含prefill的合计输出tok/s|总接受率|
|---|---|---|---:|---:|
|Q8 K/V，2路|26.44 / 11.71|7.28 / 4.19|13.25|53.16%|
|Q8 K/V，4路|18.67 / 21.03 / 10.70 / 6.16|10.72 / 10.72 / 8.91 / 4.22|17.30|42.01%|
|Q8K/Q4V，2路|29.52 / 25.65|6.91 / 6.91|13.67|50.30%|
|Q8K/Q4V，4路|17.44 / 20.36 / 5.83 / 10.47|10.54 / 10.54 / 3.69 / 8.79|17.28|44.79%|

这些是64-token启动微波，受图捕获、prefill/生成交错、请求次序与页缓存影响，不是稳态生产容量/吞吐。server每路decode计时也含其他请求造成的等待。合计值用所有输出/wave墙钟，包含prefill；不是纯decode合计。

harness原字段 `aggregate_tg_tokens_per_second` 命名不准确，现已改为 `aggregate_output_tps_e2e` 并保留deprecated别名；新成功流保存完整token IDs和request/first-token/end时间戳。第一波Q8结果只保留ID数和SHA，不能逐位置展开；与Q4的同输入输出hash比较仍可进行。

两格式的6条对应输出hash全部不同，**不能延用重复文本“Q4无翻转”的结论，也没有证明质量下降的原因唯一是量化**。需要同格式重复/任务质量验收；当前只说明接口、并发与cache格式运行成功。

pad日志使用DEBUG5，而当前wrapper为verbosity4，marker=0只能表示未观测，不能证明pad关闭/开启；下一次验证需DEBUG日志。Q4 cache字段不等于`q8_0-q4_0`融合FA kernel激活；winner的FA_QUANTS没有该pair，dispatch仍待独立采集。

两配置最高GPU/主机传感器42°C，采样最小单卡空余分别4,735/4,743MiB；无thermal/memory abort。测试已停止，六卡释放。

## 2. 图UID问题：真实GGML GPU小图的具体失效机制

冻结winner CUDA backend，两个F32 ADD/MUL节点，真实独立A/B GPU buffers；先eager/capture/replay，再在同graph UID下把src切B。

- 同UID：**仍读A，未读B**，已复现。
- 新UID控制与独立gf控制：都正确读B。
- 实际host计数eager3/capture3/replay4；capture与replay均发生。
- 小图首次build因公共header只forward-declare ggml_cgraph失败；leader加入private `ggml-impl.h`/include目录后成功，初始失败日志保留。

这支持“旧实验原地重绑指针而不改变graph UID，违背捕获图地址稳定性约定”的具体风险。它不是证明winner正常路径都坏了，也尚未独自解释先前整模型token37分岔。

独立双frame CUDA fixture提前分配/capture全部资源，然后异步提交；只有分别查询GPU事件、A2/B1的start已完成且两个done未完成时才报告overlap。真实事件观察为true；输入发布后变异、peer传递、独立work/view、64→97新epoch输出均通过。没有把小图重叠换算成模型提速。

## 3. PLE：字节工作集很小，页I/O放大明显

真实表IQ4_NL `[160,320001536]`，每行90B；绝对文件偏移47,039,862,208。微样本416次gather、232 unique、184 duplicate，6行跨64KiB页，含末head尾部有效行。

源码hash/ngram与真实metadata吻合，GGML CPU GET_ROWS与IQ4_NL参考解量化逐bit一致；取消/超预算均回退原序读取。保守总I/O57,502,864B <128MiB；cache charge peak35,728B，maxRSS62,464KiB。

小样本GET_ROWS76.11ms、随后已受页缓存影响的原序pread0.88ms、合并页3.54ms、prefetch+合并2.58ms、两遍LRU0.44ms。数据量/顺序/缓存状态不同，不能宣称合并页一定更快。

已有 `GGML_CPU_PLE_ROW_CACHE=1` 真正接入lazy IQ4_NL CPU GET_ROWS，默认关闭；固定槽数据32MiB，含tag/age约36MiB。多流实机退出日志实际出现cache统计：Q8 hits61640/direct46014，Q4 hits62259/direct46137，fallback0。配套DLL映射锁定bundle的0.25.1；无LD路径的裸ldd落到/usr/local不代表实际runner。

## 4. KV与专家局部验证

K8V4 CPU fixture调用实际GGML的量化/反量化、Hadamard matmul、set_rows/get_rows。合成样本相对RMSE：Q8≈0.00868，Q4≈0.12568；这不是模型质量分数。

主48层确认七种专家量化；Q8_0位于追加MTP层单列。512专家三分为171/171/170；四个真实expert的gate/up/down完整bytes按owner分发并重组一致。24/24账本是假设下界，不能替代runtime逐卡峰值。

SM70岛内peer单测每次8MiB，8轮：单peer copy约0.178–0.183ms，双peer gather约0.218–0.225ms；有限compute窗口约4.67ms，checksum/事件依赖通过，copy可被该窗口覆盖。两slot全部数据与checksum scratch<64MiB。serial记录与pipeline总墙钟的迭代范围不同，**不计算吞吐提升比例**；不外推MoE/整模型收益。

## 队列交接与下一项

1. 先把图UID/地址稳定性micro约定用于私有流水原型，验证真实请求历史、状态reset与工作区依赖。
2. 多流GPU MTP链需要真正泛化token/hidden ownership与图shape；当前CPU参考fixture不等于实现。
3. 保留已有PLE缓存组合候选，后续数据衡量实际冷/热收益与缓存冲突。
4. K8V4不因request smoke就接受为无损生产设置；补同格式重复及实际dispatch。
5. DWDP只有raw权重/传输局部证据，下一层预取何时能覆盖本模型compute仍待计算层micro。

原始实机数据均在 `runs/`。GPU队列已清空，未部署新生产服务、未重启、未做长压测。

## Phase 2: parallel integration and microverification

| Lane | Actual result | Decision |
|---|---|---|
| Multi MTP carry | Configured and active markers present; real 2/4 streams completed | Default remains off: no gain and parity failed |
| Overlap / graph | Same UID source switch now reads fresh B; eager/capture/replay 3/3/4 | Retain opt-in pointer validation; full double-frame pipeline still unimplemented |
| PLE | Fresh private CPU DSO, 4-thread GET_ROWS exact float bits, 416 occurrences/232 unique | Retain bounded candidate; no separate end-to-end gain claimed |
| K8V4 | Native Q8K/Q4V vector dispatch compiled and passed; F16 conversion removed in fixture | Optional candidate only; QSA sparse Q4V remains unsupported |
| DWDP | Real 1.306MB expert P2P then gate/up/down compute bitwise exact | Two quantization types proved; full model and all seven formats not yet integrated |

### Model microtests

Real disjoint 1K prompts; 4 configured slots; Q8 K/V; ctx65536; UB256; 128 output tokens, 2 rounds. E2E aggregate includes prefill. DEBUG5 is shared by both profiles.

| Warm round | Control decode per flow (tok/s) | Carry decode per flow | E2E aggregate control -> carry |
|---|---|---|
| 2 flows | 31.81, 34.19 | 30.93, 27.14 | 39.40 -> 35.77 |
| 4 flows | 21.70, 18.00, 12.28, 13.30 | 13.19, 21.37, 10.22, 12.25 | 39.71 -> 34.03 |

Carry is rejected for default use. The 2-flow wave in a 4-slot context does not activate its dense 4-row guard. The 4-flow activation marker is real, but this is hidden carry, not the complete scalar GPU token-chain port.

The frozen V1 short control also has repeat discrepancies: first differences at 60/26 in 2 flows and 12/1/36/none in 4 flows (64-token outputs). Thus a mismatch cannot be uniquely attributed to carry, PLE dedup or UID validation. Investigate multistream determinism before production promotion.

The old CPU helper result was not an integrated cache-on proof because that fixture unset the gate. The current 03 cache-on fixture explicitly sets the gate before GET_ROWS, skips the manual cache helper, and requires one DSO activation and one teardown log. Its direct GET_ROWS float-bit check passes. Raw-byte equality belongs to the separate historical helper.

The first three GPU operator tests shared another model and are correctness-only; their durations are not optimization wins. The native mixed-vector fixture ran after the model queue was released.

See PHASE2-SUMMARY.json for exact run paths, measured temperature/headroom, library hashes and raw timing samples. No normal service, live source, firmware or hardware settings were changed by this lane.
