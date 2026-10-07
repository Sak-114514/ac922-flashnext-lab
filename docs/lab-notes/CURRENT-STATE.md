# Active experiment checkpoint — 2026-10-01

User explicitly canceled no-MTP comparisons on2026-10-01. Remaining matrix is7MTP cases: concurrency1/2/4 at8192/65536 and single131072;512output per flow. Diagnose slow prefill, then run bounded UB256 probe with decode retention. Do not claim no-MTP measured.

## Latest speed-first steering and live job

User then rejected repeated baseline/control-variable tests and requested all plausibly faster settings together. Original7MTP cases have completed and `audit_matrix.py --mtp-only` independently passes with zeroissues. No-MTP canceled. Frozen old scheduler retired; MTP-only finish job completed; UB256 probe passed exact512-ID parity: PP367.77(+8.2%),TG75.99(vs76.16).

Current job `ac922-flashnext-speed-bundle-single.service` PID27266, model `ac922-flashnext-test.service` PID27271 loading. Script `scripts/speed_bundle.py --profile single`: CPU PLE, MTP4/PAD6, targetUB512/b2048,draftUB32, HC_FUSED1,GDNx4, NCCLautotune(no forced LL/channels2/buf1MiB),NUMA0 CPU/memory binding, HTTPthreads8, existinggraphs/GPUargmax/Q8KV enabled. New numeric-validated Q8FA remains off because earlier measured slower. Runs optimized1×8K then1×64K,512output. Observe these handles, not retiredsuite23479. Logs speed-bundle-single.log / speed-bundle-single-runner.log.

Source rebuilt successfully after optional LLAMA_STEP_TIMING_PERIOD(default200,candidate32,tokensmin/max) and LLAMA_QSA_GATHER_MAX_TOKENS(default16,candidate32)+pathtrace. Gather retainsallshape/sequence-maskguards; allows4×MTP4=20verifytokens ifeligible. Do not removepooled-cache single-sequencegate. Actual multi ordinarygather eligibilitymustbetraced: unifiedmemory hasstream1 evenwith4slots; priorclaimmultialwaysdensewaswrong. Existingctxsynchronize alreadyhas`if(synced)return`, so repeatedsamplercalls do notproverepeatedbackendCUDA sync.

Next multi bundle is prepared in speed_bundle.py --profile multi, notyetdispatched:4slots,shared409600,individualcap132096,targetUB256/b2048,MTP4/PAD6,samefastflags;cases2/4×8Kand2×64K. Poolincreasefrom266240 to409600 adds~614MiB/cache perlimitingGPU, compute/graphgrowthadditional; notyetloadedorvalidated. Use singlespeedactualresults before decidingmulti nextwork. User wantsfastestbundledtuning; noautomaticrestartofoldunoptimizedornomtp profiles.

## Live state

- Old suite parent PID23479 is deliberately SIGSTOPped to prevent its already-loaded no-MTP profiles from launching. Model23482 and matrix_runner23483 keep running. New `ac922-flashnext-mtp-only-finish.service` PID25834 waits for actual current MTP completion, verifies7casefiles, retires frozen parent withSIGKILL (without resuming it), stops test model, writes7-case completion marker, then runs the bounded UB256 probe. Observe this specific new job; do not resume frozen parent.
- Current profile `final-mtp-multi-fair`: completed2/4×8192 and2×65536; current4×65536. Authoritative `manifests/final-mtp-multi-fair/state.json`.
- No-MTP profiles are canceled and will not launch. Final marker records explicit cancellation. Marker alone does not prove completion; use audit_matrix.py --mtp-only after pulling results.
- Logs `logs/final-suite-fair.log`, `logs/final-mtp-multi-fair.log`; profile telemetry `gpu-telemetry.jsonl`. Old suite traceback exit5 is a prior failed start, fixed before the currently running process.
- Keep experiments running during read-only prefill diagnosis, explicitly requested by user. Do not restart based on observation timeout.

## Selected profile

CPU PLE, complete IQ3_XXS model/shared Q8 MTP head; MTP3/PAD5, target UB128/b1024, draft UB32, graphs enabled. GPU order0,4,2,1,3,5, tensor split1×6, NCCL LL/channels2/buffer1MiB. Optional new Q8 FA/GDN kernels off: combined short test lowered decode76.2→72.5. Single shared ctx132096; multi shared ctx266240/four slots/individual cap132096. This is configured capacity, not a proved maximum.

Fair-share prompt quota enabled for multi. 2/4×8K exactly match512 output IDs of quota-off references. Preserve pooled QSA indexer gate n_seq_max==1: its fast selector lacks multi-sequence visibility.

| Completed selected MTP case | Prefill per flow t/s | Decode per flow t/s |
| --- | --- | --- |
| 1×8192 |339.94|76.16|
| 1×65536 |341.67|68.94|
| 1×131072 |328.98|63.90|
| 2×8192 |169.07 /169.06|37.41 /37.88|
| 4×8192 |86.65 /86.66 /86.67 /94.96|18.26 /18.14 /18.26 /13.42|
| 2×65536 |124.18 /124.19|17.34 /17.49|
| 4×65536 |52.23 /52.37 /52.37 /52.23|6.41 /5.85 /5.75 /6.41|

Exact input/512-output counts passed. Rates are per-flow server timings, not wave-wall throughput.2×64K also matches512 IDs per flow against the quota-off reference.4×64K is still running.

## Safety and paths

Use aliases ac922/ac922-bmc only. Old coding/ACE-Step/API gateway stopped. Preserve BMC boot guard/watchdog and inactive stock monitor/control. No unmask, reboot, power-cycle, firmware/IOMMU/memory-guard changes. Local observer session41441 and host telemetry stop only test unit at58C. Latest temperatures47–49C, minimum GPU free3120MiB.

Remote root `/home/debian/ac922-llama-next`; model `/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`78615622080bytes; all1256 tensors/source hashes verified. Main source v0.5.0/7fe450e + pentacoxian fixed v3 + bounded local PLE/SM70/fairness changes. Separate Jusko fork d343373 built but lacks Qwen4Exp MTP/NextN. Backups `artifacts/v100-v3-baseline`, `artifacts/pre-sm70-kernels`.

Stop test unit before relinking in-use libraries. Collected missing unit's Result=success is not success evidence. GPU-PLE132K MTP failed graph instantiation with184/192MiB limiting-card headroom: aggregate capacity cannot satisfy per-device allocation. CPU PLE preserves the complete model, with better measured MTP speed and headroom.

## Next work

Let full14-case matrix finish. Independent prefill diagnosis in PREFILL-DIAGNOSIS.md. New read-only audit_matrix.py covers all14 cases; its five tests pass and the actual6completed files pass, with eight missing cases keeping status PENDING/exit2. Outputs final-results/MATRIX-AUDIT.md and matrix-audit.json are checkpoints, not final reports. Direct selected-profile memory evidence is final-results/MEMORY-AUDIT.md with copied startup logs.

After matrix, a bounded UB256 probe is prepared in scripts/prefill_probe.py, not yet dispatched. It refuses while suite is active or complete marker absent, keeps all other settings/prompt fixed, verifies512-ID parity, records speed ratios, and stops only the test unit when done. Further UB512/protocol probes only with evidence and safe headroom. Pull/audit all case files and telemetry, report complete matrix and actual KV/indexer/recurrent/compute buffers. Do not mark complete from short measurements.

2026-10-01 S1唯一当前任务：停止组合调参，冻结532.63/78.85基线，使用artifacts/pre-prefill-source-changes；QSA四头融合/GET_ROWS/KV-pad均挂起。Nsight2023.4两次8192/512请求完成（A PP512.64/TG55.83，B PP531.25/TG58.64，profiling开销），两份qdstrm均Type48乱序无法导入，四项占比暂无；2024.2临时工具未能维持Power目标进程，正试官方支持的2023.2.1工具。仅工具变化，不动BMC/风扇/驱动/内核，不自动开始S2；58C观察PID9681保留。
2026-10-01 S1完成｜唯一变动：任务目录采样工具改用 Nsight Systems 2023.2.1；冻结推理二进制与参数不变，两次1×8K/512。A(case_start→case_complete，含生成)25.810s：NCCL31.50%、可识别MoE GEMM17.74%、attention/QSA专用核2.56%、CUDA API launch+sync60.28%；B(n_decoded=1→完成)9.138s：25.47%/13.29%/2.28%/56.95%。判定：NCCL双窗口均≥25%，API覆盖率最高；S5待后续阶段先查岛内P2P/NCCL；S2等用户决定。测试服务已停、GPU显存0，58°C观察保持运行，BMC/风扇/驱动/内核均未改。详见final-results/S1/S1-RESULTS.md。

2026-10-02 GPT C14｜修正v13 Q-gate错误均分，恢复6144粒度与2个KV头整组对应（3成员合法[0,6144,6144]）；显式真实岛GPU顺序0,1,2,3,4,5。编译通过，修后1×8K在warmup命中SET_ROWS cache_k_l27镜像/axis0冲突，未产生速度或parity结果。静态定位KV/RS ctx_map按同名Meta()合并，C15直接修buffer身份，不弱化SET_ROWS检查。
2026-10-02 GPT C15｜四处buft comparator补同名对象身份排序（KV/RS/DSV4/loader），actual comparator编译回归4/4通过，增量构建通过。132096ctx启动/预检/warmup/HTTP就绪成功：主KV两份274.13MiB、RS101.12/89.62MiB、两岛NCCL各nranks3。1×8K实际prefill约12s发生host SIGSEGV，无有效PP/TG/parity；不放宽SET_ROWS、不回滚真实cache归属修复，临时test-only backtrace追本次新增执行故障。
2026-10-02 GPT C16｜split-state大描述符递归在POWER栈帧5488+19808B/层导致栈溢出（backtrace/objdump定位）；改显式后序栈+缓存命中快路，COMPUTE NONE叶不遍历allocation依赖，host-view延迟AR分析增加类型门。900层旧快照失效CPU数值回归=901通过，编译通过；132096ctx两岛启动成功，1×8K/512：PP633.7046（+18.97% vs532.63），TG99.9288（+26.73% vs78.85），512-ID parity=True。两岛NCCL仅rank3，原每层跨岛AllReduce已移除；仍有边界activation copy，未用nsys量化总同步次数。下一步仅加已验证H和短双路验收，不加入会禁用岛模式的E。
2026-10-02 GPT C17｜仅加入已验H GGML_CUDA_MOE_GROUPED=1：1×8K/512 PP749.7925（+40.77% vs532.63，+18.32% vsC16），TG100.0355（+26.87% vs78.85，vsC16+0.11%），512-ID parity=True。保留C+H；下一步短双路+QSA长度门（8K不激活B），并将快照失效挪到依赖遍历前，避免计算中清cache破坏后序保证。
2026-10-02 GPT C18/C19｜快照清cache提前；双路C+H+B长度门：PP299.7187/299.7110（各+32.4%），TG45.7447/46.3856（+33.7%/+31.0%），两路各512-ID parity=True。409600ctx/4slot启动通过，minfree1459MiB。额外debug fixture发现旧日志解析source会清正在写入的cache，C19改只读日志后，900层/1MiB栈/debug=1 数值901通过，最终构建通过。C19不重跑GPU（仅debug日志分支修复）。C+H单路/短双路整合完成；长C+B及四路满窗未验。证据final-results/GPT-ISLAND/RESULTS.md；测试停止/GPU释放，没有推生产。

2026-10-02T12:25:37+08:00 R14 A｜同一C+H基线，仅岛on/off，1×8K/512两组parity=True。直接STEP_TIMING target64采样步：岛开compute22.8795ms/步（reused22.0119），关22.7580（reused21.0712）；draft decode2.437 vs3.382ms/次。性能门<12ms未过且target未优于baseline，冻结性能承诺/保留结构代码，不跑nsys、不跑2×64K三合一。F未接线、G未部署旧结论已推翻；正在补现成适配接线各一次测试，失败即弃不修。
2026-10-02T12:29:31+08:00 R15 G｜部署现成Onecat XQA gate，编译成功，新标签单次1×8K/512。日志using opt-in Onecat确认选择；PP747.7879/TG22.4000，parity=False，MTP accepted0/2034=0%。按用户弃用，后续显式G=0，不修不重测。
2026-10-02T12:49:01+08:00 R15 F｜现成PLE卷积适配接线/CPU参考/CUDA/meta/模型门控，部署前修正history偏移和时间/通道转置。完整构建通过，5-slot CPU fixture通过，单次1×8K/512 PP748.7413/TG100.7669 parity=True，实际CUDA dispatch日志确认；vs本轮745.8584/101.5556为+0.39%/-0.78%，无明显收益，F可选默认off、G失败off。14+15结束；未跑nsys/长请求，无生产或硬件修改。

2026-10-02 R16 A｜1×8K/512 parityTrue，103步50.1101ms：target24.946/draft6.343/sync1.795/sample2.779/book0.435/wrapper13.653/server0.158；sampling+book3.214<10，进入B。wrapper未细分底层等待。GPU PLE H+E132K实际Eactive但graph_reserve CUDA1申请981.40MiB OOM，未发8K请求。B由主代理亲写SM70 cooperative原型：全路由/量化/FFN/归并一次launch，首组IQ2_XS+Q2_0 T5数值通过~113us vs旧150us；全7格式矩阵进行中，未集成模型。

2026-10-02 R16 B｜主代理完成SM70 cooperative单launch原型与模型decode≤32接线，prefill/不支持格式原路径。五gate+两down格式40例fixture通过，CUDA graph重放逐位一致；最终编译库192/224实际TP3分片20例也全过，maxNRMS约0.00177。完整模型gpt-r16-monomoe-20261002-145028-f633e3开始1×8K/512，验收parity+TG>=108.0432，未推生产。

2026-10-02 R16 B结论｜修扩展workspace登记与routing scale=0哨兵后，1×8K PP751.66/TG102.2761，512-ID parityTrue、MTP408/408；仅+2.235%未过8%门，默认关闭/未推生产。七格式40+20 fixture通过，graph replay逐位一致。三条后续局部优化更慢已否定。C无新路径launch/sync>50证据未触发。GPU PLE H+E132K graph reserve CUDA1额外981.40MiB OOM，失败已记录。

2026-10-02 R17 fixture｜两座三卡岛×3种尺寸×空分片开关=12例，四次串行归约、32次变输入运行，所有卡graph-ready且逐位正确；GPU P2P小测试通过，准备唯一1×8K整模门。

2026-10-02 R17 P2P初测｜仅取消P2P_AR_DISABLE，默认512节点分块，1×8K/512 parityTrue，PP758.58/TG102.70（对749.79/100.04为+1.17%/+2.66%）；target compute19.5665ms vs22.8795（-14.48%），draft2.426ms；小收益已实测，下一单变量取消分块。

2026-10-02 R17完整岛图｜仅GGML_META_CHUNK_NODES=0，1×8K/512 parityTrue，PP758.40/TG102.50，对锚点+1.15%/+2.46%；无额外端到端增益。开始新标签decode nsys，实证实际图内P2P覆盖和launch/sync门。

2026-10-02 R17裁决｜每GPU97个主图重放，岛0/1每图50/47次P2P；图内gap3.557%+communication9.879%=13.436%，CPU API57.039%不得混作图内。常驻执行器>50%门未满足，不立项；完整岛GPU图小收益配置保存，未推生产。证据ROUND17。

2026-10-02 R17收尾｜全部逐条温度记录复核：GPU/host最高均50C（45C仅采样结束快照，已纠正），低于58C；测试inactive，六卡0MiB/40–41C。条件第二阶段裁决不立项，未写常驻执行器；第一阶段配置/fixtures/完整图证据已保存。

2026-10-02 R18-A｜argmax改CUDA_Host锁页回读，1×8K/512 parityTrue，PP752.54/TG100.49（+0.45%），未达TG+5%门；源码已还原，补丁存ROUND18/argmax-rejected.patch，下一项GPU隐藏状态接力。加载期旧server映像时序问题单独记作无请求setup-invalid，不计测试。

2026-10-02 R18-B｜PAD6持久GPU隐藏输入+图内反馈，fixture36次跨图重建/重播/重新播种通过，1×8K/512 parityTrue，PP749.40/TG103.82（+3.78%），未达5%门，独立方案默认关闭/归档。下一项固定greedy草稿链去逐token回读，B代码仅作候选依赖，不作为已验收优化。

2026-10-02 R18-C启用修正｜首轮GPU链被backend_sampling请求值误挡（实际已fallback CPU），该轮只有GPU隐藏状态，不计GPU链收益。改检查实际backend_chains后新标签重跑；链异常RAII清理和argmax格式门已补。

2026-10-02 R18-C｜实际GPU token链已启用，128次队列fixture通过；1×8K/512输出parityTrue但draft接受260/999=26.0%，PP752.18/TG43.00，性能失败；B+C七个源码全部还原pre-R18，补丁归档，跳跨岛异步传输。不能把target输出一致当草稿链等价。

2026-10-02 R18-D｜双向/三尺寸/双槽四队列fixture通过；1×8K/512 parityTrue，PP805.82/TG103.39（+7.47%/+3.35%），未过prefill+10%门，默认off归档。下一项在此基础上加稳定GPU地址+锁页输入快照FIFO预填充流水，保持GPU同层顺序。

2026-10-02 R18-E局部门｜稳定GPU输入地址+两份锁页host快照+Meta复合事件；12批覆盖输入与两岛递归状态fixture在FIFO开/关均逐位通过，实际frames12/staged_inputs12/cross_copies12。构建已通过，开始唯一1×8K整模，固定prefill KV视图8192，decode/MTP保留旧路径。

2026-10-02 R18-E验收｜1×8K/512 parityTrue，PP966.51/TG102.31，对749.79/100.04锚点+28.90%/+2.27%，达到prefill+10%且TG退化<=3%门。实际frames15/staged_inputs375/cross_copies40；保存island-h-fifo-single opt-in，未推生产。增益为FIFO+KV_PAD8192组合，未分离padding收益/真实GPU重叠比例。

2026-10-02 R18完成｜五项闭环，唯FIFO组合达标并保留opt-in；A/B/C失败源码移除，D仅作为E依赖。四个保留源码本地/remote SHA一致；GPUmax50/hostmax49，当前测试inactive/六卡0MiB，无生产或硬件变更。最终总表ROUND18/RESULTS.md，配置island-h-fifo-single。

2026-10-02 R19开始｜用户放开架构/组合/实验门槛；冻结R18运行库并核对进程映像。FIFO+P2P全图1×8K parityTrue，PP939.90/TG104.42，保留取舍候选；继续组合MonoMoE。MTP坏链改为逐步token/hidden追踪，不再仅凭target最终parity。

2026-10-02 R19输入依赖修复｜GPU_INPUT flagged tensor复制到CPU时只同步dest，buffer_get使用cudaStreamPerThread而producer在backend stream。实际回归fixture旧库4/4读旧值0，新库4/4读期望7/8/9/10。已补device input producer同步；MTP链整模与逐步token/hidden审计进行中，尚不宣称接受率恢复。

2026-10-02 R19链修复验证｜producer依赖修复后MTP接受408/408，512-ID parityTrue，审计GPUargmax==CPUargmax==feedback，h误差0，PP964.41/TG107.05（带前4链审计）。GPU驻留IQ3_S embedding260.5MiB/member+MTP FIFO后，PP960.79/TG111.07，parityTrue；真实语料对照待做，目标仍active。

2026-10-02 R19真实语料警报｜冻结R18 FIFO同一进程同tokenSHA/temp0/seed1234/cache_promptfalse两次输出第4token分岔；cold207.89/50.42 vswarm1076.06/42.12。8K+512暖波增556MB读盘/8447major faults，64KB页，CPU32GB均在node0(node8无RAM)，SATA BX500 SSD。已保存warning；正在关闭FIFO保持其余相同检查重复性，尚不能用此不稳oracle判断真实提速。

2026-10-02 R19请求重置｜FIFOoff冻结oracle仍同输入重复输出第4token分岔（205.08/50.90 vs1038.02/57.02）。源码确认MTP pending_h仅构造时清0，begin在prefill后调用，前一请求carry会污染新prompt首次draftcatchup；已加process中pos0按seq重置pending_h/verify/deferred/draft_pending，构建中，重复性尚待验证。

2026-10-02 R19重置实测｜carry-reset日志首波0/次波1；real8K256输出首次分岔由index3推迟index91，未完全消除，CPU链off205.43/45.36冷 vs1086.31/58.47暖。修复只证明移除残留carry，继续查剩余状态/数值路径。

2026-10-02 R19真实GPU链｜8K256两波207.58/54.82与1057.89/48.65，接受180/297与172/330，输出index2分岔，不能确认真实提速。仅重复回归111.07有效。已发现QSA radix top-k以atomic到达顺序选边界同分/排列，正在实现稳定选择fixture，不直接归因为已证实原因。独立MTP新布局保持目标3+3，GPU0预算13.28/15.47GiB，加载中。

2026-10-02 R19独立MTP｜目标仍3+3，draft仅CUDA0，GPU embedding一份260.5MiB，无draft Meta/NCCL。1×8K512 PP1045.87/TG87.29，parityTrue/408接受全中，draft compute约3.7ms/次，高于TP3。当前不选用，下一版仅裁剪TP3-MTP未使用的第一岛backend，保留draft TP3。

2026-10-02 R19稳定topk局部门｜root实现CUDA连续段计数+确定性gather，同分按最小index选入、输出顺序固定。27形状+2对抗×12变输入graph replay全通过；8K×6调用约0.06ms。GGML_CUDA_TOPK_STABLE默认off。下一次真实8K重复测检查目标重复性；尚未将原因定论。

2026-10-02 R19真实确定性确认｜stable topk+GPU链+裁剪未用MTP Meta，两次同prompt8K256输出逐token一致、接受均182/288；PP206.71/1038.65，TG59.38/83.03冷/暖。因同时裁剪backend，不把整体性能增益单独归因topk；此前原子无序topk风险及新fixture有独立证据。当前anon2.263GiB/shmem0.781GiB，图元数据收缩待验。

2026-10-02 R19 compact闭环｜27+2 stable topk fixture之后真实8K重复逐token一致，regression512 PP1055.16/TG113.13 parityTrue。compact arena+NCCL WARN真实8K256 PP206.77/1038.82、TG59.54/82.99，接受同182/288，匹配stable；暖波read_bytes增0。Anon由2373120降2331008kB，约40MiB，非GiB。正在测试packed split-cache以减少更大的heap驻留。

2026-10-02 R19内存尾项｜packed真实8K两波同gold/182接受，PP207.03/1040.41、TG59.60/82.47；Anon2293440kB，较compact另省约37MiB。malloc_trim再省约34MiB但TG无改善(58.77/82.68)，hook和失败单GPU MTP布局已归档回退；保持正确链、确定性topk、可选compact/packed，转入独立六卡GPU PLE新布局。

2026-10-02 R20独立PLE工程｜主模型model.devices仍两岛；新增只承载IQ4_NL PLE表的6成员Meta，行范围均匀拆分、单独callback，target context注册/MTP不注册。loader额外buffer type、联合per-card权重预检2GiB余量、island成员数guard区分PLE。发现单shard4.47GiB不能沿用cudaMemcpy2Dpitch，Meta单连续块改tensor_set；正在构建/六卡gather fixture，尚未整模运行。

2026-10-02 R20 fixture确认｜IQ4_NL160宽、6卡均匀行分片、4gather尺寸×12次变索引，全部F32逐位等于CPU解码。修复测试中的静态/计算buffer区分后通过；提前发现NCCL大tensor默认BF16会损失row-gather精度，bounded GET_ROWS归约改F32保真。开始132096ctx/layer boundary28整模加载，尚无速度/整模parity。

2026-10-02 R20首轮整模｜联合GPU权重预算12.46–12.84GiB/card通过；132096ctx全GPU加载并READY，但首8K prefill CUDA3 cuMemCreate OOM，无输出/速度结论。Loader MAP_POPULATE先整读73GB使启动约5m35，准备prefetch0。下一轮UB256/boundary29、共享物理GPU直接复制与F32 PLE归并，保持132K。

2026-10-02 R20真实张量裁决｜同B29/UB256 CPU对照PP288.70/955.11、TG57.57/75.22；GPU真实966.07/968.48、72.15/71.67，重复稳定但CPU/GPU第71token不同，首logprob差max0.243。1K局部审计：实际PLE lookup655360、key2621440、value655360浮点数CPU/GPU全部逐位一致。GPU PLE暂不提升正式选择，差异在后续图/数值路径；继续CPU原图精确行缓存方向，减少64KiB页读放大。

2026-10-02 R21 I/O简化优先｜源码确认CPU PLE现预设GET_ROWS_THREADS未设，实际默认1，n_rows<512也强制1。先用已有32路prefill读取并新增显式DECODE_THREADS=8（rows/4上限）保持原数值路径；准备hash fixture+真实8K对照。O_DIRECT行缓存暂未实现，按实测是否仍需要决定。R20诊断callback已归档撤掉，六卡PLE仍只作实验候选。

2026-10-02 用户转向｜当前行缓存只做最后一次真实8K256。若无明确收益，切到准备好的1Cat-vLLM基准+现有DFlash；已开始核对源码/拓扑，模型范围待澄清（现成27BTP3 vsFlashNext缺适配）。缓存fixture逐值通过并验证同地址不同inode失效，32/8读取前轮241.95/54.01冷、981.3/76.22暖，输出与CPUgold完全一致。

2026-10-02 R21最后一轮收尾｜encoded row cache(O_DIRECT4KiB miss+resident mmap+4way262k rows)fixture全值/同地址不同inode通过；真实8K256 PP360.0217/TG60.7285，178/305，256tokens与CPU_B29参考全同。相对32/8无缓存冷241.95/54.01有收益，但仍未解决低冷prefill。按用户转向归档，测试已停/GPU释放；不再扩展llama试验，转准备好的1Cat基准+DFlash，模型范围待明确。
