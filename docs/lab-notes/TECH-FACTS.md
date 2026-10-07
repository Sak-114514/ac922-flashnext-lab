# 技术事实台账（防幻觉档案）

> 规则：每条标注证据等级。**[M]** = 机器实测（manifests/zcode-* 有数据）；**[C]** = 本人代码级验证（行号/diff）；**[A]** = 代理声称、待验证；**[X]** = 已推翻（附推翻证据）。
> 维护：新结论先落此档再引用；发现旧条目错误必须显式推翻并记录证据，不得静默修改。

## 一、性能基线与瓶颈（全部 [M]）

- 冻结基线（speed-bundle，2026-10-01 复现两次误差 <1.5%）：单路 1×8K PP 532.6/TG 78.85；2×8K 226.4/34.2+35.4；2×64K 153.3+151.6/13.2+17.5；无 MTP 单路 43.1。
- nsys（S1）：NCCL 占整请求 31.5%/decode 窗口 25.5%；MoE GEMM（含混合核）24-29%；attention/QSA 专用核仅 2.3-2.6%；launch/sync API 覆盖 57-60%（与 GPU 重叠不可加和）。
- 单路 decode 23.2ms/步中目标计算仅 ~3.3ms；draft 头（1 层）单 forward 3.4-4.7ms ≈ 全模型 forward → 固定开销主导 [M]。
- 多路 QSA：单路每步 kvlen=512 vs 2×64K kvlen=148480（290×）[M，qsa-path trace]。
- BF16 matvec 家族 decode 累计 = 全部量化 matvec 之和 ×2 [M，S1 kernel 表]。
- 基准提示词为重复文本 → MTP 接受率 100% 是虚高上限 [C，benchmark_server.py:171]。

## 二、优化项状态

| 项 | 状态 | 证据 |
|---|---|---|
| H：MoE 死块消除 `GGML_CUDA_MOE_GROUPED=1` | **转正**：1×8K PP 586.2(+10.1%) parity=True；2×1K PP 435/369（+118/85%）；64K 多路 PP 175.2 且 decode matvec 沾光 | [M] h-moe-1x8k / h-moe-2x1k / bh-2x64k |
| B：QSA 快路径 `LLAMA_QSA_POOLED_MULTI=1` | **转正（限长上下文）**：单独 2×64K TG 25.5/27.9(+93/59%)；**B+H 合体 38.7/39.3，总吞吐 30.7→77.9(+154%) parity=True** | [M] b3-2x64k / bh-2x64k |
| B 的 8K 劣势（20.9/21.3 vs 34.2） | 未解；round-5 假设=接受率混淆（拟合负斜率证明非纯延迟） | [A]（B round4/5 设计），夜间实现中 |
| C：岛内分组 `LLAMA_ISLAND_SPLIT=3,3` | 未通；已过 7 层（见下）；v7 修 GDN 镜像态但 SET_ROWS 断言层未跑完 | [M] 各轮 crash 签名 |
| E：草稿去 TP | 预检回退机制 [M]；6 路放不下（19.2>15.5GiB）[M]；不对称分割方案夜间实现中 | [A] |
| D：BF16→F16 | parity=True 但 PP+2~4%/TG-4~5%，冻结 | [M] d-f16 |
| F：PLE 卷积内核 | 已编译未测 | [A] |
| G：Flash-V100 XQA | 已编译未测；期望 ≤1% | [A] |

## 三、岛（C）的洋葱层记录（每层 [M] crash 签名）

1. v1：17.57GB 落单卡（成员分布）→ 2. v2 预检拦截 46.13GB=两岛全部（splits 推断被 meta 大显存骗成 [0,1]）→ 3. v3 ctx 同名合并（ctx_key_comparator 用 strcmp；46.13=A24.10+B25.44 精确）→ 4. v4 诊断化 → 5. v5 空张量 ne=0（我修：→MIRRORED）→ 6. v6 后 GDN src5=axis10(MIRRORED)（自报家门实测 [M]）→ 7. v7 修 GDN（镜像态全局头索引 head_offset 注入 op param slot1 [C]）→ 8. **SET_ROWS split_states_equal 断言**（[C] 静态根因：GDN 返回退化描述符每缓冲区和=0 vs CPY 传播的真实总和；6 路下该路径被 gdn→cpy 图融合隐藏）。我的修复=split_states_compatible 通配符 [C，已应用待测]；C 评审中。
- 岛账本 [A]（C 的计算）：跨岛字节 decode −92%、同步 96→0、步时预估 23→9-11ms；显存每卡 12.1→8.5GB（A/B 岛 25.89+20.24 GiB，预检实测 [M]）；必须 CPU PLE（23GB 表 3 卡岛放不下）。

## 四、图缓存（重要推翻记录）

- [X] ~~"TTL 驱逐"~~：我最初的时间 TTL 补丁无效（graph-ttl-600s 测试持平 [M]）。
- [M→X] 容量硬编码 4 张 LRU（ggml-cuda.cu `c.entries.size() == 4`，用户提供证据 [C] 行号确认）→ 但 cap16 实测在 1K/8K 均无差异（cap16-2x1k/cap4-2x1k/cap16-2x8k [M]）→ **容量也非基线路径瓶颈**；B 路径的 81 次驱逐是确定性 prefill 形状序列 [M，B round4]。
- 我已加 `GGML_CUDA_GRAPH_VARIANTS_CAP` env（默认 4，行为不变）[C]。

## 五、1Cat 遗产评估（[C]+[A]）

- 可用：TurboQuant SM70 经验、ppc64le 构建工装、实测带宽（岛内 peer 37-38GB/s 单路/39 聚合 [M，历史文档]）。
- [X] 不可用：FP8 KV（V100 负优化实测）、Remote KV sidecar（5.31ms/层）、1Cat HC/GDN 内核（dense 几何硬编码+与 F16 方案互斥）。
- Flash-V100 attention = 1Cat 自制 sm70 内核（非上游 FA2；FA2 需 SM80+）；XQA decode 已移植 [A] 待测。

## 六、待查疑点（审核队列）

1. [已清] `mmq-moe-grouped.cuh` mtime 00:18 = H 自己的收尾编辑（早于重复代理 00:19-00:21 窗口）；.cu↔.cuh 符号引用一致（switch_type/tile_offsets 双向核对）[C 02:20]。附带确认：我的 split_states_compatible 补丁在 C 编辑后依然完好（3 处引用）[C]。
2. B round-4 的 `LLAMA_QSA_F16_GATHER`（默认开）从未单独验证——它进了 bh-2x64k 的 binary [M 间接]。
3. [部分厘清 02:25] parity 语义 [C，matrix_runner.py:99-107]：**512 个 output_token_ids 逐位比较**，不匹配→runner 状态 failed、退出码 2（但 case 文件与 parity.json 已落盘）。因此 bh-2x64k 的 parity=True 是强证据（B+H 与慢路径参考输出完全一致）。矛盾解释排序：(a) 最优——b3-2x64k 的 False 源自 round-3 缺陷（n_touch 未 pad 等），round-4 修复后 64K 已收敛到参考；(b) 次优——F16 gather 恰好抹平了块/cell 选择差（机制存疑）。**关键空白：round-4 二进制的 8K parity 从未测过**（b3-2x8k 是 round-3 二进制）。明早测试矩阵第一项：round-4/cell-sel 二进制跑 2×8K 看 parity 是否直接回 True。
4. boot-guard 01:55:57 死因（mask 超时 + target readback mismatch，与岛测试 GDN abort 同秒）——因果未证。

## 七、夜间迭代审核日志

### 2026-10-02 03:0x — C 夜间迭代 1-3（v8/v9）：**通过静态审核**

- 验证项 [C]：GDN 真实段结构在 meta.cpp:953-966（`ne[j]=ne[0]×heads_j/H_all` + 行对齐 abort）；SET_ROWS 陈旧缓存 WARN 在 1588-1591；concat 轴规则在 600+；`cpy.cu:473` nbytes 断言亲验属实（混合 CPY 若到内核=响亮崩溃非静默拷错）；我的 compatible() 保留为轴门（3 处）；括号平衡；.v7b-user 备份保存了我的补丁版本。
- **推翻记录（对不起了，又是我）**：我的通配符补丁被 C 证明有两洞——(a) 崩溃会搬到 ratio-takeover 块（~1228 行）的一致性断言；(b) 更危险：退化 src0 → 每设备切片为空 → set_rows **静默不写 → 缓存陈旧的隐性数值错误**。C 的"真实段结构 + takeover 零和惰性化 + 陈旧检测 WARN"三层修复替代。**教训 #2：放行型补丁必须追查被放行数据在下游每个消费点的行为，"不崩"≠"对"。**
- 上游交叉 [A→C 部分]：set_rows 上游为 ids 直驱、无设备感知/越界防护（语义论断支撑我的原始论证）；**meta/split-state 机制 100% fork 特有**（7fe450e 基线无等价物）——上游没有裁判，只能内部一致性裁决。
- takeover 惰性化的"独立程序验证"声明 [A]（未亲验其测试程序）。
- 晨测验收门：岛 3,3 1×8K，新日志三特征（岛指针互异/预检两行/无 GDN·SET_ROWS WARN）→ parity 逐位 → TG ≥2×。

### 2026-10-02 03:2x — E 夜间迭代 v2（不对称分割）：**通过静态审核**，夜班收官

- 验证项 [C]：TRUNK_W 解析+范围校验（llama.cpp:269-274）；逐卡预检公式与回退（llama-model.cpp:1816-1857，账本：草稿卡 12.16GiB✓/其余 14.87✓ vs v1 的 19.22✗）；岛互斥实现+注释（llama.cpp:234-236，DETP 优先折叠单岛）；与 C 补丁同文件共存（ISLAND×4/DETP×110 hunks 并存）；三文件括号平衡。
- rotation 抵消论断（条目 k ≡ 物理卡 k）[C-partial]：注释与代码在位（llama-model.cpp:928 注释、E 自查纠正过一版错误实现），完整代数重推导未做——**明早用逐卡内存实测数字裁决**（若权重落错卡，预检数字会直接暴露）。
- E 的自我推翻记录：首版按 scan 下标加权踩中 rotation 陷阱，自查纠正为按条目（=物理卡）下标。
- 上游对照 [A]：splits ∝ tensor_split[k]/Σ 与 master 一致（联网核对，URL 未亲验）。

### 夜班总评（03:2x）

三份交付全部通过静态审核，**每份都包含至少一次对错误（我的规格/我的补丁/它自己的首版）的健康推翻**——防幻觉循环有效。剩余技术点全部属于"需机器裁决"类，已在明早测试矩阵中排队：岛 v8/v9 parity+TG、8K 三开关 parity、E-asym 激活、F/G 开关、boot-guard 因果、bh/b3 parity 矛盾终裁。代码侧无已知未理清点。

### 2026-10-02 02:4x — B round-5（cell 选块 + 长度门）：**通过静态审核**

- 验证项（全部 [C] 本人核对）：CELLSEL/MIN_CTX 开关存在（hybrid-idx/qwen4exp 共 8 处）；**关键布局修正属实**——qwen4exp.cpp:2092-2095 用 `reshape_3d(1,n_rows,nt)` + `new_tensor_3d(r,n_rows,nt)` + repeat，flat 下标 = row·r+j，与 row_cells 平铺同空间；旧 reshape3d/permute/cont 链 0 残留；三文件括号平衡独立复核；.fix4 备份在。
- **推翻记录**：B 推翻了我 round-5 任务书里的 "repeat ne1=r" 规格——该布局 flat 下标 = row+j·n_rows，cell id 错序。B 的替代方案经验证正确。**教训：我给的规格也会错，布局类细节必须过下标代数。**
- 上游交叉声明 [A]：top_k= argsort(DESC) 前 k、tie 顺序实现定义（与我对 ggml 的认知一致，URL 未亲验）。
- **测试矩阵修订（重要）**：2×8192 总量 16384 < 默认 MIN_CTX=32768 → 会被长度门 demote 成慢路径（parity 虚假全对）。明早 8K 测试必须三开关同开：`LLAMA_QSA_POOLED_MULTI=1 LLAMA_QSA_MULTI_CELLSEL=1 LLAMA_QSA_MULTI_MIN_CTX=0`。
- 验收门（明早）：2×8K parity 应回 True（或 tie-only 差异）且 TG ≥34；64K 生产路径不受门影响（131072≥32768）。


- （待各代理交付后逐轮追加：交付内容 → 我的验证方式 → 结论/推翻）

## 八、机器裁决轮（晨间）

### 2026-10-02 09:5x — 裁决轮 1：夜班 meta.cpp 改动两处被机器推翻

- **运维大发现 [M]**：主机重启后 **vLLM 生产服务（ac922-coding-api，TP3+FLASH_V100+FP8）自动复活**，占走 GPU0-2 各 15.6GB 并诱发 09:16 主机冻结（journal 戛然而止于我 scp 开始时刻；与 PHB4/SATA 病史吻合）。停服后 GPU 清零、机器恢复稳定。**昨晚的稳定窗口正是建立在这些服务被 GPT 停掉的基础上**——测试前必须确认它们没复活。apigateway uvicorn 会重生，需正式 disable（留给用户决定）。
- **推翻 C v8 之一 [M]**：默认 6 路（无岛无 DETP env）在 meta.cpp:854 `split_states_compatible` 断言崩溃——v8 的 GDN 真实段公式（heads_j 取自 src_ss[2]=x 输入的段结构）在 6 成员 meta 下算出的每成员 ne 与递归缓存段不等。**真实段修复激活了 6 路下的休眠路径**。静态审核盲区：6 路回归不在审核范围。已退回 C（v10）。
- **推翻 C v8 之二 [M]**：岛 free 聚合退化（46.41→15.95 GiB，GPU 全空复现）+ meta.cpp:2057 bufs[i] 空洞断言——成员簿记被 v8 惰性化改动破坏；llama-model.cpp:2161 的 meta free 查询返回单成员值。并入 v10。
- 开机教训 [M]：BMC HostState 在 degraded 态不可信（误报致我误杀过已启动主机一次）；RequestedPowerTransition 同值重置无边沿（须走 Host Transition.On）。

### 2026-10-02 10:4x — 裁决轮 2（v10 二进制）

- **E 不对称分割 ✅转正 [M]**：m-asym-1x8k = PP 573.9 / **TG 84.52（基线 78.85 的 +7.2%）**，parity=True。日志双确认："DETP active: draft layers 48..48 in full on CUDA0 (draft 2.58G + head 0.41G + trunk weight 0.5)"；draft-timing decode 3.4-4.7ms → **2.2-2.2ms（-35~50%）**。未达 ≥90 理想值（draft 仍剩 2.2ms 固定开销）但质量门全过。开关：`LLAMA_DRAFT_DETP=1 LLAMA_DRAFT_DETP_TRUNK_W=0.5`。
- **B cell-sel ❌被推翻 [M]**：m-cellsel-2x8k（三开关含 MIN_CTX=0）= TG 21.45/16.88、**parity=False**——round-5 预测（parity 回 True/tie-only + TG≥34）双双未兑现。8K 差距根因不是块/cell 选择粒度。B 的下一步假设需要重新排序（候选：接受率混淆的直接验证 E2 仍未做、或长度门作为唯一产品化方案——64K 已验证的 +74% 不受影响）。
- **岛 v11 [进行中]**：插桩数值 `dst=MIRRORED[0,0,0] vs values=AXIS0[0,256,256]`（成员0空段 + dst 镜像簿记存疑）已交 C。
- 流程修正：日志追加污染导致旧崩溃行误读了一次（854/861 实为 v10 前残留）——已换新标签新日志取数；教训入档：**同标签重跑必须清 log**。

### 2026-10-02 10:5x — 裁决轮 3（v11 命名证据）

- 岛 v11 的命名 abort 一次命中：**`SET_ROWS cache_k_l27`——岛内层（非边界层）的 K 缓存视图**，dst 自身 sums=[0,256,256]（成员0空），values 为 MIRRORED 退化。与上轮 values 侧 [0,256,256] 同签名 → **{0,2,2} 每成员头分布污染源在缓存视图的 split-state 推导上游**（非 GDN 传播、非惰性化——后者已被 C 静态排除）。
- 已交 C v12：在污染产生点加断言（任何轴切分状态若成员和 0/非0 混合 → abort 带链路），一步定位源头。
- 洋葱层级更新：v1 分布→v2 边界→v3 ctx合并→v4 诊断→v5 空张量→v6 CPY→v7 GDN镜像→v8/v10 真实段+分流→v11 命名化→**v12 = {0,2,2} 污染源头（进行中）**。

### 2026-10-02 11:0x — 裁决轮 4（v12 产生点命中 + 假设反转）

- v12 产生点断言命中：**`blk.3.attn_q.weight` axis=1 sums=[0,6144,6144]**——污染源头在权重张量的 split-state 构造（加载侧），非任何计算 handler。
- **[X] 我的"C×E 回调交互"假设被自己的插桩推翻**（10 分钟）：回调首调 `tensor_split=[1,1,1] detp_scale=0 trunk_w=1.0` 完全干净。教训 #3：先插桩再归因，两补丁同函数≠交互 bug。
- 数学线索：12288/2=6144 → 边界 [0,6144,12288] 疑似某处按 2 分割或 scan 首条目清零。已交 C v13（止损线第 2 轮，最后）。

### 2026-10-02 11:1x — 裁决轮 5（v13 终审）：岛冻结

- v13 粒度修复生效：v12 污染断言不再触发、{0,·,·} 家族签名整体消失 [M]。
- 但暴露新层：meta.cpp:1381 比例守恒断言（新签名，无名字）。
- **执行用户止损线：岛冻结**。全部改动 env 门控保留（`LLAMA_ISLAND_SPLIT`），默认路径无残留；现场与建议已固化到 GPT-HANDOVER.md。
- 战役最终记分：B+H（64K 多路 +154%）、H（PP +10%）、E（+7.2%）三项转正；岛冻结待续；cell-sel 等证伪项见台账。

### 2026-10-02 — 裁决轮14：岛性能直接计时门（确认/推翻）

- [M] 新标签 `gpt-r1415-island-on-20261002-121734-45a006` 和 `...island-off...`，均1×8192/512，parity=True；同一C+H preset，仅岛开关不同，真实GPU顺序0..5。
- [M] target step-timing的64个纯decode采样调用：岛开t_compute **22.8795ms/步**，岛关 **22.7580ms/步**；去除rebuilt后分别 **22.0119 / 21.0712ms/调用**。每组两段32-call窗口、各1次rebuilt。含build/alloc/inputs的process_ubatch total分别23.621/23.662ms。
- [M] draft-timing的300-call窗口：decode **2.437 / 3.382ms/次**（岛开/关），约−27.9%；gpu_wait分别38/53us。没有用tok/s换算步时。
- [M] 端到端旁证：岛开PP745.8584/TG101.5556，关579.5224/78.7614；结构和输出门均通过，但不能把TG提升归为target compute提速。
- [X] “target步时23→9–11ms”预测被本轮直接计时推翻：岛开≥本轮基线且不满足<12ms，**岛性能门失败，性能承诺冻结，结构代码保留**。
- [M] 按用户门禁，不执行本轮nsys decode和2×64K岛+B+H；NCCL占比/总同步次数仍无新增profiling证据。
- [C] F/G旧“已编译待测”记录不符合live状态：F未接模型图/ggml op，G仅local有port、remote未部署。不得把设置环境变量当真启用。正在补现成适配的接线以分别测一次，parity失败即弃、不修数值。
- 原始日志和直接计时JSON：`final-results/ROUND14-15/{island-on.log,island-off.log,island-on-timing.json,island-off-timing.json,A-verdict.json}`。


### 2026-10-02 — 裁决轮15 G：Onecat XQA弃用

- [C/M] remote原运行时未部署Onecat源文件；本轮将local现成port/dispatch部署，编译成功，gate默认off。
- [M] 新标签gpt-r1415-g-20261002-121734-45a006，唯一1×8K/512；日志出现using opt-in Onecat Flash-V100 XQA decode path，真选择门通过。保留MTP4/PAD6/FOLD1和block F16 gather，未强制DENSE。
- [M/X] PP747.79、TG22.40，parity=False，MTP accepted0/2034=0%，与旧输出不一致。按用户要求弃用，saved preset显式GGML_CUDA_FATTN_ONECAT=0，不修内核、不重测。
- [M] 是否已有V100特化内核：已有S1 decode.sqlite确有gated_delta_net_cuda_128x4_volta<true>21774次、mul_mat_vec_iq2_s_moe_volta<true>6060次、volta_sgemm_32x32_sliced1x4_tn7260次。这是S1既有执行证据，区别于本轮新F/G；Q8_HEAD_TC=0仍关闭。


### 2026-10-02 — 裁决轮15 F：功能通过，速度无明确收益

- [C] 补齐已有PLE adapter的ggml op/API、CPU fallback、CUDA dispatch、严格meta规则及build_ple环境门；op追加到末尾，旧opcode数值不变。部署前修正history写回byte offset和[C,H]→[H,C]转置，未修改CUDA内核算法或舍入。
- [M] 完整llama-server重编parallel6通过；CPU接线fixture通过prefix数值和全部5个rollback slots。
- [M] 唯一新标签gpt-r1415-f-20261002-121734-45a006，1×8K/512；实际CUDA host-dispatch日志using opt-in Onecat PLE SM70 convolution。F=1、F16半舍入=0、G=0。PP748.74/TG100.77，parity=True，MTP408/408接受。
- [M] 对本轮同配置岛开基线745.86/101.56，PP+0.39%、TG-0.78%，单样本无明确速度收益。保留可选F=1 preset，但普通preset显式F=0/G=0，不推生产。
- [C/M] 只验证单序列；多序列F的输出view存在tail间隙，不能从本次结果推定multi支持。后续若要启用多路，先处理contiguity并单独验。
- [M] R14+R15共四个1×8K/512请求，各新label/new log；没有额外GPU重测。A门失败，因此不跑nsys和2×64K三合一。测试结束，GPU显存释放。


### 2026-10-02 — 裁决轮16 A：全步墙钟拆账

- [M] 唯一1×8K/512，gpt-r16-wall-20261002-131303-c4965a，parity=True，PP745.15/TG98.98（带埋点）。103个完整生成循环：全步50.1101ms，target graph_compute24.9460、draft graph_compute合计6.3434、显式context同步1.7953、采样扣同步2.7786、投机簿记0.4355、模型输入输出包装/未细分底层等待13.6531、server剩余0.1583ms。轮间gap另0.0165ms。
- [M] 稳态101个5-token循环：47.7459ms，其中target23.6720、draft6.1957、sync1.8090、sampling2.7998、bookkeeping0.4388、wrapper12.6770、server0.1535ms。计时表闭合，无tok/s换算。
- [X] 图外27ms不能全归因CPU采样/投机簿记：两者合计仅3.2141ms（稳态3.2386），未触发>10ms优先分支。wrapper含未细分backend等待，不能宣称全部CPU可消除。按序进入B。
- [C/M] 埋点默认off，无新增sync；raw spans和解析器在final-results/ROUND16-A。当前没有graph内launch/sync占比证据，C条件尚不成立。

### 2026-10-02 — GPU PLE按H+E新账本重试

- [M] gpt-gpuple-he-20261002-134348-725176，GPU PLE=1/H=1/E=1/TRUNK_W=0.5、ctx132096、graphs开；E实际激活，draft2.58GiB和LM-head0.41GiB放CUDA0。
- [M/X] 权重上卡后，graph reserve在CUDA device1申请981.40MiB时OOM，服务退出；尚未到CUDA Graph实例化和HTTP ready，8K请求未开始，无parity/速度结果。本次仍不能使用GPU PLE 132K。
- [C/M] CUDA_VISIBLE_DEVICES=0,4,2,1,3,5，device1对应物理GPU4。原始记录final-results/GPU-PLE-RETRY；没有继续改ctx或负载测试。


### 2026-10-02 — 裁决轮16 B：SM70单核原型，性能门未过

- [C/M] 主代理实现七专家格式SM70单launch路由→量化→gate/up/SwiGLU→down→归并，decode≤32；prefill保留旧compact-grid。40类fixture和最终库20个真实192/224分片例全通过，graph重放逐位一致。
- [M] workspace扩展登记遗漏先导致启动退出；随后首次模型测试发现routing scale=0的旧哨兵语义误当乘零，parity失败，速度作废。修0→有效1后，1×8K/512 **PP751.66/TG102.2761，parity=True**，MTP408/408。
- [M/X] 相对100.04仅+2.235%，未达≥8%（108.0432），**B性能未验收，默认关闭，不推生产**。七格式算子正确性不等于整机加速验收。
- [M/X] token复用compact计划、单CTA驻留、单expert CTA共享中间值均局部变慢，没有再用完整模型重复这些否定分支。
- [C] C条件未证：A未独立隔离新路径graph内launch/sync占比，不据旧S1结果越过条件立项。证据和源码final-results/ROUND16-B及experiment/monomoe-sm70。


### 2026-10-02 — R17岛内P2P整图与常驻执行器门禁

- [C] 启动器原有P2P_AR_DISABLE按存在性判断，设0也会禁用；新增AC922_TEST_P2P_AR=1执行unset，默认仍关。未改推理源/硬件/生产。
- [M] 两岛12个fixture（长度257/15360/65536×空分片开关），四连归约、32次变输入运行，三卡均graph-ready且逐位正确。
- [M] 仅开P2P/default chunk512：1×8K/512，758.58/102.70，parityTrue；target compute19.5665ms，draft2.426ms。再仅chunk0：758.40/102.50，parityTrue；target19.4505ms/draft2.413ms。对现锚点749.79/100.04为+1.15%/+2.46%；对R14岛开745.86/101.56仅+0.93%TG，单样本无重复性承诺。
- [M] 新标签nsys2023.2.1 decode采样，parityTrue：每GPU97次主图重放，第一岛每图50次P2P，第二岛47次；合计32028个P2P核，残余606个NCCL核集中4个窗口。CPU PLE/跨岛边界仍在图外；不是全模型单GPU常驻实现。
- [M/X] 图内间隙3.557%、通信核9.879%，合计候选上界13.436%（分母为各设备GPU图跨度之和）。CPU launch/sync占phase墙钟57.039%与GPU重叠，不能当图内>50%的证据。**现门槛未满足，整步常驻执行器不立项**；未声称实现它。MonoMoE<8%仅满足另一半条件。
- [C/M] opt-in preset manifests/island-h-p2p-whole-single；结果final-results/ROUND17。主测试已停，58C线保留；GPU实测最高50C。无长上下文/多路/真实语料新增验证。


### 2026-10-02 — R18剩余优化顺序闭环

- [M] 预设门：decode候选TG至少+5%、PP退化<=3%；prefill候选PP至少+10%、TG退化<=3%；均要求1×8K/512输出IDs完全一致。锚点749.79/100.04；compute参考R14同参数另一次采样22.8795ms。
- [M/X] A锁页argmax：compute23.591ms，PP752.54/TG100.49，parityTrue，TG+0.45%未过门，源码回退。
- [M/X] B GPU隐藏状态PAD6反馈：compute23.4315ms，PP749.40/TG103.82，parityTrue，TG+3.78%未过门；其后作为C依赖，最终B+C源码全部回退。
- [M/X] C固定GPU token链：GPU反馈fixture128次通过；实际启用后compute25.1351ms，PP752.18/TG43.00，最终输出parityTrue但draft接受率260/999=26.0%，拒绝。不能把target校验通过当作draft行为等价。启用前一轮被backend_sampling请求值误挡，仅跑到B，已排除。
- [M/X] D异步跨岛锁页桥：双向×三尺寸×四队列fixture全目的卡正确，实际桥接日志确认；compute22.0385ms，PP805.82/TG103.39，parityTrue，+7.47%/+3.35%，独立方案未过门。
- [M] E保留FIFO预填充组合：稳定GPU地址+每输入双份锁页快照+Meta复合事件+异步跨岛桥+LLAMA_PREFILL_KV_PAD=8192。两岛递归状态12批fixture开/关都逐位正确；1×8K/512 **compute22.5495ms，PP966.51/TG102.31，parityTrue**，相对锚点+28.90%/+2.27%，过prefill门。
- [C/M] 运行记录frames15/staged_inputs375/cross_copies40；cross是调度路径计数，不等于40次全都async成功。增益是组合结果，未隔离KV padding收益，未实测GPU重叠比例。compute为主机提交调用耗时，不是GPU完成墙钟。新开关默认off，保存opt-in `island-h-fifo-single`，没有推生产。
- [C/M] 仅四个runtime源码文件保留，公共header字节恢复原样，无新依赖。失败补丁与原始日志归档ROUND18。MonoMoE+2.235%/GPU PLE132K OOM沿用已验证失败结果，本轮未重复。常驻执行器仍未满足R17门。
- [M] 本轮记录GPU最高50C、host最高49C；测试服务inactive、六卡0MiB。未动BMC/风扇/驱动/内核/重启。长输入、多并发和生产可靠性未新增验证。

## 2026-10-02 GPT 接手后的修正（覆盖旧C线状态）

- [X] v13“6144降4096三路均分可修污染”、KV粒度512、合法[0,6144,6144]/[0,256,256]是污染的结论均推翻：真实KV粒度256；2个KV头整GQA分组必须保持匹配，空成员合法。
- [M/C] KV/RS按同名Meta()合并为首岛的真实buffer map漏修已补齐；两岛各KV274.13MiB、RS101.12/89.62MiB。
- [M/C] Power深split-state递归栈溢出（frame5488+19808B）改显式栈；900层旧快照失效数值901通过，最终1MiB栈/debug开也通过。日志不能再递归解析descriptor并清正在写入的cache。
- [M] C：1×8K633.70/99.93；C+H：749.79/100.04；短双路299.72/299.71、45.74/46.39，各512-token parity全部True。NCCL只在真实三卡岛内组执行，边界copy保留；未重测nsys同步总数。
- [C/M] C+H preset落盘；B累计真实长度32768门短双路未触发，长C+B仍待验。E会禁用岛模式，不应称C+E整合。完整证据见final-results/GPT-ISLAND/RESULTS.md。

### 2026-10-02 12:09 — 岛解冻：GPT 接手后修通（v14+）

- **[M-转述]** 接力包生效：GPT 按 §2 入口（meta.cpp:1381 比例断言）继续修复，岛 3+3 已跑通——前半层岛 A TP3 → 激活过岛 → 岛 B TP3 后半层；岛内 3-rank AllReduce 确认、完整推理跑通、512-token parity 一致。
- **未确认（重要）**：§2 的性能预测（跨岛同步 96→0、步时 23→9-11ms）**没有任何 profiling 背书**。注意 tok/s ≠ 步时：MTP 接受 ~5 token/步时，10ms/token 可能对应 ~50ms/step——数字方向未知。
- GPT 更新了 GPT-HANDOVER.md §2（跨代理文档互改，符合"可推翻"精神）。
- **验收缺口清单（下一位执行者）**：① LLAMA_STEP_TIMING=1 岛开/关各跑 1×8K，直接比 t_compute/步（比 nsys 便宜且直击步时）；② 顺手收 draft-timing（岛下草稿 forward 应显著降）；③ 若步时确降，再跑 nsys 双相位收 NCCL kernel 占比（对比 S1 的 25.5%）；④ 4×64K 多路岛测试（岛+B+H 三合一的终极组合）。

### 2026-10-02 R19: GPU chain repair and real-corpus determinism

- [M/C] Repaired device-input producer ordering (old delayed-producer fixture stale4/4, new0/4), restored MTP GPU hidden/token chain, resident IQ3_S embedding and FIFO. Fresh repeated8K512: PP960.79/TG111.07,512-ID parity and408/408; target compute host-call22.000ms.
- [M/C] Found and cleared stale pending_h before new-prompt MTP catch-up. A remaining nondeterministic QSA radix gather was replaced with stable tie membership/index order.27+2 shapes×12 CUDA replays pass; same real8K256 prompt now repeats256/256 tokens and182/288 acceptance.
- [M] Real corpus cold206.71PP/59.38TG, warm1038.65/83.03. Warm third-request regression1055.16/113.13; do not label this a fresh-load comparison or use repeated text to claim real1.0 acceptance.
- [M/X] Single-device draft with target islands passed512 parity but only87.29TG, so placement changes archived/reverted. Compact graph metadata saved~40MiB anonymous PSS, packed lossless split cache~37MiB more; speed neutral. malloc_trim no meaningful speed gain and removed.
- [M] CPU RAM31.5GiB onlynode0, node8RAM0; SATA BX500SSD,64KiB basepages. Prior "HDD" or RAM interleave advice would be wrong. CPU PLE26.82GiB gives large cold/warm disparity.
- [C/M] R20 new route separates six-GPU row-sharded PLE from two TP3 trunk islands. IQ4_NL six-way gather48 replays exact;132K reachesREADY but firstUB512prefill GPU3 workspaceOOM. NewUB256/boundary29+same-devicebridge and no whole-file prefetch under test; no R20 performance success claimed yet.
- Full R19 measurements: final-results/ROUND19/RESULTS.md. Overall free-optimization goal remains active.


### 2026-10-06 — 上游回移与真实岛组合（Codex）

- [C/M] 只在 `experiment/codex-upstream-20261006` 隔离副本工作；3674 个原源码文件 SHA 核对无变化，生产/固件/风扇/驱动/内核/电源未改。
- [C/M] 本轮 `prefill-parallel` 旧 PROD_ENV 的 PCI_BUS_ID + visible `0,4,2,1,3,5` 混合两岛；明确 UUID 排序修正为 0004:04/05/06 与 0035:03/04/05。新日志两组真实图中 P2P AR 激活。历史测试须按各自设备顺序判断，不能统称以前所有岛实验都错误。
- [C/M] 选择回移 #30020 NextN reserve、#29825 最终 fused scorer/CUDA四头、#29901 64-key×8-token tile；9/9 CUDA indexer数值fixture与实际CPU NextN/cache fixture通过。#29994不同标准kpool布局不强套，现有partial seq_cp等语义保持。
- [M] 冻结本轮基准8K648.86/93.67、64K738.80/83.02。真实岛+P2P+新indexer+R19 GPU chain/embed/FIFO/stable top-k/compact/cache：8K920.56/125.06、64K1003.40/108.81，两档512-token回归一致。target graph_compute host调用窗口中位14.603/14.371ms；不是GPU完成时间或全MTP步时。
- [M] 固定真实8K两波输出512/512一致、367/573接受；冷183.83/71.29、热949.21/98.48。重复文本408/408不可作为真实接受率。速度候选132096ctx、MTP4、UB1024、Q8 K/V；单卡最小空余2281MiB、最高GPU/主机传感器46C（5秒采样）。
- [M] 262144ctx+Q8K/Q4V+UB512：8K881.43/113.92、64K987.49/99.20回归通过；只验证分配及64K实际输入，不是262K满窗。UB1024首次prefill workspace4824MiB OOM，已退出。
- [M/X] 新层流水小模型跨SYS双目标层、4-copy、reset/tail/自动输出drain通过8×128logits（max1.164e-10）；整模型64K1176.37/88.46、重复文本512parity，但真实重复token37分岔。同layer/R19/UB512关闭新流水时真实512/512一致。新增流水未过整模型门，源码3文件与runtime已回退V1；完整失败实现归档，不能将1176作为合格候选。
- 工件：`codex-upstream-20261006/REPORT.md`、`results.json`、`artifacts/upstream-bundle-v1.patch`、`artifacts/rejected-stage-pipeline/`。可复跑 `python3 experiment/codex-upstream-20261006/scripts/replay.py fast132`。测试unit全部退出，GPU已释放，未推生产。
