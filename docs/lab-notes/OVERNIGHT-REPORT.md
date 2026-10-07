# 夜间优化战役总结报告（2026-10-02 凌晨）

> 监工：ZCode | 执行：5 个子代理（A/B/C/D/E/H/F/G）+ 12 轮编译 + 30+ 次实测
> 机器状态：**主机已关机**（BMC 正常、watchdog active、风扇 4000 安全档、boot-guard failed 待查）

---

## 一、最终记分板

### ✅ 已转正（parity 门禁通过，可日常使用）

| 优化 | 开关 | 实测收益 |
|---|---|---|
| **H：MoE 死块消除** | `GGML_CUDA_MOE_GROUPED=1` | 单路 PP 532→586（+10%）；64K 多路 PP +14%；短上下文 PP +85~118%；**decode 的 MoE matvec 也提速**；输出逐位一致 |
| **B：每序列 QSA 快路径** | `LLAMA_QSA_POOLED_MULTI=1`（仅多路长上下文） | 2×64K decode 13.2/17.5 → 25.5/27.9（+74%）；与 H 合体后见下 |
| **B+H 最优组合** | 两开关同开 | **2×64K：decode 38.7/39.3（基线的 2.9×/2.2×），总吞吐 30.7→77.9 tok/s（+154%），PP 175.2，parity=True** |

**明日最优配置一句话**：`GGML_CUDA_MOE_GROUPED=1` 常开；多路长上下文（>16K/流）加 `LLAMA_QSA_POOLED_MULTI=1`。

参考基线（今夜冻结）：单路 532.6/78.85；2×8K 226/34.2+35.4；2×64K 153/13.2+17.5。

### ⏳ 未完成（有明确下一步）

- **C：岛内 3+3 分组**（潜力最大：跨岛通信 ÷13、decode 步时 23→9-11ms 预估）
  - 修复链已剥 7 层：成员分布(v1)→确定性边界+预检(v2)→ctx 同名合并(v3)→诊断化(v4)→空张量(v5,我)→CPY 轴传播(v6)→GDN 镜像态全局头索引(**v7 已交付已编译**)→下一个死点 SET_ROWS 断言(v8 待做)
  - **v7 从未完整跑过一次**（v7b 测试死于提前关机）——明早第一件事：直接重测 `LLAMA_ISLAND_SPLIT=3,3` 1×8K，可能已经能过
  - 若再崩 SET_ROWS：meta.cpp:819，已装自报家门断言（打印 src0/src2 轴值），证据发给 C 做 v8
- **E：草稿头去 TP**（单路 decode 30-40% 潜力）：6 路下草稿卡放不下（需 19.2G>15.5G），**改为不对称分割**（草稿卡主干份额减半 0.5,1,1,1,1,1），设计已在 todo，等 C 收工（llama-model.cpp 冲突）
- **F：PLE 卷积内核**（已编译进树未测）：`LLAMA_PLE_ONECAT=1`，省 ~80MB/卡 + launch 次数减少
- **G：Flash-V100 XQA decode 内核**（已编译进树未测）：`GGML_CUDA_FATTN_ONECAT=1`，与 B 的 F16 gather 有联动杠杆，期望收益 ≤1%（对冲性质）

### ❌ 证伪/冻结

- D：BF16→F16（parity 完美，PP +2~4% TG -4~5%，收益边缘）——模型留盘 `IQ3_XXS-MTP-Q8.bf16-f16.gguf` 备用
- 图缓存容量假设（cap16 在 1K/8K 均无差异，硬编码 4 保持默认）
- A：多路定长投机步（hybrid 递归目标数学上无收益，代码保留）
- FP8 KV / Remote KV sidecar / 1Cat HC/GDN 内核移植（尸检结论，勿重试）
- NCCL 调参全灭（autotune 已最优；强制 LL -22% PP；Tree 全面更差）

---

## 二、机器状态与开机方法

- **主机已关机**。关机经过（诚实记录）：我挂的"测试后自动关机"链条在本地"取消"时远端已生效，于 ~02:05 执行了 poweroff——早于"测到成功为止"的意图，**v7b 测试结果因此丢失**。有序关机，无硬件损伤。
- BMC 侧：watchdog active；风扇 target 4000（安全档，boot-guard 的 ExecStopPost 正确回退）；**ac922-boot-guard.service failed**——死于 01:55:57，与主机侧 v7 岛测试 GDN abort 同一秒，日志显示 "phosphor-fan-control mask timed out"+"target readback mismatch" 后安全回退并退出。明早开机后需 `systemctl status ac922-boot-guard` 排查并重启它。
- **开机命令**：`ssh ac922-bmc 'busctl set-property xyz.openbmc_project.State.Chassis /xyz/openbmc_project/state/chassis0 xyz.openbmc_project.State.Chassis RequestedPowerTransition s xyz.openbmc_project.State.Chassis.Transition.On'`（或按电源键）。开机后风扇会先 4000 后由 guard 回 3000。
- 主机起来后记得：nvidia 驱动加载（nvidia-startup.timer）、模型盘挂载。

## 三、关机期间的静态分析与修复（凌晨补充）

**岛 v7 之后的死点已定位并本地修复**（v7b 未跑完即遇关机，但静态分析完成了它的未竟之业）：
- 根因：`handle_gated_delta_net` 返回**退化段描述符**（每缓冲区和=0），而 v6 的 CPY 传播让 SET_ROWS 的目标缓存视图携带**真实非零总和** → `split_states_equal` 必炸。6 路不炸是因为该路径被 gdn→cpy 图融合隐藏，**SET_ROWS 在岛模式首次露面**。
- 修复（本地 `ggml-backend-meta.cpp` 已应用、括号平衡）：新增 `split_states_compatible`——轴相同且任一侧退化（零和）时放行。已交 C 代理评审（可能升级为"GDN 返回真实段结构"方案）。
- **明早开机第一动作**：同步本地 `ggml-backend-meta.cpp`（含此补丁 + C 的评审修订）→ 编译 → 测 `LLAMA_ISLAND_SPLIT=3,3` 1×8K。自报家门断言（GDN 六轴 + SET_ROWS 双侧）都还装着，若再崩会直接点名。

## 四、明日待办（按优先级）

1. 重测岛 v7（见上）
2. 若过 → 跑 phase-0 验收（NCCL 占比 <5%、decode <12ms、parity）→ 岛+H+B 三合一测最优配置
3. 排查并重启 boot-guard
4. E 不对称分割设计落地
5. F/G 开关各一次 1×8K 测试
6. B 的 8K 差距修复（cell 粒度选块，round5 设计已就绪）或服务层长度感知门（3 行改动方案已给出）

## 五、资产索引

- 所有补丁设计文档：`patches/*-NOTES.md`（8 份）
- 全部实测数据：ac922 `~/ac922-llama-next/manifests/zcode-*`
- 全部日志：ac922 `~/ac922-llama-next/logs/zcode-*`
- 测试工具（可复用探测框架）：ac922 `scripts/zcode_probe.py`（支持 --frozen/--ref/--set/--cases）
- 源码基线：本地与 ac922 同步（`.zcode-orig/` 有各代理的分层备份）
- F16 转换模型：`models/.../IQ3_XXS-MTP-Q8.bf16-f16.gguf` + 转换器 `scripts/convert_bf16_f16.py`
