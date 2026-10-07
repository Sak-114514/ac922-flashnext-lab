
### 2026-10-06 — 上游回移与真实岛组合（Codex）

- [C/M] 只在 `experiment/codex-upstream-20261006` 隔离副本工作；3674 个原源码文件 SHA 核对无变化，生产/固件/风扇/驱动/内核/电源未改。
- [C/M] 本轮 `prefill-parallel` 旧 PROD_ENV 的 PCI_BUS_ID + visible `0,4,2,1,3,5` 混合两岛；明确 UUID 排序修正为 0004:04/05/06 与 0035:03/04/05。新日志两组真实图中 P2P AR 激活。历史测试须按各自设备顺序判断，不能统称以前所有岛实验都错误。
- [C/M] 选择回移 #30020 NextN reserve、#29825 最终 fused scorer/CUDA四头、#29901 64-key×8-token tile；9/9 CUDA indexer数值fixture与实际CPU NextN/cache fixture通过。#29994不同标准kpool布局不强套，现有partial seq_cp等语义保持。
- [M] 冻结本轮基准8K648.86/93.67、64K738.80/83.02。真实岛+P2P+新indexer+R19 GPU chain/embed/FIFO/stable top-k/compact/cache：8K920.56/125.06、64K1003.40/108.81，两档512-token回归一致。target graph_compute host调用窗口中位14.603/14.371ms；不是GPU完成时间或全MTP步时。
- [M] 固定真实8K两波输出512/512一致、367/573接受；冷183.83/71.29、热949.21/98.48。重复文本408/408不可作为真实接受率。速度候选132096ctx、MTP4、UB1024、Q8 K/V；单卡最小空余2281MiB、最高GPU/主机传感器46C（5秒采样）。
- [M] 262144ctx+Q8K/Q4V+UB512：8K881.43/113.92、64K987.49/99.20回归通过；只验证分配及64K实际输入，不是262K满窗。UB1024首次prefill workspace4824MiB OOM，已退出。
- [M/X] 新层流水小模型跨SYS双目标层、4-copy、reset/tail/自动输出drain通过8×128logits（max1.164e-10）；整模型64K1176.37/88.46、重复文本512parity，但真实重复token37分岔。同layer/R19/UB512关闭新流水时真实512/512一致。新增流水未过整模型门，源码3文件与runtime已回退V1；完整失败实现归档，不能将1176作为合格候选。
- 工件：`codex-upstream-20261006/REPORT.md`、`results.json`、`artifacts/upstream-bundle-v1.patch`、`artifacts/rejected-stage-pipeline/`。可复跑 `python3 experiment/codex-upstream-20261006/scripts/replay.py fast132`。测试unit全部退出，GPU已释放，未推生产。
