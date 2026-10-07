# sm_70 kernel 战役总账（2026-10-03 深夜）

基线：llama.cpp fork 83.03 tok/s（real corpus warm 8K/256，R19 锚点）。
起点诊断：R17 trace = BF16 matvec 33% kernel 时间 + 77.8 万 kernel 事件/5.2s + GPU busy 65%。

## 战役一：BF16 GEMV 单 kernel（已落袋 ✅）

- 产出：`bf16-gemv/`（kernel_sm70_bf16.cuh + bench + WIRING.md + INTEGRATION.md）
- 微基准三轮：大 K +27~35%；TP 分片形状假设被运行时真相修正（hc 系不切分，真实命中
  320×10240 实测 2.3×）；staged（shared 预取 y）判死
- 接线：42 行最小 diff，白名单（k≥10240 || nc≥5∧N≥8192），GGML_SM70_BF16_GEMV 默认开
- **实测 +8.9%**（111.2/110.5 vs 100.2/103.4，同口径 A/B 各 2 轮），parity 4×512 逐位一致，
  draft 接受率 408/408 不动
- 教训：copy 上界基准在小形状被 L2 污染（5MB<6MB L2 跑出 130%"峰值"）

## 战役二：整层融合 launch（微基准过 ✅，接线中）

- 产出：`layer-fusion/`（fused_projection_sm70.cuh 760 行 + bench + DESIGN.md）
- 设计核心：一层 ≠ 1 launch（依赖链+归约序敏感）= **8 个依赖 wave**；16-17 launch → 6
  （只融合 n_task≥2 的 wave）；任务表 SoA + 前缀偏移 + block 二分认领；小矩阵 block 回填
  大矩阵 SM 空洞 = 收益物理来源；static（参数空间表）恒优于 dyn
- **V100 实测四组 1.41-1.50×**（GDN nc1 85.8→60.9µs、nc5 132.7→90.1、QSA nc1 81.9→54.7）；
  W7 四小矩阵组 -66%；正确性逐位 0.0
- **教科书 bug 入档**：`__ldg` 用于 kernel 参数空间指针（static 表按值传参）→ sticky
  "operation not supported" 毒死后续全部 CUDA 调用——首跑 7/8 wave 假失败的根因
- 折算步时 +7~10%（两项独立可乘）

## 账本

| 阶段 | tok/s |
|---|---|
| R19 基线 | 83.03 |
| +BF16 GEMV（已落袋，合成口径 +8.9%）| ≈ 90（real 口径折算）|
| +fused（接线中，+7~10%）| **97~104 预期** |

## 素材库
- Luce 评估（reports/luce-port-assessment.md）：Route C 判决；megakernel 是 0.8B 硬编码 PoC 不可用
- onecat-kernels/：1Cat 遗产（glm53_gemv 骨架、HMMA 汇编、marlin）
- Strata 收档（reports/strata-ac922-port-record.md）：66.1 tok/s + 移植知识（sm_70 零拷贝陷阱等）
