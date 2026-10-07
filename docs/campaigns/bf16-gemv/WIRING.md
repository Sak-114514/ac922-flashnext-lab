# sm_70 BF16 GEMV kernel -- 接线方案与实测记录（2026-10-03）

## 文件
- `kernel_sm70_bf16.cuh` -- 自包含 kernel + host chooser/dispatch（子代理B产出，已评审）
- `bench_gemv_sm70.cu` -- 三版本微基准（naive=fork模式模拟 / sm70新kernel / copy带宽上界）

## 微基准首战数据（V100 SXM2, CUDA 12.4, -arch=sm_70, relerr 全部 0.0）

| 形状 | nc | naive | sm70 | copy上界 | 判决 |
|---|---|---|---|---|---|
| 12288×2560 | 5 | 57.8% | **73.7%** | 95.1% | **+28%** |
| 2560×10240 | 5 | 50.8% | **68.5%** | 94.4% | **+35%** |
| 2560×2560  | 5 | 46.8% | 51.3% | 83.1% | +10% |
| 12288×2560 | 1 | 88.6% | 92.0% | 95.0% | +4%（fork已近满速）|
| 2560×10240 | 1 | 84.2% | 90.8% | 94.4% | +8% |
| 512×2560   | 5 | 29.7% | 23.2% | 76.9% | **倒退**——需 K-split |
| 640×2560   | 5 | 34.4% | 38.5% | 100.2% | 小胜 |

结论：ncols=5（MTP 验证窗口=R17 trace 的 33% 主犯）大形状实打实 +28~35%；
nc=1 大形状 fork 现状已 84-89%（trace 时间大头应是 TP 分片后的小矩阵，待补测真实形状）；
小 N_out（<2048）需要 K-split 变体（多 block 合作一行 + 二阶段归约）。

## 二战数据（真实 TP 分片形状 + STAGE_Y 判死）

| 形状 | nc | naive | sm70 | staged | 判决 |
|---|---|---|---|---|---|
| 2048×10240 | 5 | 48.0% | **61.1%** | - | **+27%** |
| 2048×10240 | 1 | 79.5% | **89.9%** | - | +13% |
| 2048×2560 | 5 | 43.7% | 44.5% | 38.4% | 打平；STAGE_Y 判死 |
| 2048×2560 | 1 | 67.2% | 71.2% | 66.6% | +4% |
| 512×2560 | 5 | 29.6% | 23.2% | - | 倒退（形状太小）|

注意：copy 上界在小形状被 L2 污染（1024×2560 跑出 130%"峰值"——5MB<6MB L2）。
K=2560 主力形状卡 44% 的根因分析：形状小（26µs 总时长里尾巴/预热占比大）+
nc=5 时 LSU 24 条/迭代（20 条是 y 的 L1 命中读）混合延迟——**单 kernel 视角已到
该架构合理水平，进一步收益在整层融合 launch（Luce Route C-1/2 方向）**。

折算全模型预期：BF16 matvec 33% kernel 时间 × ~30% 改善（大 K 形状）× 65% kernel 占比
≈ **+5~8% decode**（低于最初 +8~15% 估计，如实修正）。

## 接线方案（进 fork）

1. 放置：`ggml/src/ggml-cuda/kernel-sm70-bf16.cuh`（照抄，namespace 不变）
2. 挂点：`ggml-cuda.cu` 的 `ggml_cuda_mul_mat` 分发里，在现有 mmvf 路径之前加守卫分支：
   ```cpp
   if (src0->type == GGML_TYPE_BF16 && cc == 70 && ne11 <= 8 &&
       !ids && !fusion && K % 8 == 0 && 对齐OK) {
       if (gemv_sm70::gemv_sm70_bf16_launch_generic(w, y, dst, ...)) return;
   }
   // fall through to existing mul_mat_vec path
   ```
3. 运行时开关：`GGML_SM70_BF16_GEMV`（默认开；=0 强制回退），照 DQ_MMV 模式
4. CUDA Graph：kernel 无动态分配/无 setAttribute，launch 可直接捕获（设计 D8）
5. parity 门：epsilon 2e-4 相对（FMA 归约序不同于 CPU 参考）；进生产前跑 512-ID 输出一致性
6. MUL_MAT_ID（专家路径）和 fused-GLU 变体：二期（当前 kernel 只覆盖 plain MUL_MAT）

## 风险与回退
- 倒退形状（小 N）由 chooser 回退保护：pick_config 发现 N_out/4 < 320 时已降 ROWS；
  实测仍慢的形状在接线层加形状白名单（只对 N_out >= 2048 启用，其余走原路径）
- parity 挂了：开关一键回退
