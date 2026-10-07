# sm_70 BF16 GEMV kernel 接入 llama.cpp fork — 集成与验证报告（2026-10-03）

## 结论（TL;DR）

- **接入完成，parity 门全过，decode 实测 +7~11%（均值 +8.9%），prefill 不受影响。**
- 生产 preset（island-h-fifo-single，8192 输入 / 512 输出贪心，seed 1234）同二进制 A/B：
  - TG（decode）：开关开 **111.24 / 110.50 tok/s**（两次），开关关 **100.17 / 103.35 tok/s**（两次）
    → 每臂取均值：110.87 vs 101.76 = **+8.9%**；最差对（110.50 vs 103.35）= +6.9%，最好对（111.24 vs 100.17）= +11.0%
  - PP（prefill）：955.0/953.3 vs 952.3（冷）/1090.7（热页缓存）→ 预期内持平（nc=8192 远超 matvec 阈值，走 cuBLAS 路径不经过本 kernel）
- **parity：4 次运行（每臂各 2 次）512/512 token ID 全部逐位一致**，且与 R19 锚点参考
  `manifests/speed-bundle-single/c1-p8192.json` 完全一致；两臂 MTP draft 接受率均为 408/408。
- 开关默认开；`GGML_SM70_BF16_GEMV=0` 一键回退原路径（无需重编译）。

## 改动文件清单（相对接入前的工作树，其余文件零改动，已用 diff-of-diffs 逐位验证）

| 文件（服务器 /home/debian/ac922-llama-next/sources/llama-v100-v3/） | 改动 |
|---|---|
| `ggml/src/ggml-cuda/kernel_sm70_bf16.cuh` | **新增**（拷贝自 bf16-gemv/kernel_sm70_bf16.cuh，namespace gemv_sm70，自包含） |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | +1 行 include；+31 行分发分支（见下） |

完整 patch：`integration-artifacts/ggml-cuda.cu.integration.patch`（42 行）。

### ggml-cuda.cu 改动内容

1. include（mmf 与 mmq 之间）：`#include "ggml-cuda/kernel_sm70_bf16.cuh"`
2. `ggml_cuda_mul_mat()` 内、现有 `ggml_cuda_should_use_mmvf(...)` 判定**之前**插入分支：
   - 守卫（全部满足才走新 kernel，否则落入原路径）：
     `src0->type==BF16`、`cc==GGML_CUDA_CC_VOLTA(700)`、`ne11<=8`（batch≤8，即 decode/MTP-verify 域）、
     2D 纯矩阵（ne02/ne03/ne12/ne13/ne2/ne3 全 1）、元素连续（nb00/nb10/nb0 等于类型大小）、
     **形状白名单** `ne00>=10240 || (ne11>=5 && ne01>=8192)`、
     以及 kernel 内建守卫（K%8、stride_w%8、stride_y%4、W/y 16B 对齐，任一不过返回 false 回退）
   - 该插入点天然只有**纯 MUL_MAT**（无 ids、无融合）：MUL_MAT_ID、mmvf-multi、mmvf+sigmoid/softplus、
     fused-GLU、bias 融合等路径全部直接调 `ggml_cuda_mul_mat_vec_f(..., ids/fusion, ...)`，不经过本函数，零影响
   - 环境开关（照 fork 内 `GGML_CUDA_MMVF_SPLITK` 的 static-lambda-getenv 模式）：
     `GGML_SM70_BF16_GEMV`（缺省=开；`0`=强制原路径；`2`=开+stderr 记录每次命中的形状）
   - launch 直接 CUDA-Graph 可捕获（无动态共享内存/setAttribute/分配，设计 D8），preset 的 CUDA graphs 开启状态下在捕获期分发一次

## 白名单的运行时核实（关键事实修正）

微基准"二战"假定的 TP 分片形状（2048×10240 等）**并未出现在运行时**：本模型这些 BF16 张量
（hc_*_down 320×10240 ×97、hc_*_inject 4×10240 ×96 等）**不参与行切分**，以全局形状到达分发点。
用 `GGML_SM70_BF16_GEMV=2` 冒烟实测命中直方图（6 卡 llama-cli，无 MTP）：

```
320×10240 @ ncols 1/2/4（各 1200/1140/570 次）+ 4×10240 @ ncols 1/2/4（各 1170/1140/570 次）
```

对这两个**真实运行时形状**补测微基准（V100, -arch=sm_70, relerr 全 0.0，naive≈fork mmvf 访存模式）：

| 形状 | nc | naive | sm70 | 加速 |
|---|---|---|---|---|
| 320×10240 | 1 | 18.8µs / 349GB/s | **8.0µs / 824GB/s** | **2.3x** |
| 320×10240 | 5 | 30.8µs / 213GB/s | **16.4µs / 399GB/s** | **1.9x** |
| 4×10240 | 1 | 9.5µs | **3.9µs** | 2.4x |
| 4×10240 | 5 | 16.1µs | **7.8µs** | 2.1x |
| 2048×10240（对照） | 1 | 57.7µs | **51.5µs** | +12%（复现 +13%） |
| 2048×10240（对照） | 5 | 89.4µs | **74.0µs** | +21%（复现 +27%） |

即：白名单 `k_dim>=10240` 实际拦截的全部形状都有实测收益证据；`ncols>=5 && n_out>=8192`
子句在本模型运行时不命中（无每卡 n_out≥8192 的 BF16 matvec），仅作为有微基准依据（12288×2560 nc5 +28%）的前瞻项保留。
已知倒退形状（512×2560 等）全部落在白名单之外，走原路径。

## 验证记录（全部在服务器，构建树 build-ac922，增量编译 18 目标，无新增警告）

1. **冒烟**：6 卡 llama-cli 加载 IQ3_XXS-MTP-Q8、24 token 贪心生成正常，kernel 命中 5790 次无崩溃（histogram 见上）。
2. **parity**：preset 服务 systemd-run 起（`scripts/launch_bf16gemv_arm.sh <0|1>`，env 取自 manifests/island-h-fifo-single/configuration.json），
   `matrix_runner.py --cases 1:8192`（8192 in / 512 out，temp 0.0，seed 1234）× 4 次：
   开/关两臂输出 token ID 互相一致、且与 speed-bundle-single 参考逐位一致（`matched: true` ×4，manifests/bf16gemv-{on,off}{,-r2}/c1-p8192.json.parity.json）。
   注：本 kernel 的 FMA 归约顺序与原 mmvf 不同（理论上有 ~1e-7 级相对差），512 贪心 token 未发生翻转——这是实测结论，不是逐位等价保证。
3. **性能 A/B**（数字见 TL;DR；EVIDENCE.txt 含完整 server_timings：predicted_ms 4593.8/4624.5 vs 5101.2/4944.3）。

## 回退方法

- **运行时（首选）**：设 `GGML_SM70_BF16_GEMV=0`（launcher 脚本已支持），无需重编译；缺省即开。
- **文件级（彻底移除）**：
  1. `cp experiment/bf16-gemv-integration/ggml-cuda.cu.pre-integration sources/llama-v100-v3/ggml/src/ggml-cuda/ggml-cuda.cu`
  2. `rm sources/llama-v100-v3/ggml/src/ggml-cuda/kernel_sm70_bf16.cuh`
  3. build-ac922 内重跑 `ninja llama-server`
  已验证：pre-integration 文件放回后 `git diff` 与接入前基线 `/tmp/pre_bf16gemv_baseline.diff` **逐位一致**。

## 工件路径

- 服务器：`/home/debian/ac922-llama-next/experiment/bf16-gemv-integration/`（EVIDENCE.txt、integration.patch、pre-integration 文件、kernel 副本）
- 服务器：`/home/debian/ac922-llama-next/manifests/bf16gemv-{on,off}{,-r2}/`（4 组基准+parity JSON）、`logs/bf16gemv-arm{0,1}.log`
- 服务器：`/home/debian/ac922-llama-next/scripts/launch_bf16gemv_arm.sh`（A/B 启动器，0/1 参数控制开关）
- 本地：`bf16-gemv/integration-artifacts/`（上述工件镜像）；本地镜像 sources/llama-v100-v3 已同步改动
- 接入前工作树基线快照：服务器 `/tmp/pre_bf16gemv_baseline.diff`（16493 行，接入前全部既有改动的完整 diff）

## 残留事项 / 边界

- 测试后 systemd 单元已停、6 卡显存 0MiB、无残留进程。
- 本轮只覆盖 plain MUL_MAT 的 BF16 matvec；MUL_MAT_ID（专家）与 fused-GLU 变体按 WIRING.md 属二期，未动。
- A/B 为单请求两臂各 2 次（fork 纪律下的 bounded 单波测量）；decode 差异（+6.9%~+11.0%）远大于臂内波动（±1.5 tok/s），方向明确。
- 512-ID parity 是贪心 token 级一致；若未来把 kernel 用于其它形状/精度域，需重跑 parity 门。
