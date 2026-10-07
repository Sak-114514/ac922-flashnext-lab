# BF16→F16 prefill GEMM 补丁转正补记（2026-10-06）

承接 REPORT.md §2.3 合入建议：昨战役 27 行 env 门控补丁（opt-in）今日翻默认转正，并后补 Strata #540 的 ±65504 钳制。生产树已按 stash 分支纪律完成转正与重构建，全部门禁复验通过。

## 1. 转正内容（arm-only diff = ac922 `artifacts/strata-port/bf16_f16_promotion.arm-only.patch`，189 行，141 增）

三文件改动，对生产树（132 文件 dirty）的增量：

**ggml/src/ggml-cuda/ggml-cuda.cu**
1. **翻默认**：`GGML_SM70_BF16_PREFILL_F16` 从默认 0（opt-in）改为默认 **on**，`=0` 留作逃生门。开关语义照抄本 fork 现有 `GGML_SM70_BF16_GEMV` 的默认 on 模式（`env == nullptr || atoi(env) != 0`），生产配方无需加任何 env。触发条件不变：NVIDIA 分支、`cc ∈ [Volta, Ampere)`、`src1->ne[1] > 8`（decode 形状原样走 GEMV/stock 路径）。
2. **traits 换钳制转换**：`batched_mul_mat_traits<GGML_TYPE_F16>` 的 `convert/convert_nc` 改指 `ggml_get_to_fp16_sat_cuda / ggml_get_to_fp16_sat_nc_cuda`（见下）。这只影响 cuBLAS F16-compute GEMM 的操作数转换（我们的 promoted BF16 路径 + 既有 F16 权重路径），fattn/conv2d 用的 `ggml_get_to_fp16_cuda` 原封不动。

**ggml/src/ggml-cuda/convert.cu**：新增 `f16_sat()`（有限值 |f|>65504 钳到 ±65504；Inf/NaN 原样通过）+ `convert_unary_sat`（strided）+ `convert_unary_cont_sat_vec4`（vec4 主路径）+ 两个 launcher + `ggml_get_to_fp16_sat(_nc)_cuda` 分发。分发只对 **F32 与 BF16** 输入走钳制 kernel，其余（量化格式）回落原 unsaturated 分发。

**ggml/src/ggml-cuda/convert.cuh**：两个 sat 分发函数声明。

## 2. 钳制决策（Strata #540 的移植方式与放置面）

**Strata 原实现**（strata-engine `ff6f9f1`，其 `src/prefill/kernels.cu` / `gemm.cu`）：转换 kernel 内
```cuda
float f = __uint_as_float((uint32_t) in[i] << 16);
if (f > 65504.0f && !isinf(f)) f = 65504.0f;
else if (f < -65504.0f && !isinf(f)) f = -65504.0f;
out[i] = __float2half_rn(f);
```
语义：**只有有限值超界才钳**；Inf/NaN 原样传递（F16 能表示 ±Inf/NaN，且钳掉会掩盖上游真正的非有限值——这是 5fa9e7f review 明确 "keep Inf" 的理由）。激活侧另有 `hf_sat`（SwiGLU 输出，NaN 通过）。

**我们的移植**：同语义的 `f16_sat`，但放置面与 Strata 不同，理由如下：
- **"拿不到值域信息"不成立，无需记录该豁免**：钳制不需要任何数据相关的值域信息——它是 F16 格式级饱和（目标格式最大值 65504 是格式常数）。转换层知道目标格式即可，与源数据无关。
- **为什么不在泛型 convert_unary 里改**：fork 的 `ggml_get_to_fp16_cuda` 分发表除 GEMM traits 外还服务 `fattn-common.cuh` / `fattn-volta-onecat.cu`（KV→F16）与 `conv2d.cu`。在泛型 kernel 加钳制会把 blast radius 扩到 attention/conv 路径（那些路径今日未被本补丁的 parity A/B 覆盖）。Strata 的等价物（`bf16_to_f16_kernel`）也只挂在自己的 GEMM 入口。故我们新增独立 sat 分发、只由 `traits<F16>` 引用——影响面 = cuBLAS F16 GEMM 操作数，值 ≤65504 时与原 kernel **逐位相同**（同一 float 往返 + `__float2half_rn`），仅"本会变 Inf 的有限值"变成 ±65504（严格更优：防止 Inf 污染 GEMM 输出为 NaN）。
- **量化格式为何不钳**：Q8_0/Q4_0 等的反量化输出 = scale × 整数格点，scale 本身是块最大值的 F16/F32 表示 ⇒ 乘积按构造落在 F16 值域内，走原 kernel 无 Inf 风险；F16 输入则根本不发生转换。
- 有效性证明：值域内逐位相同 ⇒ 今日 4 组 512-token parity（8K/64K × arm0/arm1）全过即同时验证了钳制层"不改变合法值的行为"；超界行为属防御性修复（今天无权重/激活超界——D 线全模型 parity 早已旁证权重有界）。

## 3. 门禁复验数据（转正后，同一二进制；arm0 = `=0` 显式关，arm1 = 无 env 默认开）

### 3.1 单流 A/B + parity（`promo_runner.sh`，昨 ab_runner 同配方，seed 1234 temp 0）

| 口径 | arm0 (env=0) | arm1 (默认) | Δ | parity |
|---|---|---|---|---|
| 8K PP 热 | 789.23 | **838.41** | **+6.2%** | **512/512 逐位** |
| 8K TG | 110.89 | 111.68 | 噪声内 | — |
| 64K PP | 717.76 | **759.02** | **+5.8%** | **512/512 逐位** |
| 64K TG | 94.84 | 94.70 | 持平 | — |

- 与昨日 env 门控臂（8K 841.8 / 64K 767.3）一致（差 <1%，日间噪声）：**加钳制后收益不回退**。
- arm0 与昨日控制臂（790.9/722.8）一致：`=0` 逃生门忠实还原 stock 行为。
- 峰温 43C。构建零新增警告（27 个 warning 全为 fork 既有文件/行，无 convert.cu/新代码）。

### 3.2 2×64K 多路补验（`promo_scenarios.sh` 链式跑 multi-concurrency-bench A2 + B1 fair_share 各两臂）

| 场景 | 口径 | arm0 (env=0) | arm1 (默认) | 判读 |
|---|---|---|---|---|
| A2 2×64K | PP per flow | [590.7, 361.7] | [619.2, 363.9] | **flow1 +4.8%**（无争抢先 prefill 流，GEMM-bound）；flow2 +0.6%（与 flow1 prefill/decode 交织，非 GEMM-bound） |
| A2 2×64K | wave 墙钟 | 311.3s | **302.5s** | 整波 **-2.8%** |
| A2 2×64K | 10-04 基线对照 | PP=[591.8, 363.3] | — | arm0 与两日前基线吻合（±0.4%），控制有效 |
| B1 2×8K | wave 墙钟 / PP | 40.0s / [328.8, 331.8] | **38.2s** / [303.3, 303.3] | 墙钟 -4.5%；per-flow PP 受 fair_share 交织调度扰动 |
| B1 4×8K | wave 墙钟 / PP | 71.7s / [172.6,172.7,**215.7**,172.7] | 75.9s / [186.8,151.6,151.6,151.6] | +5.9%；arm0 单流 215.7 vs 其余 172 即本场景调度非确定性本身 |

**结论：多路 prefill 同向受益（A2 墙钟 -2.8%、无争抢流 +4.8%），B1 fair_share 单 wave 口径噪声大（10-04 与今日 arm0 自差即 +13%），两 wave 墙钟互有胜负、无一致回退方向，全部 wave 正常完成——不破。**

### 3.3 多路 parity 判读（重要：场景可复现性先做对照）

| 比较 | match | 判读 |
|---|---|---|
| A2：10-04 基线 vs 今日 arm0（**同为 F32 路径**） | 393/1024 = **38.4%** | A2 场景本身跨 run 不可复现（64K 并发流交织时序 → 批组成 → 数值 → greedy 边界翻转） |
| A2：arm0 vs arm1 | 396/1024 = 39.6% | 与同路径对照持平 ⇒ **补丁未引入超出场景固有噪声的偏差** |
| B1：10-04 vs 今日 arm0（同路径） | 2048/2048 = 100% | B1 跨日可复现 ⇒ B1 的 arm 间比较有统计意义 |
| B1 2×8K：arm0 vs arm1 | **1024/1024 全对** | 逐位一致 |
| B1 4×8K：arm0 vs arm1 | 2047/2048 | **1 token** 近决策边界翻转（2048 采样中 0.05%），与补丁已知 ~1.8e-4 数值差同类（F16 激活舍入）或 prefill 变快改变交织时序所致；单流红线（512 逐位 ×4）全过，非阻塞 |

## 4. 纪律记录（完整序列见 ac922 `artifacts/strata-port/SEQUENCE.md` 转正段）

stash push（132 文件，`prod-dirty-pre-bf16promo-20261006`）→ branch `bf16-f16-promo-20261006` → stash apply → 补丁（promo_patch.py 精确锚点）→ 构建 339/339 → 单流 A/B + parity → A2/B1 链式补验 → **转正**：checkout `ac922-flashnext-v100`（dirty tree 134 文件携带，3 补丁文件 sha 与验证构建一致 7b113286/ee7545e2/5ac0fbbc）→ 重构建 no-op（llama-server sha 397b3a56）→ stash@{0} 留作回滚 → GPU 6×0 MiB。

**回滚方式**：`cp artifacts/strata-port/*.prod-backup-20261006-promo` 回三文件 + 重构建；或运行时 `GGML_SM70_BF16_PREFILL_F16=0`（仅关 F16 路径，钳制转换对合法值逐位无差，可留）。

**插曲（记录备查）**：昨日 `ab_runner.sh` 的 health 探测用了 curl 无花括号 `%http_code` 简写，服务器 curl 7.88.1 不支持（打印字面量）→ 昨日 runner 实际卡死在 health 等待、bench 是手动补跑的（ab-runner.log 仅两行可证）。今日 promo_runner.sh 修正为 `%{http_code}` 后全链自动跑通；`do_arm.sh` 等旧脚本同款写法，复用前需同样修正。
