# 复活实验：P2P_AR + MONOMOE 组合 A/B（2026-10-04）

零源码改动、纯环境变量配置实验。目的：同口径重测两个当年被单项门槛（>1.5% 档）拒绝的历史正收益开关，判定组合增益与互斥性。

- 硬件：ac922（6×V100-SXM2-16GB，CUDA 12.4 运行链 with-ac922-cuda525），实验期间 GPU 空闲、全程独占
- 二进制/模型：`sources/llama-v100-v3/build-ac922/bin/llama-server` + `IQ3_XXS-MTP-Q8.gguf`（与生产一致，未动源码）
- 基准：`matrix_runner.py --cases 1:8192 --reference-directory manifests/speed-bundle-single`（c1 单流、8192 prompt / 512 gen、MTP d4）
- 臂公平性：四臂同一会话背靠背执行、同一参考目录、同一启动模板（`launch_integration_arm.sh` 逐行照抄，仅参数化两个开关）
- 生产配方（臂0）：island preset 全套 + `GGML_SM70_BF16_GEMV=1` + `GGML_SM70_MOE_REWORK=1`（即 TG 94.3 时代配方）

## 一、四臂结果（第 1 轮）

| 臂 | 配置增量 | PP (tok/s) | ΔPP | TG (tok/s) | ΔTG | parity | MoE kernel 活性 |
|----|----------|-----------|------|-----------|------|--------|------------------|
| 0 基线 | — | 632.35 | — | 93.18 | — | 全匹配 | MOE_REWORK 90 hits（探针），0 MonoMoE |
| 1 | +`AC922_TEST_P2P_AR=1` | 633.96 | +0.25% | 93.45 | +0.29% | 全匹配 | 同臂0（REWORK 在干活） |
| 2 | +`LLAMA_MONOMOE_SM70=1` | 621.64 | **−1.69%** | 93.13 | −0.06% | 全匹配 | MonoMoE 10 banners，REWORK 0 hits |
| 3 | 两者同开 | 623.74 | −1.36% | 93.27 | +0.09% | 全匹配 | MonoMoE 10 banners，REWORK 0 hits |

判定规则（预登记）：臂1/2/3 任一比臂0 高 >1.5% 才加测第 2 轮——**无一达标**。

### 补充：臂0/臂1 加测的第 2 轮（低于门槛的边缘信号尽调，非预登记要求）

臂1 是唯一正信号（+0.29% TG），略高于同会话噪声带（见下），故追加一组确认对：

| 臂 | PP | TG | parity |
|----|-----|-----|--------|
| 0-r2 | 619.43 | 93.20 | 全匹配 |
| 1-r2 | 624.53 | 93.22 | 全匹配 |

第 2 轮 Δ（臂1−臂0）：PP +0.82%、TG +0.02%。两轮均值：臂0 PP 625.89 / TG 93.19，臂1 PP 629.24 (+0.54%) / TG 93.34 (+0.15%)。第 1 轮的 +0.29% TG **未复现**——落入同配置重复噪声带内。

### 噪声参照（判定增益是否超出噪声）

- 同会话同配置重复（当日早前 moerework A/B）：on 622.11/93.15 vs on-r2 622.73/92.98 → PP ±0.1%、TG ±0.18%
- 同配方跨会话（integration-allon r1/r2）：621.38/93.46 vs 685.10/95.20 → PP 跨会话可漂 ±10%、TG ±2%
- 本实验四臂为同会话顺序执行，适用前者（紧噪声带）：TG ±0.2%、PP ±0.3% 左右

## 二、互斥判决：结构性互斥（REWORK vs MONOMOE 抢同一路径）

探针方法：用 `GGML_SM70_MOE_REWORK=2`（=2 额外逐次打印命中）起短命服务，发一个小 completion（48 token），grep 日志计数：

| 探针 | 配置 | sm70_moe_rework 命中 | MonoMoE banner |
|------|------|---------------------|----------------|
| probe-base | MONO=0, REWORK=2 | **90**（K=256 N=2560、K=192 N=2560 等 Q2_0 形状） | 0 |
| probe-mono | MONO=1, REWORK=2 | **0** | 10（gate=iq2_xs/iq2_xxs/iq3_s，down=q2_0/iq4_nl） |

机理（源码确认，未改动）：
- `LLAMA_MONOMOE_SM70` 在 `src/models/qwen4exp.cpp::build_layer_ffn`（约 :2501）建图时判定 `n_tokens<=32` 即把整个 MoE FFN（gate+up+down 单 op）换成 `ggml_monomoe_sm70` 自定义算子——MUL_MAT_ID 的 Q2_0 down 投影调用根本不再发生；
- `GGML_SM70_MOE_REWORK` 挂在 `ggml-cuda/mmvq.cu`（:1272 起）的 Q2_0 MUL_MAT_ID 路径上，MONOMOE 开启后被整段饿死（90→0 命中）。

结论：**互斥是结构性的，不是性能打架**——两者覆盖同一段 decode MoE matvec，只能二选一，无法叠加。臂3 ≈ 臂2（ΔTG +0.09% vs −0.06%，差异在噪声内）亦印证"同开 = MONOMOE 独占，P2P_AR 不改变 MoE 路径归属"。

## 三、组合增益结论

1. **`AC922_TEST_P2P_AR=1`（R17 岛内 P2P 整图归约，历史 +2.46%）**：本轮 +0.25% PP / +0.29% TG，第 2 轮 +0.82% / +0.02%，两轮均值 +0.54% / +0.15%——**无可复现增益，仍在门槛（1.5%）之下**。历史 +2.46% 出自不同时代锚点，未能在当前 94.3 配方同口径复现。
2. **`LLAMA_MONOMOE_SM70=1`（R16-B 单 launch MoE，历史 +2.235%）**：−1.69% PP / −0.06% TG，且**完全顶替** MOE_REWORK（互斥）。即 MonoMoE 相对"MOE_REWORK 已在生产"的现状是净不优（TG 持平、PP 略退）。历史 +2.235% 的对照系当年没有 MOE_REWORK 内线，如今内线更强，复活无意义。
3. **组合（臂3）**：无组合增益，数字≈臂2。两个开关一个中性一个负贡献，且 MoE 路径只剩一个能干活。
4. **建议**：生产配方维持原样（`AC922_TEST_P2P_AR=0`、`LLAMA_MONOMOE_SM70=0`），两开关继续归档。数字如实记录：本轮同口径下没有复活价值。

## 四、R21 开关名记录（本轮不做主实验）

- **`GGML_CPU_PLE_ROW_CACHE`**（`ggml/src/ggml-cpu/ple-row-cache.h:187`，=1 启用；=0 默认关）
- 机制：CPU PLE 编码行缓存——O_DIRECT 4KiB miss + resident mmap + 4-way 262k rows；R21 实测冷 prefill +49%（360.02 PP / 8K256），暖基准测不出其价值（本轮 matrix_runner 为暖路径），故仅记录开关名，未进臂。

## 五、复现材料

- 服务器脚本（新建，模板逐行照抄仅参数化）：
  - `/home/debian/ac922-llama-next/scripts/launch_revival_arm.sh`（`$1`=臂名/日志名，`$2`=P2P_AR，`$3`=MONOMOE，`$4`=MOE_REWORK 默认1，`$5`=unit 名默认 ac922-flashnext-test）
  - `/home/debian/ac922-llama-next/scripts/run_revival_arm.sh`（单臂一轮：stop→launch→health→matrix_runner→stop）
- 结果目录：`manifests/revival-arm{0,1,2,3}`、`revival-arm0-r2`、`revival-arm1-r2`（各含 c1-p8192.json + .parity.json + gpu-telemetry）
- 服务日志：`logs/revival-arm{0..3}.log`、`logs/revival-probe-{base,mono}.log`
- 复跑单臂示例：`bash scripts/run_revival_arm.sh arm1 1 0 1 ac922-flashnext-test revival-arm1`
- unit 名说明：基准臂沿用固定名 `ac922-flashnext-test`（matrix_runner.py 硬编码检查该 unit 的 ActiveState），每臂先 stop 再 start；探针用独立名 `ac922-flashnext-probe` 避免碰撞
- 参考目录固定 `manifests/speed-bundle-single`（PP 532.63 / TG 78.85 锚），parity 判定 = output_token_ids 逐 token 全同

## 六、收尾状态

- systemd user 单元 ac922-flashnext-test / ac922-flashnext-probe 均已 stop，无残留 ac922-* 单元
- `nvidia-smi`：6 卡全部 0% / 0 MiB；无 llama-server 进程
- 源码树 `sources/llama-v100-v3` 零改动（仅新增 scripts/ 下两个启动脚本）
