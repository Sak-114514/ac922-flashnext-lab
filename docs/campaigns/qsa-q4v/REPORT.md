# QSA block-sparse kernel q4_0 V 支持战役报告（qsa-q4v）

- 日期：2026-10-06（深夜 00:40 起测）· 执行：ZCode（qsa-q4v 线）
- 任务：给 fork 的 QSA block-sparse attention kernel 增加 q4_0 V 支持（env 门控 `LLAMA_QSA_Q4V=1`），消除 262K+K8V4 配置的 gather 回退（UB1024 下 K/V F16 各 2304 MiB → 4824.31 MiB OOM 的根因），并回收 K8V4 的 -4~5% prefill 税。
- 前置：fattn-stream 战役 redirected P2 项（其 REPORT.md §5）。

## TL;DR 判决（门禁表）

| 门禁 | 要求 | 实测 | 判定 |
|---|---|---|---|
| parity 8K（on vs off，512-ID 温度 0 逐 token） | 逐位一致 | **token 0 即分岔**（on=3343 vs off=264，两臂均为 K8V4@262K UB512，仅 `LLAMA_QSA_Q4V` 差异） | **FAIL** |
| parity 64K | 逐位一致 | 未达（8K 分岔后 runner 中止） | — |
| 262K PP on vs off | ≥ +3% | 8K 单点 +3.4%（654.15 vs 632.49）——但 parity FAIL 使性能数字无效 | 无效 |
| UB1024@262K 不再 OOM | 峰值空余 ≥512 MiB | 未测（parity 判死后按纪律停止） | — |
| TG 不劣化 >1% | ≥ -1% | **-19.1%**（75.70 vs 93.66；次生效应，见 §3 归因） | FAIL（次生） |

**判决：parity 红线 FAIL，判死，不合入生产。** 且实验中途（00:50）服务器整机失联（SSH/ping/ARP 全不可达，持续 >80 分钟），bug 行级定位与生产树还原均被中断——还原状态与补救清单见 §6，**生产树当前不在生产分支上**，恢复后必须第一时间执行 §6 的还原步骤。

## 1. 实施（完成，55 行，2 文件）

分支 `qsa-q4v-20261006`（隔离协议照 MWE §7）：`496ef52`（冻结生产脏基线 79M/55?）→ `dd710c8`（主体）→ `1c5fd2a`（编译修复：QSA_LAUNCH 宏元数）。编译 build-ac922 全目标通过（339/339，0 FAILED）；`libggml-cuda.so`/`libllama.so` 各含 1 处 `LLAMA_QSA_Q4V` 字符串（env 门控生效面），默认关。

| 文件 | 改动 |
|---|---|
| `ggml/src/ggml-cuda/qsa-attn.cu`（+38 行） | `qsa_raw_v4`/`qsa_load_raw_v4`/`qsa_dequant_v4`：in-kernel q4_0 V 反量化，算术镜像 gather 回退的 GET_ROWS（`((float)nib - 8.0f) * d` F32 后 RN 舍入到 F16——回退物化的是 F16 V）；kernel 模板加 `bool VQ4`（K 保持 q8_0）；dispatch 按 `v->type` 选实例；`supported()` 接受 q4_0 V（env 门控） |
| `src/models/qwen4exp.cpp`（+14/-1） | `qwen4exp_qsa_q4v_enabled()`（`LLAMA_QSA_Q4V`，默认 0）+ `sparse_ok` 的 V 类型条件放宽为 `q8_0 || (q4_0 && env)` |

关键设计（阶段 0 侦察结论）：
- **Q4V 失效点**（改动前）：`qwen4exp.cpp build_attn_qsa_blk` 的 `sparse_ok` 要求 `v_all->type == GGML_TYPE_Q8_0`（图构建层）+ `qsa-attn.cu supported()` L282 同款检查（CUDA 层）。
- **gather 回退换算式**（逐位 parity 的镜像目标，已记录）：单发 `GGML_OP_GET_ROWS` 直出 F16（非 GET_ROWS→CAST 两跳），CUDA 端 `k_get_rows` + `dequantize_q4_0`（dequantize.cuh L46-58）：`f = ((float)(qs & 0xF) - 8.0f) * d`（F32，低 nibble→dim i、高 nibble→dim i+16），写 F16 dst（`ggml_cuda_cast` else 分支 `float(x)` 隐式转换 = `__float2half` RN）。kernel 侧镜像：`__half2float(__float2half_rn(((float)nib - 8.0f) * d))`。
- q4_0 布局核对：block 18 B（half d + qs[16]），lane 8 dims = 8 连续字节的 nibble（lane%4∈{0,1}→低半字节 bytes 8*(lane%2) 起，{2,3}→高半字节），byte load 规避 18 B 块对齐问题；V cache 步长实测语义 nb[1]=144/nb[2]=288（与 kv-262k 的 V@132K=435.38 MiB 账目吻合）。

## 2. A/B 数据（262K ctx + K8V4 + UB512，生产 env，私有单元 zcode-qsa-q4v）

| 臂 | 8K PP | 8K TG | 64K PP | 64K TG | parity |
|---|---:|---:|---:|---:|---|
| off（Q4V=0，gather 回退） | 632.49 | 93.66 | 710.69 | 83.31 | vs speed-bundle-single：8K/64K 双 matched ✓ |
| on（Q4V=1，sparse kernel） | 654.15 | **75.70** | （中止） | — | vs off：**token 0 分岔** |

- off 臂 = 现生产行为的复现证据：PP/TG 与 kv-262k arm1（624.6/93.63、707.6/83.37）同量级，且与 q8/q8 生产参照 speed-bundle-single 逐位一致（复现 kv-262k 的零翻转结论）。
- on 臂 8K PP +3.4% 方向符合预期（读路径回收），但输出流分岔后该数字无意义。

## 3. 分岔归因（证据链）

1. **token 0 即分岔**（on=3343/` who`类 vs off=264=`The`）——非末位舍入差异模式。第一生成 token 由 prefill 末位 logits 决定 → on 臂 prefill 的 QSA 输出有实质数值误差。
2. **输出是流畅英文流而非乱码**（on 臂 token 序含 271/198 换行、正常词片）——误差量级是"V 数值错"而非"索引完全崩坏"的模式，但足以翻转贪心 token。
3. **TG -19% 是次生效应**：合成基准 MTP accept 应饱和 1.000（kv-262k 四臂全部 408/408）；on 臂 accept **0.89462**（399/446，mean len 4.56）——主模型 prefill 表示错 → 贪心流进不同续写族 → draft 不再全中 → TG 掉。decode 路径两臂相同（n_tok≤16 走 gather，sparse_ok 的 n_tok>16 条件未动），kv-262k 已证 decode 无损。
4. **未定位到行级根因**：手推复核了 lane→nibble 映射（lane=5/6 逐例）、算术镜像公式、cache 步长语义（nbv1=144/nbv2=288 与实测账目吻合）、mask/scale/Hadamard 对称性、sparse 节点构造唯一性（build_attn_qsa_blk 单点 + decode 不走 sparse）——均未发现错误。已备好独立单测 `test_qsa_v4.cu`（kernel dequant vs GET_ROWS 参考逐元素对拍，sm_70 单卡 10 分钟可跑），**服务器失联前未及执行**。剩余嫌疑：fork 的 block_q4_0/量化写入端布局变体、meta backend GET_ROWS 非 CUDA 执行路径、或 kernel 集成处的编译期问题（如 -use_fast_math 与 F16 intrinsic 交互）——均待单测裁决。

## 4. 过程事故：服务器整机失联（还原被中断）

- **00:50**（on 臂 8K case 完成、服务已停、GPU 已释放后约 3 分钟）起 SSH/ping/ARP 全部不可达（`arp` 显示 incomplete，整机断电/重启/断网级别，非 sshd 故障）。
- 失联时点的服务器安全状态：我方 systemd 单元 `zcode-qsa-q4v.service` 已 inactive（臂 2 正常结束停服），GPU 0 MiB 占用（臂 2 结束后确认），无 GPU 队列锁。
- 持续轮询 >100 分钟未恢复（POWER9 冷启动+内存训练可达 20-30 分钟，此时长已远超；截至本报告定稿 02:30 仍不可达）。
- **中断影响**：① parity bug 行级定位未完成；② UB1024 OOM 臂、归因臂未跑；③ **生产树还原未执行**（见 §6）；④ build-ac922 二进制为实验版。

## 5. 门禁判定

- **parity：FAIL（判死）**。on vs off 首 token 分岔，不可解释为反量化舍入（同 dtype 同数据的两臂，镜像算术的差异上界是 F32 累加次序；而 kv-262k 证明连 q8↔q4 的 V dtype 全换都不翻转 token）。分岔模式指向 kernel 数值 bug，非设计性差异。
- **性能/TG/内存门禁：无效或未测**（parity 死后按纪律停止）。
- **改动规模：52 行，受控**（未触发 >200 行判死条件）。

## 6. 还原状态与恢复后操作清单（最高优先级）

当前生产树 `sources/llama-v100-v3` 状态：分支 `qsa-q4v-20261006`（非生产分支 `ac922-flashnext-v100`），三个 commit（496ef52 冻结基线 / dd710c8 / 1c5fd2a）；`git stash list` 顶部为本战役的 `stash@{0}: On ac922-flashnext-v100: qsa-q4v-freeze-20261006`（内含生产 79 文件脏态）；55 个未跟踪生产文件从未入 git、在盘未动（其中 `qsa-attn.cu`/`.cuh` 工作树当前为**实验版**，原版备份于 `~/ac922-llama-next/experiment/zcode-qsa-q4v-20261006/qsa-attn.cu.orig`，md5 d6e43a61c226dbe935b545ff2e1e39c1）。

服务器恢复后按序执行（预计 5 分钟 + 重编 ~10 分钟）：

```bash
cd /home/debian/ac922-llama-next/sources/llama-v100-v3
# 1. 留档实验版（分支 commit 里已有，此处冗余备份工作树态）
cp ggml/src/ggml-cuda/qsa-attn.cu ~/ac922-llama-next/experiment/zcode-qsa-q4v-20261006/qsa-attn.cu.patched
# 2. 回生产分支（qsa-attn.cu 因在分支上 tracked、生产分支不存在，会被从工作树删除——预期行为）
git checkout ac922-flashnext-v100
# 3. 恢复 qsa-attn.cu/.cuh 原生产版（untracked 原版回写）
cp ~/ac922-llama-next/experiment/zcode-qsa-q4v-20261006/qsa-attn.cu.orig  ggml/src/ggml-cuda/qsa-attn.cu
cp ~/ac922-llama-next/experiment/zcode-qsa-q4v-20261006/qsa-attn.cuh.orig ggml/src/ggml-cuda/qsa-attn.cuh
# 4. 恢复 79 文件生产脏态
git stash apply   # 用 stash@{0}: qsa-q4v-freeze-20261006（不要 drop）
# 5. 对账：应回到 79 M + 55 ??，diff --stat = 79 files, 12554 insertions(+), 430 deletions(-)
git status --porcelain | awk '{print $1}' | sort | uniq -c
git diff --stat | tail -1
md5sum ggml/src/ggml-cuda/qsa-attn.cu   # 应 = d6e43a61c226dbe935b545ff2e1e39c1
# 6. 重编 build-ac922 并验证无实验符号
cmake --build build-ac922 -j32 2>&1 | tail -2
for f in build-ac922/bin/libggml-cuda.so build-ac922/bin/libllama.so; do strings $f | grep -c LLAMA_QSA_Q4V; done  # 应输出 0 0
```

注：分支 `qsa-q4v-20261006` 与 stash 保留不删（存档）；若后续会话要继续修 parity bug，从分支 checkout 或取 `qsa-attn.cu.patched`。

## 7. 工件清单

- 本地：`~/Desktop/ssh/flashnext-llama-experiment/qsa-q4v/`（本报告、`src/qsa-attn.cu` 改动版、`patch_qwen4exp.py`、`q4v_arms.sh` 驱动、`test_qsa_v4.cu` 单测-未执行）
- 服务器（失联前已落盘）：分支 `qsa-q4v-20261006`（3 commit）；`experiment/zcode-qsa-q4v-20261006/`（qsa-attn 原版备份、patch 脚本、runner 副本 matrix_runner_iso.py/benchmark_server.py、q4v_arms.sh、build1-3.log、arm1/arm2 驱动输出）；`manifests/zcode-qsa-q4v-{off,on}-262k-ub512/`（state/parity/遥测/case json）；`logs/zcode-qsa-q4v-*.log`（服务器日志含 accept 统计）
- 单元纪律：全程私有单元 `zcode-qsa-q4v.service`（未触碰共享单元）；GPU 纪律：臂间 wait-drain（min_free≥14000），失联前最后确认 GPU 0 占用；温度全程 ≤36°C。
- `qsa-q4v.diff`（对生产脏基线的净增量）：**未及生成**（失联中断）——恢复后 `git diff 496ef52..HEAD -- src/models/qwen4exp.cpp > qsa-q4v-qwen4exp.diff` + `diff qsa-attn.cu.orig qsa-attn.cu.patched > qsa-q4v-qsa-attn.diff`（或直接用分支 commit dd710c8 的 diff，其中 qsa-attn.cu 相对基线为全文件新增）。

## 8. 给 leader 的遗留判断

1. **机制方向未被证伪**：off 臂复现基线干净（parity ✓ + 数字与 kv-262k 吻合），on 臂 PP +3.4% 说明读路径回收的量级存在；死因是我这版 kernel 的数值 bug，不是"q4_0 V 进 sparse kernel"本身不可行。
2. 下一步若复活：先跑 `test_qsa_v4.cu`（10 分钟）裁决 loader/dequant 映射；若 PASS，则嫌疑收窄到 kernel 集成/编译期行为（-use_fast_math、寄存器压力下的代码重排），可用 `LLAMA_QSA_PATH_TRACE` + 单层 logits dump 二分。
3. 归因臂（读/写路径税分解）未跑——fattn-stream 的 P2 估算（624.6→~650）与本次 on 臂 8K +3.4%（632→654，同链同晚）方向一致，供参考。
