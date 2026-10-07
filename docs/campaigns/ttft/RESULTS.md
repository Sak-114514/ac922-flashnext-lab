# TTFT / 冷 prefill 专项实验 — R21 行缓存复活 + 冷启动手段实测

日期：2026-10-04 ｜ 树：`/home/debian/ac922-llama-next/sources/llama-v100-v3`（源码零改动，仅 env）｜ 硬件：ac922，6×V100，**主机 RAM 31GB**，模型 73.22GiB（PLE 表 26.82GiB IQ4_NL，文件偏移 43.81GiB）
配方：`launch_integration_arm.sh` 94.3 生产配方逐行照抄仅参数化（`scripts/launch_ttft_arm.sh`）

## 一、冷启动模拟方法验证（先于一切臂）

无 sudo（`drop_caches` 不可用）。候选法：**dd 读 40GB 文件（>31GB RAM）挤掉页缓存**，每次臂前执行（实测 72–95s/次，~425MB/s）。

验证结果（任务要求的"两轮同配置"法 + 两个对照）：

| 验证 | 结果 | 结论 |
|---|---|---|
| 挤缓存后首请求 vs 同 server 第二请求（随机 token 探针，臂0b） | 冷 PP 120.43 / 热 679.47（**5.6×**），TTFT 68.0s vs 12.1s | **dd 挤缓存法有效**，复现 R21 冷场景 |
| 基线重复性：arm0 vs arm0b（不同进程、各自挤缓存） | 冷 PP 622.82 / 623.90（差 0.2%）；热 676.09 / 685.83（差 1.5%） | 冷流程可复现 |
| **restart 对照**：热跑后仅重启 server（不 dd），首请求 | **仍然全冷**：PP 112.93 / TTFT 72.6s，第二请求才热（690.04） | 见下方关键发现 |
| `LLAMA_LOAD_PREFETCH=0`（关加载期 WILLNEED 预读） | 冷/热 PP、加载时长（140 vs 135s）全部与默认无差 | 预读开关与本问题无关 |

**关键发现 A（restart 对照的解释）**：server 加载本身要从磁盘读 ~46GiB GPU 张量，在 31GB RAM 上必然把 PLE 表页缓存挤掉。**这台机器上每次 server 重启，首个高多样性 prompt 请求必然是冷的，与之前是否热跑无关**。dd 挤缓存的作用只是让冷状态确定化（公平），并非制造"假冷"。历史担忧"重启不清页缓存"在这台 RAM < 模型的机器上不成立——加载自 churn。

**关键发现 B（基准 prompt 掩盖冷问题）**：matrix_runner 固定 prompt 是同一段落重复 ~186 次，去重后仅几十个不同 token ID → PLE 冷读只有几百个 4KiB 页 → **matrix 口径冷热差仅 ~1s / 8%**（622 vs 676）。R21 的冷病根只在**高 token 多样性** prompt（真实流量形态）下出现。因此本实验增加**随机 token 探针**（`probe_cold.py`：固定 seed 20261004，8192 个 0..248319 随机 token，8052 个不同 ID，greedy，n_predict 512；跨臂输入完全一致）作为冷场景的专门度量。matrix 臂照跑（任务规定口径 + parity 判定）。

## 二、臂设计（每臂 = 挤缓存 → 冷启动 → 首请求[冷] → 同 server 第二请求[热]）

94.3 配方之上的唯一变量（全在树内、默认关）：

| 臂 | GGML_CPU_PLE_ROW_CACHE | GGML_CPU_PLE_GET_ROWS_THREADS | GGML_CPU_PLE_DECODE_THREADS |
|---|---|---|---|
| 基线（arm0/arm0b） | 0 | 1 | 1 |
| 臂1 行缓存 | 1 | 1 | 1 |
| 臂2 多线程读 | 0 | 32 | 8 |
| 臂3 全开 | 1 | 32 | 8 |
| 臂4 range-warm（附加） | 0（基线 env）+ 启动后 dd 只读 PLE 的 26.8GiB 文件区间（63s） | | |

未入臂：`LLAMA_PLE_SHARD/SHARD_ROWS/SEPARATE`（R20 家族六卡 GPU 分片，数值分岔判缓，不碰）；`LLAMA_PLE_ONECAT`（R15 已拒，配方显式 0）。

## 三、结果

### 3.1 随机 token 高多样性探针（8K prompt，冷场景主指标；n=1/臂）

| 臂 | 冷 PP (tok/s) | 冷 PP vs 基线 | 冷 TTFT (s) | 冷总 wall (s) | 冷 TG | 热 PP | 热 TTFT (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 基线 arm0b | 120.43 | — | **68.05** | 80.8 | 39.93 | 679.47 | 12.08 |
| 臂1 行缓存 | 191.02 | **+58.6%** | 42.91（-36.9%） | 54.3 | 45.00 | 479.94（**-29.4%**） | 17.08 |
| 臂2 32r+8d | 121.86 | +1.2% | 67.25 | 79.6 | 41.21 | 685.54（+0.9%） | 11.98 |
| 臂3 全开 | 196.23 | **+63.0%** | **41.76**（-38.6%） | **50.5** | **58.55** | 500.00（-26.4%） | 16.41 |
| 臂4 range-warm | 179.84 | +49.3% | 45.57 | 52.3 | 76.29 | 666.97（-1.8%） | 12.30 |
| ctrl 重启不 dd | 112.93 | -6.2% | 72.56 | 82.2 | 53.10 | 690.04 | 11.89 |

（TG 列在随机垃圾 prompt 下 MTP 接受率波动大，仅参考；PP/TTFT 是稳健信号。TTFT = server prompt_ms + 首 decode 步均值。）

### 3.2 matrix_runner 基准（1:8192/512，任务规定口径；所有臂 parity vs speed-bundle-single **全部 matched**）

| 臂 | 冷 PP | 冷 TTFT (s) | 热 PP | 热 TTFT (s) |
|---|---:|---:|---:|---:|
| arm0（出厂配方，prefetch=1） | 622.82 | 13.16 | 676.09 | 12.13 |
| arm0p（prefetch=0） | 622.59 | 13.17 | 675.00 | 12.15 |
| arm0b（双载荷重跑） | 623.90 | 13.14 | 685.83 | 11.96 |
| 臂1 行缓存 | 625.78 | 13.10 | 685.68 | 11.96 |
| 臂2 32r+8d | 626.41 | 13.09 | 690.91 | 11.87 |
| 臂3 全开 | 573.46 | 14.30 | 690.76 | 11.87 |

重复性 prompt 下各臂冷/热基本无差（-8% 的臂3 冷是行缓存查找 + 32 线程锁竞争在该口径的小代价）。加载时长各臂 135–141s 一致。

## 四、结论

1. **R21 行缓存复活成功且同口径实测 +59~63% 冷 PP**（120.43→191/196），TTFT 68.0→41.8–42.9s（**-37~-39%**），冷请求总 wall -37%（80.8→50.5s）。历史 +49% 同数量级、本轮口径下更好（历史对照自带 32/8 readers 且更快，本轮对照为纯基线；同构对比臂3 vs 臂2 = +61%）。数值正确性：matrix 全臂 512-token parity matched（R21 fixture 结论在当前树+94.3 配方下复成立）。
2. **但行缓存有稳态税**：高多样性 prompt 热路径 PP 679→480-500（**-26~-29%**），机制是每层每行的锁+哈希+128B 拷贝（8192 行 × 49 层 ≈ 40 万次查找/请求）。重复性 prompt（几十个不同 ID）无感。**不是免费赢**。
3. **32/8 多线程读取对冷无益**（+1.2%，中性）——与 R21 历史一致（当年单独用反而 241.95 vs 单读者 288.70）。页错误仍是 64KiB 同步读放大，并行读同行不救冷。它的价值只在冷 decode（臂3 冷 TG 58.55 vs 基线 39.93）与配合行缓存摊薄 miss。
4. **prefetch 开关无影响**：加载器无论如何都全量读 GPU 张量（mmap 拷贝），PLE 冷态由 RAM 容量决定，`LLAMA_LOAD_PREFETCH` 不改变冷请求命运。
5. **range-warm（启动后 dd 预热 PLE 区间 63s）**：冷 TTFT 45.6s，**无稳态税**，等效行缓冷的七成收益；但 27.4GiB 表 vs 运行中 server 仅 ~26GiB 可用页缓存，只能部分驻留，无法全热（热 TTFT 12.3s 达不到）。
6. 生产现状定量：**每次重启后首个高多样性请求 TTFT ≈ 68s（vs 热 12s）**，随真实流量逐步变热（每请求 ~8000 行 × 4–64KiB 随机读，26GB 缓存内累积）；重复性/模板化流量则几乎无冷感。

## 五、生产建议

- **不建议把 `GGML_CPU_PLE_ROW_CACHE=1` 加入生产 preset**：它把"重启后第一个请求"的一次性 68s 换成此后**每个**高多样性 prefill 永久 -26~-29%。除非业务画像明确"频繁重启 + 只在意首请求 SLO"，才值得选择性开启（届时配 `GGML_CPU_PLE_DECODE_THREADS=8`，冷 decode 收益明显、无热税）。开关继续归档，默认 0。
- 若需消除重启冷窗：优先**带外预热**（启动脚本 health 后加 `dd if=model.gguf of=/dev/null bs=1M skip=44860 count=27465`，+63s 换首请求 TTFT 68→46s，零稳态代价，不改任何 env/源码）；全热受 RAM 31GB 限制不可达，属硬件上限。
- `GGML_CPU_PLE_GET_ROWS_THREADS=32` 单独无价值，不进配方。
- 本轮所有结论基于 8K prompt、单并发、n=1/臂（冷态受磁盘时序影响，重复性已用 arm0/arm0b 交叉验证 0.2–1.5%）；上生产前如需可再跑 64K/并发口径。

## 六、复现材料

- 脚本（ac922 `/home/debian/ac922-llama-next/scripts/`）：`launch_ttft_arm.sh`（94.3 配方 + $2 ROW_CACHE/$3 GET_ROWS_THREADS/$4 DECODE_THREADS/$5 LLAMA_LOAD_PREFETCH）、`do_arm.sh`（双载荷臂：挤缓存→随机探针冷/热→再挤缓存→matrix 冷/热）、`probe_cold.py`（固定 seed 高多样性探针）、`run_extra_controls.sh`（restart 对照 + range-warm）
- 挤缓存文件：`dd if=/dev/zero of=/home/debian/ttft-evict-40g.bin bs=1M count=40960`（已删，用时重建）
- 结果：`manifests/ttft-{arm0,arm0p,arm0b,arm1,arm2,arm3,arm4,ctrl}/`（c1-p8192.json + .parity.json + probe-rand.json + gpu-telemetry）；服务日志 `logs/ttft-arm*.log`、runner 日志 `logs/ttft-runner-*.log`
- PLE 区间：`per_layer_token_embd.weight` offset 47,039,862,208 B（43.81GiB），size 28,800,138,240 B（26.82GiB），IQ4_NL ne=[160,262144]
- 收尾：ac922-flashnext-test 已 stop，6 卡 0 MiB / 39–41°C，挤缓存文件已删，磁盘 287G 空闲；本轮未改动任何源码文件（树内 `git status` 的 common/ 等修改为 fork 自带的 pentacoxian v3 + PLE/SM70 补丁，实验前既有），唯一新增为 `scripts/` 下 4 个脚本
