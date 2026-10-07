# Decode 优先迭代与验收

## 最新用户调整（2026-10-01）

用户明确取消无 MTP 对照，要求优先排查明显偏慢的吞吐。当前验收为有 MTP 的7项：单路8K/64K/128K、2/4路8K/64K，各生成512tokens。下方原14项表保留为原始计划记录，无 MTP 列不再执行。当前4×64K继续；完成后先测UB256与UB128，其他条件不变。扩大KV池先按实际cache/工作区与每卡余量计算，不打断当前实验。

目标：使用六张 V100 优化 Flash-Next 推理。最终给出有/无 MTP 的单路、2 路、4 路在 8K、64K 的 prefill/decode，以及 128K 单路数据和实际 KV 容量。旧推理、音乐服务下线；API 网关可保留。SSH、代理、现有 BMC/内存保护保留。

## 验收矩阵

每次输出 512 tokens，实际输入分别为 8192/65536/131072 tokens。两路/四路必须同时提交独立 slot 请求。

| 输入 | 并发 | 无 MTP | 有 MTP |
| --- | --- | --- | --- |
| 8K | 1 | 必测 | 必测 |
| 8K | 2 | 必测 | 必测 |
| 8K | 4 | 必测 | 必测 |
| 64K | 1 | 必测 | 必测 |
| 64K | 2 | 必测 | 必测 |
| 64K | 4 | 必测 | 必测 |
| 128K | 1 | 必测 | 必测 |

使用 `benchmark_server.py`，每路记录真实 processed/generated token 数、原生 server timings、输出 token IDs/hash。失败、缓存导致输入缩水、输出提前停止或截断不能算通过。报告 per-flow prefill/decode；整波 wall 吞吐另列，不能把各路 TPS 求和当成整波 wall TPS。

## 迭代范围

1. 原始 CPU PLE 路径验证六路 tensor split/QSA/NCCL 的实际运行，先用小输入排查加载和图运行错误。与 GGUF 生成同时运行的点仅作功能检查，不作为公平速度对照。
2. PLE GPU 分片与集成 MTP 各自验收；保留原版二进制及原始三个模型。
3. 小上下文比较 CPU/GPU PLE、默认/作者 fast paths、物理卡顺序/NCCL 方案。基线点须在构建/文件合并完成后跑，避免后台 I/O 干扰 CPU PLE。
4. 比较 MTP 2/3/4 深度（源码关闭了 request 级 speculative 参数调整，需要重启对应配置）；以实测 decode 为先。CPU、GPU、DIMM、VDD、PCIe 温度达到 58C 时本机观察程序仅停止实验 unit。
5. 用优选设置跑完整矩阵，不做额外持续压力循环。原生 single-slot QSA pooled indexer cache 与多 slot 路径不同，需要分别记录 server slots 配置。

## KV 容量口径

记录加载日志中的 KV/indexer/recurrent/compute buffer bytes、`n_ctx`、`n_seq_max`、`n_ctx_seq`、是否 unified、`/props` 和 `/slots` 的实际容量，以及每卡 VRAM used/free。

首轮单路拟用 132096 token 池；多路拟尝试 unified 266240 token 池、4 slots、每 slot cap 132096。后者需实际加载和 4x64K 完成才能宣称可用，OOM 时按日志分辨权重、KV 或工作区问题后调整。

MTP context 与 target context 的 KV 独立记录。不能把可配置的 token 上限宣称为已测的最大物理容量，也不能把 llama.cpp context 数直接换成 vLLM 的共享 KV block 池。

## 当前独立运行任务

- MTP 集成：`ac922-flashnext-mtp-append.service`，完成前独立输出不存在，只写临时文件；最终 metadata/tensor SHA 校验后原子安装。
- PLE 编译：`ac922-flashnext-ple-build.service`，已成功完成。
- GPU 实验：`ac922-flashnext-test.service`，仅绑定 `127.0.0.1:8002`。
- 本机温度观察：`watch_test_temperature.py`，读已有 BMC guard 函数，不调用写入逻辑。

结果存入 `manifests/` 和 `logs/`，最终汇总实测，不用作者双卡吞吐代替六卡结果。
