# AC922 Flash-Next 六卡实验准备

日期：2026-10-01。本文区分本机核实、源码核实、作者报告和待实测项目。

## 当前已完成

- AC922 已开机，IPv4 默认路由正常。
- 服务器套餐订阅已更新；代理只监听 `127.0.0.1:7890`。显式下载通过代理，不启用 TUN 或系统全局代理。
- 以 Hugging Face API 作为节点自动探测目标；实际模型 CDN 的 Range 请求返回 206 和正确 GGUF magic。
- 下载器使用 6 worker、256 MiB 分块、断点账本、分块响应范围检查、完整文件 SHA256 校验和原子改名。清单见 `downloads.json`，不含订阅凭据。
- 三个原始文件均已下载完成并通过对应 HF LFS SHA256；`model-verification.json` 保存核实记录。主模型 1224 tensors，MTP 32 tensors。第二分片是完整 IQ4_NL PLE table，28,800,138,240 tensor bytes。
- 固定版本的 V100 v3 运行时已在 POWER9 / CUDA 12.4 上完成 377 项构建，退出状态 0。`llama-server` 和 `llama-bench --version` 正常，CUDA backend 链接现有 NCCL。
- 现有生产和 ACE-Step 为用户级 systemd 服务，检查时均 active。准备阶段未停止它们。

## 固定来源

| 材料 | 固定版本 | 用途 |
| --- | --- | --- |
| ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF | `ed59f92082b1e93c0e96d60a8b11aab089b52f09` | 两分片 IQ3_XXS，75,839,998,528 bytes |
| peasantsmith/qwen38-flash-next-v100-recipe | `e7208cdae7be29418f38e611ede90a35726e888f` | shared Q8 MTP，2,786,568,256 bytes |
| ggml-org/llama.cpp v0.5.0 | `7fe450e19305b828c199d602c23a8337aaa1f03b` | 运行时基底 |
| pentacoxian-dev/Qwen3.8-Flash-Next-IQ3E-Q8D-MTP-GGUF | `75e6411f0157bb4b6d492a9868bc93c57d2b061c` | `patches/llama-v0.5.0-v100-v3.patch` |
| mxxm-t/mx-llama.cpp | `eefc4e7321c869496146697d63362f073941aed6` | PLE GPU sharding 移植参照 |

V3 patch SHA256：`21784053cabd828c5244a284af860e37c61f7d46f2799543fb6a79bb76febce9`。V4/V5 作者撤回，当前不采用。

原始模型合计 78,626,566,784 bytes，约 73.2 GiB，现已完整校验。文件校验状态以服务器 `manifests/download-status.json` 为准；稀疏 `.partial` 的逻辑文件大小不是已下载字节数。

## 本机拓扑决定通信路线

NVML 和 CUDA `cuDeviceCanAccessPeer` 已实际检查：

| GPU 组 | 组内 | 跨组 |
| --- | --- | --- |
| 0/1/2 | 每对 NVLink NV2，CUDA peer access 可用 | 到 3/4/5：SYS，CUDA peer access 不可用 |
| 3/4/5 | 每对 NVLink NV2，CUDA peer access 可用 | 到 0/1/2：SYS，CUDA peer access 不可用 |

V3 的专用 P2P AllReduce 要求所有参与卡两两可访问，不能直接用于本机六卡。六卡首轮使用已有 NCCL 2.27.7：

```text
GGML_CUDA_ALLREDUCE=nccl
GGML_CUDA_P2P_AR_DISABLE=1
```

之后如通信占比确实高，单独比较 NCCL 默认与 `NCCL_P2P_LEVEL=NVL`，再比较 Ring/Tree。初始不强制 NCCL_PROTO。无需改 IOMMU、内核或重启。

### 六路 tensor split 的实际限制（源码推导，尚未运行）

该模型有 24 个 Q heads、仅 2 个 KV heads。V3 不要求 KV heads 数整除 GPU 数；按 head/量化粒度对齐分片，所以六卡等比可用于大 MoE/linear tensor，但 full attention 实际每层仅分到两路，每路 12 Q heads 和 1 KV head。其余四路为零份额，实际图运行是否正确处理空份额仍须加载测试。

640 维 expert FFN 按 32 粒度可分为 `96,96,128,96,96,128`；PLE 16 heads 也可分六路。不是所有算子都会自动六路均匀执行。若以后要六路 attention，必须显式复制 KV heads 并改 placement/cache，这是独立改动，不能作为本次最小 PLE 移植的一部分偷偷加入。

首轮仍用 `--tensor-split 1,1,1,1,1,1`。源码推导 full-attention 活跃逻辑对会轮换 `(2,5)/(3,0)/(4,1)`；可另对照 `CUDA_VISIBLE_DEVICES=0,4,2,1,3,5`，使其中两组物理配对在同一 NVLink 岛。实际设备与层分配需在加载日志核实，不能把这个建议当成已测优化。

## MTP 接入准备

已核实的 MTP 文件有 32 个 tensor，均为 `blk.48.*`，无独立 embedding/output；声明 shared target tensors。V3 没有加载这种共享 sidecar 的 target borrowing 接口，不能仅传 `-md` 就声称可运行。

主模型为 48 层，MTP 为 49 层含 1 个 nextn 层。关键维度及前 48 个 compress ratios 匹配，MTP 第 49 项为 0。`target-header-preliminary.json` 是下载期间的旧头部快照，完整校验后又读取的 metadata 以 `model-verification.json` 为准。

优先准备一个独立 GGUF 追加产物：保留主模型所有 tokenizer、PLE、量化和 sampling metadata，保留每个 tensor 的原始量化字节；追加 MTP tensors，更新 block_count=49、nextn_predict_layers=1、compress ratios。去除分片 metadata。原始下载文件保持不变。

验证必须包括：源文件 SHA256、元数据兼容、无同名 tensor、输出 tensor shape/type/逐 tensor SHA256 一致、加载成功，以及短输入 greedy 输出与关闭 MTP 的基线一致。现有 `llama-gguf-split --merge` 不能直接拼接不属于同一 split 集合的 MTP sidecar。

补充源码调查：RCO 对各 tensor 选择不同量化格式，GSQ 在原 tensor 坐标上做分组标量量化；没有发现 hidden-state basis rotation/permutation。Recipe 作者明确组合了 ISTA IQ3_XXS 与这个 shared Q8 MTP，因此目前没有需要 MTP 反旋转的证据。完整 checkpoint ancestry 不在本地 metadata 中，仍保留实际加载、greedy 对照和 acceptance 验证。

参照：[ISTA model card](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF)、[GSQ 实现](https://github.com/IST-DASLab/GSQ/blob/main/src/quantization/gumbel_quantizer_int.py)、[作者完整 recipe](https://github.com/mw00/peasant-smith/blob/main/recipes/qwen38-flash-next-v100-32gb.md)。

追加工具 `append_mtp_gguf.py` 已准备，复用固定运行时的 GGUFReader/Writer，不增加运行时加载接口。小文件 fixture 在服务器现有 Python 环境通过：Q8/F32 原始字节、tokenizer/PLE 元数据保留、异常元数据拒绝、拒绝覆盖现有输出。最终输出使用同目录 hard-link 原子安装，保留全部源文件；正式写入需提供 SHA256 清单并重新校验。

三个真实模型文件的 `--dry-run` 已通过，计划输出 1224+32=1256 tensors。记录见 `mtp-append-dry-run.json`。这一步只检查真实文件元数据，不代表真实大文件已合并或推理已验证；本阶段尚未生成集成 GGUF。

## PLE 移植范围

MX 的 GPU PLE 路径是按整 hash head 分 GPU，局部 GET_ROWS 产生各卡部分结果，使用已有 AllReduce 合并。

| 源码位置（固定 MX 源码） | 需要保留的行为 |
| --- | --- |
| `src/llama-model.cpp`，`LLAMA_PLE_SHARD` 和 PLE tensor placement | 仅 tensor split 显式启用；按 head 放置；清除 CPU lazy gather 路径 |
| `ggml/src/ggml-backend-meta.cpp`，GET_ROWS 和 reshape | 正确传播 partial 状态并在需要的位置规约 |
| `ggml/src/ggml-cuda/getrows.cu`，bounded IQ4_NL gather | 支持 160 维 PLE 行并正确屏蔽非本卡 head |
| `src/models/qwen4exp.cpp`，PLE graph / hash | 保持原始 16 个 hash index、重排和最终语义 |

16 heads 均分到 6 卡并不等字节：

| GPU | heads | PLE 估算 | 主权重和 MTP 均分下的静态合计下界 |
| --- | --- | --- | --- |
| 0/1/3/4 | 每卡 3 heads | 约 5.03 GiB | 约 12.76 GiB/卡 |
| 2/5 | 每卡 2 heads | 约 3.35 GiB | 约 11.09 GiB/卡 |

这只是静态估算，实际还包含复制 tensor、KV、context、workspace 和通信 buffer，不能把六卡总剩余空间当成每卡均匀可用。作者的 PLE sharding 速度结果来自 AMD MI50，不是本机 V100 实测。

## 后续测试顺序

1. 完成三文件完整 SHA256 校验，准备并校验独立 MTP 追加产物。
2. 记录并暂时停止占用 GPU 的旧服务，保持 BMC 风扇与保护配置；在独立实验目录/端口运行。
3. 先证明六卡 tensor split、IQ3 与 Q8 KV 能加载；以小上下文、单次短生成检查 NCCL 通信和输出。
4. 启用 MTP3，记录实际 draft acceptance、prefill、decode 和每卡显存，排除加载和输出问题。
5. 再独立加入 PLE sharding；比较同提示、同输出长度下的吞吐和输出，确认全 GPU placement，且没有 host PLE gather 残留。
6. 基础路径成立后再做 64K、256K 单次长输入和必要的并发测量。上下文标称容量不能代替真实 prefill 完成证据。

不使用 PP2xTP3；不启动 Safetensors 转换；不回到 1Cat 重造 IQ3 kernel；不在下载阶段碰现有生产 API。当前没有六卡新模型推理结果，115-140 tok/s 等数值仅为目标。

## 更快方向的优先级

1. 先处理本机跨 socket 通信：NCCL 参数对照需有测量支持。
2. 集成 MTP3，并核实 `LLAMA_MTP_N_UBATCH`、`LLAMA_MTP_PAD` 与 GPU argmax 的实际行为。
3. PLE 全 GPU 后再检查 HC fusion、FreeToken graph cache 等作者已有优化，逐项对照，避免一次改一组无法归因。
4. Q2_0 可能更快，但属于质量和量化方案变化；本轮不另下模型。更高 MTP 深度只在短输入 acceptance 支持时另测。
5. 跨 NVLink 两岛的分层 AllReduce 是后续代码研究方向；当前源码没有可直接开启的开关。

## 运行位置与日志

- 服务器实验：`/home/debian/ac922-llama-next`
- 原始模型：`/home/debian/models/qwen3.8-flash-next-gsq-rco`
- 下载用户服务：`ac922-flashnext-model-download.service`
- 下载日志：`logs/model-download.log`
- 构建日志：`logs/v100-build.log`
- 构建产物：`sources/llama-v100-v3/build-ac922/bin`
- CUDA 运行兼容包装：`/usr/local/bin/with-ac922-cuda525`
- 可复现构建脚本：本地 `build-v100.sh`，服务器 `scripts/build-v100.sh`
- MTP 工具和 fixture：服务器 `scripts/append_mtp_gguf.py`、`scripts/verify_append_fixture.py`；复用 `/home/debian/ac922-vllm/venv/bin/python` 的现有依赖

源码和许可证已固定保存。模型使用、分发和进一步开源需要遵守各自许可证；本文没有向上游提交代码。
