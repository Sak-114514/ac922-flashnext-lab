# 1Cat 上游通信 / attention PR 审核

**快照：** 2026-10-07（Asia/Shanghai）。只读检查 `1CatAI/1Cat-vLLM` 官方 PR 页面、合并后的源码与本机现有 fork 元数据。上游 `main` 的只读 ref 为 `bc01f2899db721a99788f3bc2abc041e4b9a7fd0`。本机 `~/Desktop/ssh/opensource/super-ac922-clone` 的 `origin` 是 `Sak-114514/super-ac922`，分支 `ac922-ppc64le`，HEAD `747db5f`（2026-09-30）；它不是当前 1Cat 上游 checkout，所列 10/4–10/5 合并不在这个快照的时间范围内。

## 上游状态

| PR | 上游标题 | 2026-10-07 状态 | 与通信 / attention 的关系 |
|---|---|---|---|
| [#927](https://github.com/1CatAI/1Cat-vLLM/pull/927) | Reuse GDN tail copies without materializing QKV | 已合并（10/4） | GDN MTP 投影尾部复制，非通信 collective。 |
| [#928](https://github.com/1CatAI/1Cat-vLLM/pull/928) | Read row-strided QKV in fused MTP recurrence | 已合并（10/4） | SM70 MTP/GDN 输入布局。 |
| [#929](https://github.com/1CatAI/1Cat-vLLM/pull/929) | Reuse the MTP GDN verification output buffer | 已合并（10/4） | 重用 MTP 输出缓冲区。 |
| [#930](https://github.com/1CatAI/1Cat-vLLM/pull/930) | Screen replicated HC weight lookahead | 已关闭，未合并 | 测试候选被拒；M20 回退，保留原 schedule。 |
| [#931](https://github.com/1CatAI/1Cat-vLLM/pull/931) | Restore canonical GGUF expert model integration | 已合并（10/4） | GGUF loader/API 基础，#933 明确依赖它。 |
| [#932](https://github.com/1CatAI/1Cat-vLLM/pull/932) | Evaluate SM70 long-context attention splits | **Open / Draft** | 仅实验与 benchmark；PR 明确 production dispatch 不变、GPU 筛选 pending，256K global-Q 候选回退。 |
| [#933](https://github.com/1CatAI/1Cat-vLLM/pull/933) | Select measured GGUF expert down vectors | 已合并（10/4） | GGUF MoE/down projection，不是 attention。 |
| [#934](https://github.com/1CatAI/1Cat-vLLM/pull/934) | Record router operand selection screen | 已关闭，未合并 | FP32 router 候选在 GPU 测试前被撤回；没有可同步的默认路径。 |
| [#935](https://github.com/1CatAI/1Cat-vLLM/pull/935) | Fuse SM70 DFlash2 query preparation and attention layouts | 已合并（10/4） | 已落入 default SM70 DFlash2 路径，但严格限定 TP4、FP16、draft7、1024/2048-token cache pages。 |
| [#936](https://github.com/1CatAI/1Cat-vLLM/pull/936) | Reuse GDN tail copies on the default forward path | 已合并（10/4） | GDN MTP default forward 的 copy 路径。 |
| [#957](https://github.com/1CatAI/1Cat-vLLM/pull/957) | Reduce SM70 fused allreduce norm synchronization | 已合并（10/5，`5ad5bbd`） | TP4 fused all-reduce + Gemma RMSNorm，full-NVLink-mesh admission。 |
| [#992](https://github.com/1CatAI/1Cat-vLLM/pull/992) | Use TP4 HC shards with direct NVLink LL exchange | 已合并（10/5，`55b5e45`） | TP4 HC rank-local weight shards + direct peer LL exchange；非任意 TP4 拓扑。 |
| [#998](https://github.com/1CatAI/1Cat-vLLM/pull/998) | Reuse joint QPN staging for GGUF attention inputs | 已合并（10/5，`17310de`） | 精确 M8 的 GGUF Q/K/V 联合投影；其他 M 保留 canonical fallback。 |

这些状态是 1CatAI 上游 PR 页面状态，不表示它们已进入本机 `super-ac922` fork，也不代表在 AC922 上通过了构建或模型验证。

## 三个目标 PR 的边界

### #992：TP4 HC LL 需要可用的四卡直接 NVLink 拓扑

PR 说明支持“四卡 NVLink ring、两条 SYS 对角”，并使用两次 launch 的 LL exchange。每 rank 读四分之一 HC 权重；32-bit word 装 16-bit tag 与 FP16 值，逻辑对角数据由邻 rank 转发。[PR 说明与测试结果](https://github.com/1CatAI/1Cat-vLLM/pull/992)

合并源码的准入为 TP4、CUDA SM70、打包好的 `_C` custom ops、全组 CUDA device 可见、NVML 报告所需直接 NVLink peers，以及 native peer atomics 成功。拓扑排序需为每个 rank 找到两个直接 NVLink 邻居；随后只对 self 和这两个邻居注册共享缓冲区。[Python communicator](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/distributed/device_communicators/sm70_hc_ll.py#L29-L139)；[四 rank ring admission](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/distributed/device_communicators/sm70_ring.py#L18-L27)。CUDA wrapper 仍将设备锁定为 SM70、FP16、`[1..20,10240]` 输入和两个直接 peer pointers。[HC LL kernel guards](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/csrc/sm70_turbomind/ops/hc_ll_shard_sm70.cu#L376-L388)

本机 2026-10-03 记录的 AC922 peer 拓扑为 GPU 0/1/2 与 GPU 3/4/5 两个独立 NVLink island。[本机拓扑记录](<~/Desktop/ssh/onecat-flashnext-experiment/PP3X3-STATE.md:1>) 在这个图上，四 rank 子集无论取 3+1 或 2+2，都无法形成四卡直接 NVLink ring；#992 应报告 capability fallback，不能按上游 4×V100 成绩宣称 AC922 TP4 可用。此结论基于既有拓扑记录，本次没有实时查询硬件。

PR 作者报告：4×V100、Torch 2.10、CUDA 12.8、FP16/FP32、TP4；M5/M8 operator ABBA 改善而 M20 慢约 0.6–1.1 µs。集成模型的 teacher top-1 为 63/64，acceptance 差值 +0.482pp 且 95% CI 跨 0；C4 比较还先于最终 inactive-row 修复。应把这些当作限定拓扑/模型的上游证据，不归因成普遍并发收益。[PR 测试结果](https://github.com/1CatAI/1Cat-vLLM/pull/992)

### #957：测试拓扑是 full NV2，代码只在 full mesh 走候选

候选优化 SM70 TP4 fused all-reduce + Gemma RMSNorm。源码 dispatch 还要求 `ngpus == 4`、FP32 residual、恰好 8 tokens、`fully_connected_` 为真且 push buffers 已注册；几何或硬件不符时走既有路径。[dispatch guard](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/csrc/custom_all_reduce.cuh#L2346-L2384)

PR 测试机是 4×V100-SXM2-16GB、full NV2、Torch 2.10 / CUDA 12.8、TP4；1K/8K 每步约快 0.065ms。作者明确说明 C4 复测没有把吞吐变化归因于此改动。[PR 测试结果](https://github.com/1CatAI/1Cat-vLLM/pull/957) AC922 既有 3+3 双岛不是四卡 full mesh，不能假设该 fast route 会被准入。

### #998：可作为 rank-local attention micro-candidate，仍是精确形状专用

当前主线把 GGUF Q/K/V 的 canonical 多次投影替换为一个 SM70 shared-A/QPN launch：`M=8, K=5120, N=3584`，Q/K/V 宽度为 3072/256/256，FP16 activation/output 与 FP32 source accumulation；只准入 14 组测量过的逻辑类型组合，其余 M/未测形状走原 canonical path。[design contract / measured scope](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/docs/design/gguf_qpn_attention_inputs.md#L1-L18)；[CUDA input/shape/SM70 guards](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/csrc/sm70_turbomind/ops/gguf_qkvz_sm70.cu#L116-L180)；[specialized entry points](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/csrc/sm70_turbomind/ops/gguf_qkvz_sm70.cu#L220-L251)。Python wiring 依赖现有 GGUF/TurboMind source adapter、`native_qkv_capabilities`、GGUF packers、canonical fallback 与 graph-safe custom op 注册。[integration and fallback](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/model_executor/layers/quantization/gguf_qkv.py#L93-L176)

PR 作者报告 14 种组合通过 official GGUF FP32 reconstruction 与 1000 次 stable CUDA graph replay；非 M8 fallback bitwise 相等。kernel/operator 估算加权每轮节省约 0.617–0.622ms，但 PR 明确说明**完整模型测量待 projection-batch integration**，所以不能当作端到端收益。[PR 测试边界](https://github.com/1CatAI/1Cat-vLLM/pull/998)；[数据与测试范围](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/docs/design/gguf_qpn_attention_inputs.md#L22-L28) [correctness / model-level caveat](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/docs/design/gguf_qpn_attention_inputs.md#L52-L69)

## #927–#936 对 attention sync 的相关性

- 真正已合并的相邻 attention 路径是 [#935](https://github.com/1CatAI/1Cat-vLLM/pull/935)：DFlash2 的 Q/K RMSNorm、RoPE、KV publication 与 paged output；只支持 SM70、TP4、FP16、draft7、1024/2048 页。上游报告 4-rank full-layer Q/output/residual bitwise，21 个 GPU tests；C1 1K/8K 有小幅变化，C4 在固定 request order 下提高，但接受率区间重叠。它不是 generic QSA/任意 KV format patch。
- [#932](https://github.com/1CatAI/1Cat-vLLM/pull/932) 仍是 draft；GPU screening pending，runtime dispatch 未变。只可当研究记录，不可当可同步 production path。
- [#930](https://github.com/1CatAI/1Cat-vLLM/pull/930) 与 [#934](https://github.com/1CatAI/1Cat-vLLM/pull/934) 是明确关闭/撤回的筛选方案，不要 cherry-pick 成活跃优化。
- #927/#928/#929/#936 是 GDN/MTP 数据移动；#931/#933 是 GGUF MoE loader/dispatch。状态虽已合并，仍需按各自依赖与模型入口同步，不构成一个可以整体摘取的“attention PR 区间”。

## ppc64le、SM70 与 API 依赖

- **SM70：**三条目标路径都面向 SM70/V100；相同 V100 算力版本只满足设备代际条件。#992/#957 还受 TP 通信邻接/mesh 拓扑约束；#998/#935 受量化类型、批形状、页面大小、模型入口约束。
- **ppc64le：**目标 PR 的报告列出 V100 + Torch 2.10 + CUDA 12.8，但没有给出 ppc64le host 验证。本地主线分支为 `ac922-ppc64le`，早于这些 PR。当前上游 `setup.py` 的 pinned CUDA 12.8 PyTorch wheel helper 仅接受 `linux-x86_64` 和 `linux-aarch64`，对 ppc64le 会报无 pinned wheel；这不等于源码一定无法在 AC922 编译，但说明不能复用该默认 wheel/bootstrap 路径，必须以本机 Torch/CUDA ABI 单独构建和验证。[setup.py wheel platform gate](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/setup.py#L53-L58)
- **接口闭包：**#992 不是只加 `.cu`：有新 CMake TU、`torch.ops._C` 注册、kernel config、`torch.distributed`、`torch.accelerator`、PyNVML NVLink P2P 查询、`CustomAllreduce.create_shared_buffer` / native peer atomics / close handle。#957 改共享 `custom_all_reduce.cuh/.cu` packet metadata、wrapper/binding 与测试。#998 改 C++ operator、GGUF model-executor capability/wiring、source packer 和 fallback。只同步 kernel 会缺 dispatch、注册或 graph/runtime 路径。
- **工具链：**上游 CMake 要求 CMake ≥3.26、C++/CUDA 标准 20。[CMake source](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/CMakeLists.txt#L1-L10) 需对照本机 ppc64le 编译环境与已构建 wheel ABI；PR 上游的 CPU/4-rank GPU 测试不替代这一构建验证。

## 同步建议

1. 对 AC922 3+3 拓扑，#992 的四卡 ring 与 #957 的 full-mesh fast path 均先按 fallback / 不准入处理；除非后续只读拓扑证据显示 rank 子集满足各自 admission。
2. #998 是较适合做 rank-local operator port 的候选，但先同步 capability + canonical fallback + operator registration，再在精确 M8/14 type matrix 上比较本机 runtime；不要外推到其他 M 或整模型吞吐。
3. #935 可单列为 DFlash2 特定路径候选。#932 保持未合并研究状态；#930/#934 排除。
4. 同步前先解决目标 fork 的代码基线与 API 差异。上游 merge status 只证明进入 `1CatAI/1Cat-vLLM:main`，不证明可直接 fast-forward 到 `super-ac922`，也不证明 AC922 质量/性能正确。

## 检索边界

本轮只读 GitHub 上游与本机 fork 元数据/既有拓扑记录；未连 AC922、未运行 CUDA/GPU、未编译、未启服务。拓扑判断使用 2026-10-03 既有 3+3 记录，正式执行前应重新核对实际 rank→GPU NVLink peer matrix。
