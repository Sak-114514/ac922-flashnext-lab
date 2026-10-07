# Qwen3.8-Flash-Next → Luce 移植评估报告

- 调研日期:2026-10-03
- 仓库:github.com/Luce-Org/luce-megakernel(浅克隆于 ~/Desktop/ssh/opensource/luce-megakernel,HEAD `2d9d4c0`,2026-10-03,PR #806,开发极活跃)
- 评估对象硬件:IBM AC922(POWER9 ppc64le)+ 6×V100 SXM2 16GB(NVLink1 mesh,sm_70,无 BF16 tensor core/PDL/cp.async),CUDA 12.4,32GB RAM
- 评估对象模型:Qwen3.8-Flash-Next GSQ-RCO(qwen4exp 架构,48 层 = 36 GDN + 12 QSA,512 选 10 MoE + 共享专家,PLE 稀疏嵌入表 28.8GB IQ4_NL,1 层 MTP 草稿,GGUF 73.2GB,IQ3_XXS/IQ4_NL/BF16 混合)
- 现状基线:llama.cpp fork 83 tok/s decode(6 卡 TP+岛+MTP),GPU busy ~65%,每 token ~百次 launch,BF16 matvec 占 33% kernel 时间

---

## 0. TL;DR 判决

| 选项 | 可行性 | 估算 | 结论 |
|---|---|---|---|
| **A. 把我们的模型移植进 Luce `server/` 引擎** | 可行,但工作量是"移植一个新 backend"级别(参照 qwen35 19.4k 行 / deepseek4 34.8k 行) | **75–135 人日**(分 5 阶段) | 值得做只有当目标是全面替换引擎;PLE 与 QSA-indexer 是全新子系统 |
| **B. 把 optimizations/megakernel 单 kernel 融合用于我们的模型** | 不可行(实用意义上) | 6–12+ 个月研究级项目 | megakernel 是单模型(0.8B、稠密、BF16、batch=1、单卡)研究 PoC,README 自己声明 "does not generalize without rewriting" |
| **C. 借 Luce 的技术回我们的 llama.cpp fork**(推荐) | 可行 | **15–30 人日**起 | moe-fused 标量量化专家 kernel、GDN chunked/SpecLA verify、三层专家缓存+预取、CUDA graph 缓存思路,都能独立摘取 |

**最重要的一个事实澄清**:"Luce 官方支持 Volta sm_70"指的是 **`server/` 生产引擎**(README.md:622 `70 = V100 (F16 WMMA kernels, BF16 draft → FP16 at load)`;CMakeLists.txt:309 默认架构表 `"70;75;86"`),**不是** `optimizations/megakernel/`。megakernel PoC 的代码级 sm_70 支持(Volta/Turing FP16 路径)确实存在于守卫中(kernel.cu:3),但它只服务于 Qwen3.5-0.8B 这一个硬编码模型。

---

## 1. 仓库全景:两个东西,别混淆

```
luce-megakernel/                 (即 README/PR 里的 "lucebox" 项目)
├── optimizations/megakernel/    ← 研究 PoC:单 kernel 融合,仅 Qwen3.5-0.8B(8.4k 行)
│   ├── kernel.cu                decode megakernel,1048 行,全部 24 层一个 launch
│   ├── prefill.cu               prefill(cuBLAS+独立 kernel),WMMA 仅 sm_80+
│   └── README_PASCAL.md         Pascal sm_61 降级路径(证明架构可移植性)
└── server/                      ← 生产引擎:插件式 backend,sm_60+ 官方支持
    ├── src/qwen35/              Qwen3.5/3.6/3.8 混合 GDN+注意力家族(19,439 行)
    │   └── qwen35_tensor_parallel.cpp   2×3090 NVLink TP 的实现
    ├── src/qwen35moe/           GDN 混合 + MoE(继承 Qwen35Backend)— 与我们最接近
    ├── src/deepseek4/           DeepSeek V4/V4.1(MLA+稀疏注意力 indexer+MoE+Engram,34,822 行)
    ├── src/common/moe_hybrid_*  模型无关的 MoE 双/三层专家运行时
    └── deps/llama.cpp/          vendored ggml fork(116 个算子,含 GDN/DS4 专用算子)
```

一个关键发现:**Qwen3.8-27B(2×3090 TP 79.7 tok/s 那个)本身就是 qwen35 backend 跑的**——server/README.md:639-641 说明 Qwen3.5/3.6/3.8-27B "arch is identical"。也就是说我们模型所属的 Qwen3.8 家族在 Luce 里是现成架构,差的只是 Flash-Next 的增量(GSQ-RCO、QSA indexer、PLE、512 专家 MoE)。

---

## 2. (a) 新增一个模型架构需要写什么

### 抽象层结构(docs/ARCHITECTURE.md,443 行,含完整移植指南)

插件式 backend:每个模型家族实现 `ModelBackend` 接口(`server/src/common/model_backend.h`,接口全文见 ARCHITECTURE.md:53-133),一个通用 daemon loop(`common/daemon_loop.{h,cpp}`)以 stdin/stdout 协议驱动所有 backend。可选 `DFlashTarget` 接口(ARCHITECTURE.md:149-171,9 个虚函数)接入投机解码。

### 注册点(3 处,有编译期防错)

1. **`server/src/common/backend_factory.cpp`**:按 GGUF `general.architecture` 字符串分发(第 165/230/260/273/312/326 行的 `arch == "qwen35"/"qwen35moe"/"bailingmoe3"/"laguna"/"qwen3"/"gemma4"` 链);`LUCE_CHECK_ARCH`(第 97-101 行)用 `static_assert` 强制 capability 表(`model_capabilities.h`)与 config 结构体一致——**表里声明了支持但结构体没有对应字段会直接编译失败**。
2. **`server/CMakeLists.txt`**:加源文件。
3. capability 表一行(标注 spec-decode/layer-split/FA-window 等支持级别)。

### 以参照物估计必写文件(qwen35 = 混合 GDN 参照;deepseek4 = MoE+稀疏注意力参照;qwen35moe = GDN+MoE 混合参照,它证明这条组合路径已存在)

| 组件 | qwen35 参照行数 | deepseek4 参照行数 | 我们(qwen4exp)预估 |
|---|---|---|---|
| GGUF loader(张量名→权重结构) | gguf_target_loader.cpp 1,161 | deepseek4_loader.cpp 2,903 | 1,200–2,000(QSA+PLE 元数据多) |
| 计算图 builder(decode+prefill) | qwen35_target_graph.cpp 3,632 + graph_builders.cpp 1,031 | deepseek4_graph.cpp 12,538 | 3,500–6,000 |
| backend 主体(生成循环/缓存/快照) | qwen35_backend.cpp 4,460 | deepseek4_backend.cpp 5,121 | 2,000–4,500(可分期) |
| daemon 入口 | qwen35_daemon.cpp(薄) | 同 | ~150 |
| layer-split 适配器(多卡流水,可选) | qwen35_layer_split_adapter.cpp 1,456 | deepseek4_layer_split_adapter.cpp 977 | 800–1,500(可选,后期) |
| 投机解码适配器 | qwen35_dflash_target.cpp 1,276 | deepseek4_dspark_spec.cpp 1,729 | 800–1,500(MTP) |
| TP 切分表 | qwen35_tensor_parallel.cpp 276 | — | 200–400 |
| PLE 表子系统(全新,无直接等价) | — | Engram 274+197(参照) | 600–1,200 |
| 测试 | tests/test_* | 同 | 500–1,000 |
| **合计** | 19,439 | 34,822 | **≈8,000–13,000 新增/改写行** |

`moe_hybrid.md:158-178` 给出 MoE 接入契约(5 步薄适配器,不需要再写调度器):填 `MoeHybridConfig`(n_embd/n_expert/n_expert_used/n_ff_exp/n_ff_shexp)、每层 `MoeLayerDesc`(gate/up/down 张量+可选共享专家+scale)、暴露 router ids/weights 布局、注册 metadata loader、backend 只在 routed-FFN 边界调用公共求值器。**"no fixed 256-expert architecture limit"(moe_hybrid.md:178),512 选 10 直接支持。**

---

## 3. (b) megakernel 的 sm_70 可用性

### 架构守卫(kernel.cu,直接引证)

- **kernel.cu:1-11** 文件头:"Supports Pascal (sm_6x), Volta/Turing (sm_70-75), and Ampere+ (sm_80+). BF16 for sm_80+, F16 for sm_6x-sm_75."
- **kernel.cu:20-30**:`TARGET_SM >= 70` → `__shfl_sync` + `fence.acq_rel.gpu`(均为 Volta 原生);Pascal 才降级到 `__shfl`/`membar.gl`。
- **kernel.cu:185-200** `load_128bit`:sm_80+ 用 `ld.global.L1::no_allocate`,**sm_70/75 用 `ld.global.cg.v4.b32`**,Pascal 用 `__ldg`。三级守卫齐全,sm_70 路径现成。
- **half_type.h:17-29**:`TARGET_SM >= 80` 用 BF16,否则 `__half`。V100 走 FP16 + F32 累加。
- **sm_80+/90 特性使用:零**。全文 grep 确认无 PDL(griddepcontrol)、无 cp.async、无 TMA/cluster/wgmma、无 cooperative groups launch。网格同步是**自制原子屏障**(kernel.cu:139-163 `AtomicGridSync`:atomicAdd 计数 + volatile 自旋),启动是普通 `<<<>>>` launch + 占用率钳制(kernel.cu:955-978 `query_max_safe_decode_blocks_impl`,通过 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 保证全驻留避免死锁)。**这套机制在 sm_70 上完全成立。**
- prefill.cu:705:WMMA phase2 `#if TARGET_SM >= 80`;且 setup.py:43 显示 `MEGAKERNEL_DN_PHASE2_WMMA` 默认 0(连 3090 都默认关),Volta 走标量路径。
- setup.py:20 默认 `sm_75`,sm_70 需 `MEGAKERNEL_CUDA_ARCH=sm_70`(注意 setup.py:37 `is_pascal = 60<=sm<70`,sm_70 不算 Pascal,NUM_BLOCKS 默认 82,会被占用率查询钳到 V100 的 80)。

### 结论:sm_70 上能不能跑?

**PoC 本身:能,改动≈0**(设 TARGET_SM=70 编译即可;decode 是纯标量 FP16/FP32 数学,无 tensor core 依赖;共享内存 16KB、512 线程/block 都远低于 V100 上限)。README_PASCAL.md 证明的降级路径(Pascal 连 shfl_sync 都没有都能跑)说明 Volta 只会更顺。

**但对我们毫无直接用处**,因为 README.md「Scope and limitations」明确:
- batch size 1、单模型单架构(kernel.cu:37-89 全部 constexpr 硬编码:HIDDEN_SIZE=1024、24 层、LAYER_TYPE 表);
- 无量化支持(仅 BF16/FP16 权重);
- 稠密 MLP,无 MoE;
- 单卡。
- "Megakernel fusion benefits shrink as model size grows"(README 原文)——我们 73GB MoE 模型的 decode 是访存受限 + 专家路由型,把 512 专家路由、PLE 查表、36 层 GDN 状态全部塞进一个 persistent kernel 是重写不是移植,且 V100 80 SM/900GB/s 的收益上限远低于 3090 上 0.8B 的场景。**判:不采用,仅作设计参照。**

### V100 上真正可用的部分:server/ 引擎

- server/README.md:622:构建指导含 `70 = V100 (F16 WMMA kernels, BF16 draft → FP16 at load)`;630:默认架构表含 70;666:需求 "NVIDIA sm_60+ GPU (… V100 32GB …) CUDA 12+";794:"Volta/Turing (sm_70-75) use F16 WMMA kernels"。
- CMakeLists.txt:295-324:CUDA 默认 `_luce_archs "70;75;86"`。
- vendored ggml 的关键 kernel 对 sm_70 的覆盖(逐文件查过 `__CUDA_ARCH__`):
  - `fattn-wmma-f16.cu:3-4` 注释:"It is still needed for **Volta** since the memory layout of NVIDIA tensor cores changed with Turing" —— **Volta 有专用 F16 WMMA flash attention 路径**(V100 有 F16 tensor core,这正是它与 Pascal 的差别);
  - `gated_delta_net.cu`:纯标量 F32 warp 协作,零架构守卫(仅 grouped-cols 优化要求 Ampere+,Volta 自动回退标量 launch,gated_delta_net.cu:18-29);
  - `moe-fused.cu` / `paged-attn.cu` / `ssm-conv.cu` / `ssm-scan.cu` / `mmid.cu`(MUL_MAT_ID):零 `__CUDA_ARCH__` 守卫,标量/vecdotq 式实现;
  - `ds4-indexer.cu:960-963`:`wmma_capable = cc >= GGML_CUDA_CC_VOLTA` —— V100 恰好达标;
  - 量化 dequant(vecdotq/mmvq)是上游 llama.cpp 代码,Volta 一直支持(含 IQ3_XXS/IQ4_NL)。
- **未确认**:V100 上的任何公开 benchmark(server/RESULTS.md 无 V100 条目;官方支持≠官方测过性能)。

---

## 4. (c) GDN/DeltaNet 支持深度 —— 完整度高,与我们的 36 层 GDN 几乎一一对应

### 算子层(vendored ggml,116 算子)

- 原生算子:`GGML_OP_SSM_CONV`、`GGML_OP_SSM_SCAN`、`GGML_OP_GATED_LINEAR_ATTN`、`GGML_OP_GATED_DELTA_NET`(deps/llama.cpp/ggml/include/ggml.h:579-589);
- Luce 扩展算子:`ggml_ssm_conv_tree` / `ggml_ssm_conv_step` / `ggml_ssm_conv_tree_step` / `ggml_ssm_conv_specla`(ggml.h:2956-3019)——为树形投机验证加的。

### CUDA kernel 层

- `gated_delta_net.cu`(1,140 行):标量 F32、warp-per-column、每 (head, sequence, column) 一个线程组;模板参数 `<S_v, KDA, TREE_MODE, WRITE_INTER>`;支持:树形 parent_ids 验证(gdn-replay-log.cu:31 `GGML_GDN_TREE_ROOT_PARENT`)、**槽位映射的分页状态**(gated_delta_net.cu:78-105,连续批处理)、**raw-gate 模式 `gate_bias`(dt_bias)+`gate_A`(A)** —— 正是我们的 A_log/dt decay 入口、F16 持久中间态缓冲(投机回滚用,gated_delta_net.cu:38-52)。
- `ssm-conv.cu`(726 行):depthwise causal conv + conv state,即我们的 ssm conv state。
- `delta_net_chunked.cpp`:llama.cpp `build_delta_net_chunking` 的移植(文件头注释:顺序 kernel 在 n_tokens=16-32 时 ~50ms/forward,chunking 重写为 matmul/tri/cumsum 图,verify 提速 2-3×)。
- `delta_net_specla.{h,cpp}`:SpecLA 拓扑掩码树形验证(单窗口 UT-transform,不推进状态、输出 v_new/g_ps 供提交侧推进)——36 层 GDN 上的多 token 验证有现成方案。

### 图/模型层(qwen35)

- 张量命名与我们的 GDN 对照(gguf_target_loader.cpp:24-44 文档化):`attn_qkv`(Q/K/V/beta 融合 in_proj)↔ 我们的 QKV+beta in_proj;`attn_gate`(z 投影)↔ 我们的 Z;`ssm_conv1d` ↔ conv state;`ssm_a`(dt_rank=48)↔ A_log;`ssm_alpha`/`ssm_beta` 投影 ↔ decay/beta 门;`ssm_dt.bias` ↔ dt bias;`ssm_norm`(gated RMSNorm)/`ssm_out`。**基本是 llama.cpp qwen3-next 命名,Luce 按此命名实现了完整语义(decode/chunked prefill/树验证/快照回滚)。**
- 维度全部读 GGUF 元数据(internal.h:234-238 默认:d_state=128、n_group=16、d_conv=4、dt_rank=48);internal.h:483 注释 "ssm_state size = n_delta_layers (**48**)" —— 48 层全 GDN 的模型已在支持范围。
- 36+12 混合布层:`full_attention_interval` 驱动(qwen35_tensor_parallel.cpp:47-49),任意 interleaving 可配。

### 差距评估

| 我们的需求 | Luce 现状 | 差距 |
|---|---|---|
| QKVZ in_proj | attn_qkv + attn_gate 分离两矩阵 | 若我们 GGUF 里是单融合矩阵,loader 里切 view 即可(小) |
| A_log + dt decay | ssm_a + ssm_dt.bias + raw-gate kernel 路径 | 语义一致(小);GSQ-RCO 特有的 decay 变体**未确认**,需对照我们 llama.cpp 实现逐条 diff |
| ssm conv state | ssm-conv.cu + conv_state 张量 | 一致(无) |
| 36 层 GDN decode | 完整(含 TP 切分 ssm_state,见 qwen35_tensor_parallel.cpp:129-143) | 无 |
| GDN 上的投机验证 | TREE_MODE kernel + chunked + SpecLA 三条路径 | 无 |

---

## 5. (d) MoE + 专家分层

### 路由与计算

- decode 路由走标准 ggml `MUL_MAT_ID`(`mmid.cu`,零架构守卫,任意 ggml 量化格式的专家权重都能吃);DS4 的融合验证路径另有 `moe-fused.cu`(1,102 行),内置 `dot_iq2_xs`(moe-fused.cu:21)/`dot_iq3_xxs`(moe-fused.cu:41)标量 dequant 点积 —— **IQ3_XXS 专家权重有专用融合 kernel,IQ4_NL 走 vecdotq 标准路径**。
- qwen35moe backend(src/qwen35moe/,继承 Qwen35Backend)+ `qwen35moe_pipelined_decode.cpp`:逐层缓存 DeltaNet pre-FFN 图(CachedPrefnGraph,"30/40 层缓存、全热层跳过冷路径"),把每层同步开销压掉 —— 这正对我们"百次 launch/token、GPU busy 65%"的痛点。
- top-k 路由图在 graph builder 里完成;`MoeHybridRoutingStats` 统计每层专家激活计数(可导出 CSV 热度画像)。

### 两层(GPU+CPU / GPU+GPU)混合运行时(moe_hybrid.md,412 行)

- 组成:`moe_hybrid_placement`(贪心热度/字节预算分配)+ `moe_hybrid_swap_manager`(请求边界换入换出)+ `moe_hybrid_storage`(mmap 直读 GGUF 切片进 owner 缓冲,不经过整栈驻留)+ `moe_hybrid_ffn_eval`(热/冷并发执行 + GPU 残差合并)+ `CachedFfnGraph`(避免每 token 重建图,"全热时每 token 仅 ~64 字节 PCIe" ,moe_hybrid.md:186-259)。
- 模型无关:"no fixed 256-expert architecture limit"(moe_hybrid.md:178);SwiGLU 族激活,别的激活只需加一个 policy。

### 三层(VRAM/RAM/SSD)(DS4.1,docs/DS41.md)

- 机制(DS41.md:83-140):每专家归属 primary/secondary/stream 三选一,placement JSON 按 `layer * n_expert + expert` 索引;**router bias**(f32 [n_layer][n_expert] 二进制,选择时加偏置把流量从 SSD 专家引开,校准流量上 SSD 调用 3.8%→0.5%);**protected experts**(命中必选且常驻 primary);流式层走 `moe_hybrid_expert_cache.cpp`(1,122 行):固定 slot 池、**当前层路由时用同一输入预测 L+1 层专家并预取**(预测只提前搬字节不算答案)、4 loader 线程 pread、长提示 O_DIRECT bulk 模式、host-mapped mailbox 让融合图在流式分支前等待答案。
- 实测(他们的硬件):383GB 模型、158GiB 专家流式跑 25 tok/s decode、cache hit 93.4%。
- **对我们的适配性判断**:AC922 的 32GB RAM 放不下 73GB 模型,RAM 层容量有限;但 GPU↔CPU 走 NVLink2(非 PCIe),CPU 侧 POWER9 无 AVX512 量化 kernel(ggml VSX 慢于 x86 数倍,未确认具体倍率),**CPU 冷专家计算路径在 POWER9 上不划算;SSD 流式层 + 6 卡 NVLink mesh 上的 GPU 侧 slot cache 才是可用的组合**。6×16GB=96GB,73GB 权重理论上可全 VRAM 驻留(我们现在 llama.cpp 就是这么干的),分层是超长上下文/多请求时才需要的能力。

### 量化格式

- vendored ggml 是完整 llama.cpp fork(ggml.h:402-443):**IQ3_XXS=18、IQ4_NL=20、BF16、Q8_0、Q4_K_M 全部原生**;另有 Luce 自定义 ROCmFPX 系(qtype 103/105/106,AMD 专用,与我们无关)。**我们的 GGUF 格式无需任何转换即可被 loader 读**(前提:qwen4exp 的元数据键和张量名在 loader 里映射好)。

---

## 6. (e) 多卡 TP

### 实现(2×3090 那套)

- `qwen35_tensor_parallel.cpp`(276 行)+ vendored ggml 的 **meta backend**(`ggml-backend-meta.cpp`,2,430 行):创建一个"元设备"聚合 N 张卡,按**张量名规则表**给出切分轴/粒度(如 `attn_q` 沿 axis1 按 head 组粒度切、`attn_output` 沿 axis0 切、`ffn_gate/up` axis1 / `ffn_down` axis0、ssm_state/conv_state 按 head 切、LM head 保持镜像直到有分布式 argmax 归约,qwen35_tensor_parallel.cpp:91-145)。
- 归约:优先 backend 专属 `comm_allreduce`(CUDA = **NCCL**,ggml-cuda.cu:1747,大张量自动压 BF16 归约),否则 generic fallback(ggml-backend-meta.cpp:2249)。**NCCL 对 AC922(POWER9+V100 NVLink)是官方一等公民——Summit 同构**。未确认点:NCCL 需在 ggml 构建时开启(`GGML_USE_NCCL`),ppc64le 下编译 NCCL 需自建或发行版包。
- 设备数上限 `GGML_BACKEND_META_MAX_DEVICES = 16`(ggml-backend.h:392)—— 6 卡结构上无障碍。
- 元 backend 还支持 MoE 专家按 owner 切分时的延迟归约(ggml-backend-meta.cpp:2047/2131 注释)。

### 对 6 卡的扩展性评估

- 代码层面:TP 分类表是 qwen35 张量名专用的,我们要写自己的 `qwen4exp_tensor_parallel` 表(~200-400 行,纯机械);meta backend/NCCL 归约/NVLink P2P(`common/peer_access.{h,cpp}`)全部复用。
- 性能层面:V100 NVLink1 mesh 每对 ~25GB/s(未确认具体拓扑开销),TP 每 layer 2 次 allreduce(hidden≈4k 的 F32/BF16);6 卡 allreduce 环 ~3× 单跳延迟。83 tok/s 时每 token 12ms 预算,48 层×2×(4k×4B×2)≈1.6MB/token 的归约流量在 25GB/s×多链路下 <0.5ms,可行。
- **注意**:TP 目前只接在 qwen35 家族 backend 上(deepseek4 用的是层分割 IPC + 专家 owner 并行,不同路线);我们的 backend 要自己接(照抄 qwen35 的接线,qwen35_backend.cpp 里 TP 上下文创建调用)。
- Luce 还有另一条多卡路线:layer split(层分割流水,`layer_split_*` + IPC daemon),适合异构卡;我们 6×同构 V100 用 TP 更对口。

---

## 7. (f) PLE 类稀疏查表

**没有 PLE 等价物,但有一个机制上高度相似的参照:DeepSeek V4.1 的 Engram。**

### Engram(deepseek4_engram.{h,cpp} 274 行 + deepseek4_engram_apply.cpp 197 行)

- 机制(文件头注释 1-23 行 + DS41.md):
  1. **寻址**:最近 4 个压缩 token id 做 n-gram 哈希,每层得 `cols` 行(发布模型:3 阶 n-gram × 8 头 = **24 行/token/层**)——纯整数数学,forward 前就能算出所有需要的行;
  2. **表读取**:行存在 GGUF 内 `blk.N.engram_embd`(I8 [264, rows]:256 个 E4M3 + 8 个 E8M0 块 scale),**按需 pread,从不 mmap**(DS41.md:18-20:189GiB 的表逐行读);
  3. **应用**:wkv 把拼接的行投影成 4 把 key + 1 个共享 value,每份 hyper-connection copy 按归一化点积门控接收。
- 与 DS41 的三层专家共享 platform_io/host mailbox 基础设施。
- 回滚:寻址状态是可拷贝 struct(投机拒绝即恢复,deepseek4_engram.h:31-38)。

### 映射到我们的 PLE

| 维度 | Engram | 我们的 PLE |
|---|---|---|
| 表规模 | 189 GiB(文件内) | 28.8GB IQ4_NL(16 头 × 2000 万行) |
| 每 token 读 | 24 行 | 16 行 |
| 行格式 | I8(256×E4M3+8×E8M0) | **IQ4_NL 行**(需要新行解码器) |
| 寻址 | token 历史 n-gram 哈希(CPU 整数) | **模型给出的稀疏索引(未确认具体来源:router/哈希/学习索引)** |
| 应用 | wkv 投影+门控 | PLE 特有(需对照我们 llama.cpp 实现) |

**工作量判断:中等(600-1,200 行)**。可以直接抄的:行级 pread 基础设施(platform_io)、"从不 mmap 巨表"的纪律、槽位缓存/预取模式(expert cache 同款);要新写的:IQ4_NL 行 gather 解码(或直接用 ggml `GET_ROWS`,llama.cpp 里 IQ4_NL getrows 本来就有,我们 fork 已验证)、PLE 寻址与应用数学。**注意 28.8GB 表 6 卡分布还是每卡复制需要设计**(每 token 只读 16 行 → 按 head 分片到 6 卡 + allgather,或 NVLink 远端读)。

---

## 8. 投机解码(MTP)与 QSA 补充

- **MTP 草稿**:DeepSeek V4.1 的 DSpark 就是 checkpoint 里 `mtp.*` 块转 GGUF 草稿器(deepseek4_dspark.h:1-27:3 阶段、block-of-5、Markov head + 置信门、embedding/lm_head 与 target 共享;转换脚本 `scripts/convert_ds4_dspark_draft_to_gguf.py`)。**我们的 1 层 MTP 有直接同构参照**;且 qwen35 家族还有另一条 DFlash(跨注意力草稿,架构无关)+ DDTree 动态树 + GDN 树形验证(TREE_MODE kernel)全套。
- **QSA(稀疏注意力+indexer)**:deepseek4 有完整参照——`GGML_OP_DS4_INDEXER_QAT/SCORE/MASK`(ggml.h:610-616,融合 Hadamard/打分/top-k 掩码)、`fattn-sparse.cu`(232 行,稀疏 KV 选取的 attention)、indexer top-k=512(deepseek4_internal.h:267)。V100 上 indexer WMMA 检查恰好放行(ds4-indexer.cu:961)。我们的 QSA 具体稀疏模式(GSQ-RCO 特有)与 DS4 的压缩行选择不同,**需移植数学、可复用算子骨架**。

---

## 9. 工作量分解(按工作量排序,Route A:移植进 Luce server)

| # | 工作项 | 估算(人日) | 依据 |
|---|---|---|---|
| 1 | **ppc64le + V100 构建打通**(CUDA 12.4 ppc64le、NCCL、immintrin.h 无条件 include 检查:deepseek4_graph.cpp:53(使用处有 x86 守卫 4849,include 本身在 GCC/Clang ppc64le 通常无害,需实测)、跑通官方 qwen35 模型冒烟) | 5–10 | 服务器无 ppc64le CI,首个吃螃蟹 |
| 2 | **qwen4exp backend MVP**:loader + GDN 图 + MoE(全热)+ PLE 基础版(同步 GET_ROWS) + AR decode | 30–50 | qwen35 家族可抄 70%,QSA/PLE 新写 |
| 3 | **QSA indexer/稀疏注意力移植** | 10–20 | DS4 算子骨架在,数学重写 |
| 4 | **MTP 投机解码**(草稿转换 + 验证图 + GDN 树验证接线) | 10–15 | DSpark/DFlash/SpecLA 三参照 |
| 5 | **6 卡 TP**(切分表 + NCCL allreduce + 占用率调优) | 10–20 | meta backend 现成,表 + 调试 |
| 6 | **PLE 完整版**(缓存/预取/多卡分布)+ 三层专家(可选,长上下文时) | 10–25 | expert cache 模式移植 |
| 7 | 测试对齐(与 llama.cpp fork 输出逐 token 比对) | 贯穿,含上 | — |
| | **合计** | **75–135 人日**(1 名资深工程师 3.5–6.5 个月) | |

**Route C(借技术回 llama.cpp fork)分解:**

| # | 工作项 | 估算(人日) | 预期收益 |
|---|---|---|---|
| 1 | moe-fused 式标量 IQ3_XXS/IQ4_NL 专家点积 kernel(替代逐专家 mul_mat launch) | 5–10 | MoE launch 数大降 |
| 2 | CachedPrefnGraph/CachedFfnGraph 式图缓存 + CUDA graph(每 token 重建图 50-200µs 的消除) | 3–8 | GPU busy 65%→80%+ 方向 |
| 3 | GDN chunked verify 移植(若我们 MTP 验证仍走顺序 kernel) | 3–5 | verify 提速 2-3×(delta_net_chunked.cpp 头注释) |
| 4 | BF16 matvec → FP16(WMMA)+ our 33% kernel 时间的重分布(参照 fattn-wmma-f16 Volta 路径) | 5–10 | matvec 段提速 |
| 5 | 专家 slot cache + L+1 预取(若引入 CPU/SSD 层) | 10–15 | 长上下文多请求场景 |

Route C 首月即可拿到 1-3 项收益,风险极低,不废弃 83 tok/s 基线。

---

## 10. 风险清单

| # | 风险 | 等级 | 说明/缓解 |
|---|---|---|---|
| 1 | **ppc64le 编译**(server 自身代码):immintrin.h 无条件 include(deepseek4_graph.cpp:53;使用有 x86 守卫但 include 未守卫)、可能的 x86 汇编/jemalloc/平台假设;ggml CPU 后端 POWER9 VSX 性能弱(冷专家 CPU 路径不可用) | 中 | 逐文件过编译;CPU 冷路径直接放弃,全 GPU 驻留(96GB 足够 73GB) |
| 2 | **NCCL on ppc64le**:需自编 NCCL(GGML_USE_NCCL);TP 的快速归约依赖它,fallback 路径(peer copy)性能未知 | 中 | Summit 架构背书;先行验证单机 6 卡 NCCL allreduce 微基准 |
| 3 | **V100 性能无官方数据**:server 支持 sm_70 但 RESULTS.md 无 V100 条目;F16 WMMA flash attention 在 Volta 的吞吐、IQ 量化 vecdot 标量 kernel 的带宽利用率都未确认 | 中高 | Phase 1 冒烟即测;83 tok/s 基线在手,任何阶段可退 |
| 4 | **megakernel 不可用于我们**(详见 §3) | 确定 | 不列入计划,仅作参照 |
| 5 | **QSA/GSQ-RCO/PLE 数学移植正确性**:Luce 无现成实现,我们 fork 的实现是唯一真值;Luce 测试纪律严格(如 DS41 因 sm_121 上 6/80 查询与参考不符直接拒绝 CUDA 构建,DS41.md:77),对齐成本高 | 中高 | 以我们 llama.cpp fork 为 golden,逐层逐 token 比对 |
| 6 | **上游漂移**:HEAD 日均多 PR(#806 今天),vendored ggml fork(commit 2d9d4c0)与上游 llama.cpp 的合并策略未知;我们 fork 的 qwen4exp GGUF 键若上游不同步,合并冲突自理 | 中 | 锁 tag;vendor 补丁保持独立分支 |
| 7 | **6 卡 TP 未被官方测过**(官方 TP 数据仅 2×3090,PR #637);V100 NVLink1 mesh allreduce 延迟敏感 | 中 | meta backend 支持到 16 卡是结构保证;先 2 卡再 6 卡渐进 |
| 8 | **32GB 系统内存**:mmap 加载 + 层分割 host 暂存(DS4 layer-major 预留数 GB)受限;prefix cache/快照槽位预算紧 | 中 | 全 GPU 驻留路线规避;快照槽数调小 |
| 9 | **GDN 维度/门控变体差异**(GSQ-RCO 特有 decay/scaling)**未确认** | 中 | 移植第一步做权重/中间态对照表 |
| 10 | 许可:Apache-2.0,无传染风险;但借代码回 llama.cpp fork(其许可证)时保留署名 | 低 | — |

---

## 11. 建议路线

### 结论:先 Route C(借技术),把 Route A 作为战略性备选,不碰 Route B

1. **立即(Route C,15–30 人日)**:在现有 llama.cpp fork 上摘四样东西——(a) `moe-fused.cu` 的标量 IQ3_XXS 专家点积 + `CachedFfnGraph` 图缓存模式(直击"百次 launch + GPU busy 65%");(b) `delta_net_chunked` 的 chunked verify(MTP 验证提速 2-3×);(c) Volta `fattn-wmma-f16` 的 F16 WMMA 路径思路压缩 BF16 matvec 33%;(d) 若做长上下文,`moe_hybrid_expert_cache` 的 L+1 预取模式。这些是自包含文件/模式,Apache-2.0 可直接搬。
2. **若决定 Route A(75–135 人日)**:严格按 §9 的 6 阶段走,Phase 1 交付前设 kill criteria(ppc64le+V100 冒烟跑 qwen35 官方模型 <10 人日失败即止损)。最大结构性收益:Luce 已把 Qwen3.8 家族(GDN 混合)+ MoE 混合运行时 + 投机解码 + 连续批处理 + OpenAI server 打包成一体,且全部 kernel 在 sm_70 有路径;最大成本:QSA-indexer 和 PLE 两个新子系统 + 6 卡 TP 调试。
3. **Route B(megakernel)明确不做**:单模型硬编码研究 PoC,README 自己声明不可泛化;其"自制原子网格同步替代 cooperative launch"(kernel.cu:139-163)和"占用率钳制防死锁"(kernel.cu:955-978)两段代码是未来若做我们自己的融合 kernel 时唯一值得再读的部分。

### 一句话判决

Luce 是一个**对 Volta 友好、对 Qwen3.8 混合架构家族已有 70% 现成实现**的生产引擎,但我们的三个增量(QSA 稀疏注意力+indexer、PLE 28.8GB 稀疏查表、512×10 MoE + 6 卡 TP 组合)分别对应"要移植数学"、"要新写子系统"、"要自写切分表",全量移植约 3.5–6.5 个月;相比之下,把它当**技术弹药库**(moe-fused kernel、chunked GDN verify、图缓存、专家预取)喂给现有 llama.cpp fork,一个月内就能攻我们 nsys 报告里的两个最大热点。

---

## 附:证据索引(主要文件:行号)

- megakernel sm_70 守卫:optimizations/megakernel/kernel.cu:1-11, 20-30, 185-200;half_type.h:17-29;setup.py:20, 37-43;prefill.cu:705
- megakernel 网格同步/无 PDL/cp.async:kernel.cu:139-163, 955-978, 1013;README.md「Scope and limitations」;README_PASCAL.md 降级表
- server V100 官方支持:server/README.md:622-630, 666, 794;server/CMakeLists.txt:295-324
- 模型抽象:server/docs/ARCHITECTURE.md:53-133(ModelBackend)、149-171(DFlashTarget)、261-443(移植指南);backend_factory.cpp:97-101, 165-326
- GDN:ggml.h:579-589, 2956-3019;gated_delta_net.cu:38-52, 78-105;ssm-conv.cu;delta_net_chunked.cpp:1-19;delta_net_specla.h:1-20;gguf_target_loader.cpp:1-45;internal.h:234-238, 483
- MoE/三层:server/docs/moe_hybrid.md(全文,尤其 158-178 契约、186-259 全热路径);server/docs/DS41.md:83-140(三层+router bias+流式缓存);moe_hybrid_expert_cache.cpp(1,122 行);moe-fused.cu:21, 41(IQ 点积)
- TP:qwen35_tensor_parallel.cpp:68-190(切分表)、222-227(2-16 卡);deps/llama.cpp/ggml/include/ggml-backend.h:392(16 卡上限);ggml-backend-meta.cpp:2,430 行;ggml-cuda.cu:1747(NCCL allreduce)
- Engram(PLE 参照):deepseek4_engram.h:1-23;deepseek4_engram.cpp(274 行);DS41.md:18-20(189GiB 逐行 pread)
- QSA/indexer:ggml.h:610-616;ds4-indexer.cu:960-963;fattn-sparse.cu;deepseek4_internal.h:257-268
- MTP/DSpark:deepseek4_dspark.h:1-27;scripts/convert_ds4_dspark_draft_to_gguf.py
- 量化:deps/llama.cpp/ggml/include/ggml.h:402-443(IQ3_XXS=18, IQ4_NL=20)
- ppc64le 风险点:deepseek4_graph.cpp:53(immintrin include)、4849-4880(x86 守卫的 AVX2/F16C)
