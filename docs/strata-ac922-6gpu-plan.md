# Strata 引擎 × AC922 六卡改造任务书

> 写给 Codex 的施工说明，也是给用户的决策依据。
> 引擎仓库：https://github.com/Niko1221/Strata （MIT，C++/ggml 底座，专为 Qwen3.8-Flash-Next 打造）
> 本地克隆：`~/Desktop/ssh/opensource/strata-engine/`（33MB，完整源码）
> 日期：2026-10-03

---

## 0. 一句话结论

不改架构、不写 TP：把引擎已有的三套多卡机制（静态专家层 RemoteExperts、自适应专家层 PeerExperts、pinned 门铃行）从"三张静态 + 一张自适应、互斥"改成 **1 主控 + 1 PLE 卡 + 4 自适应专家层**，让 IQ3_XXS（73.2GB）整体住进 96GB 显存，CPU 分担行归零，纯 GPU decode。

**不要先跑分再改造（用户明确要求跳过验证阶段），直接开工。** 但 sm_70 编译开关必须第一步就位（它本身就是上游现成选项，不算验证）。

## 1. 引擎现状（读码结论，含文件锚点）

引擎原生多卡机制有三套，目前**互斥**且**规模受限**：

| 机制 | 位置 | 现状 |
|---|---|---|
| RemoteExperts（静态专家层） | `include/strata/core/remote_experts.hpp`、`src/core/remote_experts.cpp` | 最多 **3 张**卡（`--expert-cache-device1..3`，generate.cpp:291 的 `std::array<int,3>`），`claimed[]` 协调分配（remote_experts.cpp `open()`），stripe / 按层两种摆放；**不自适应**（靠 profiler 画像 `src/plan/plan_main.cpp` 排序填充） |
| PeerExperts（自适应专家层） | `include/strata/core/peer_experts.hpp`、`src/core/peer_experts.cpp` | 仅 **1 张**卡（`--peer-device`），`adapt()/apply_pending()`（peer_experts.cpp:269-323）按使用热度换入换出（gain>1.5 才换）；direct 模式（`STRATA_PEER_DIRECT`，默认开）结果经 `scatter_rows_f32` **零拷贝写进验证图的 pinned host 行** |
| 互斥关系 | generate.cpp:1718 | peer 与 remote / layer_split 不能共存 |

关键事实（对改造极其有利）：

- **结果汇合点天然多生产者**：所有外部行的归宿是同一块 pinned host 行缓冲（CAP×H×f32，CAP=MAXT×10，H=2560，单层才几 MB），各卡写的行不相交（claimed 保证），无锁无竞争。
- **P2P 已就绪**：`PeerExperts::open()` 里 `cudaDeviceCanAccessPeer` 双向检查 + `cudaDeviceEnablePeerAccess`（peer_experts.cpp:97-110）。AC922 六卡 SXM2 全 NVLink mesh，全通。
- **门铃同步**：`Doorbell`（layer.hpp:402）pinned host flag，验证图内零拷贝读行前等 flag。多卡只需"全卡完成后置一次 flag"，**图不用改**。
- **sm_70 有上游开关**：`-DSTRATA_EXPERIMENTAL_SM60=ON`（CMakeLists，注释明写 "Volta sm_70, V100 / Titan V, #236, community-tested"）。issue #236：社区已在 32GB V100 跑通 **swift-1.5-iq3_xxs 和 coder-IQ1_M（同款模型同款量化）**。需要 CUDA 12.x（CUDA 13 砍了 sm_70）——AC922 上的 CUDA 12.4 正好，也是 ppc64le 的最后支持版本。
- **PLE/QSA/GDN/MTP 全是一等公民**：`src/kernels/cuda/ple.cu`、`native_qsa_score.cu`、`native_gdn`、MTP 验证图（`src/core/graph.cpp`、`verify.cpp`）。**没有 1Cat 线那场 GGUF 布局逆转仗要打**。

## 2. 目标拓扑与显存账本

用户原始设想"2-3 张主卡 + 3-4 张专家层"。**修正**：引擎无 TP，主卡 dense/注意力是小头（瓶颈在专家行），TP 化主卡是数周级负收益。正确的"第二张主卡"用途是 **PLE 专用卡**：

```
卡0  主控     dense + KV + MTP + 验证图 + 自适应热专家层(~3GB)     ≈ 8-10GB
卡1  PLE      blk.1.ple_* 六张量 + ple_block kernel + 溢出专家槽   ≈ 10-14GB
卡2-5 专家层  RemoteExperts ×4，每张 ~15.4GB(扣 600MiB reserve)    ≈ 61.6GB
```

IQ3_XXS 账本（73.2GB GGUF 拆解）：dense ≈ 2-3GB，PLE 量化后 ≈ 7-14GB（装载时打印实际值定卡1去留），MTP ≈ 1-2GB，experts.bin ≈ **55-60GB**。容量 61.6+3+4 ≈ 68GB > 60GB ✅ **全显存覆盖，约 12% 余量**。

若 PLE 实测 ≤5GB 可缩回卡0 → 退化成 1主+5层（备选 B）。装载时打印账本即可决定，不用预先验证。

## 3. 改造工作项

### A. CLI 与布局（generate.cpp，~100 行）
- `std::array<int,3> expert_cache_remote` → `std::vector<int>`；`--expert-cache-deviceN` 泛化为 `--expert-devices 2,3,4,5`。
- 解除 peer XOR remote 互斥（generate.cpp:1718）：peer 语义改为"卡0 上的第 0 自适应层"，remotes 是外层。
- 新增 `--ple-device N`（默认 0）。

### B. RemoteExperts 自适应化（核心质量项，~150 行）
- 把 `PeerExperts::adapt()/apply_pending()` 移植进 RemoteExperts：usage 统计、refill 流、gain 门槛全是现成的。
- 接上 `claimed[]`：换入的专家必须未被任何卡 claimed；换出时清 claimed（池线程单线程串行调用，天然无锁）。
- 每层每轮：卡0 热层先挑（in-graph 最快），remotes 按 rank 顺序挑。

### C. 多层启动与同步（verify.cpp 池线程路径，~100 行）
- launch：池线程逐层对 N 个 tier 调 `begin()/launch()`（全异步，不阻塞）。
- finish：**串行 spin 每张卡 → 全部完成后置一次门铃 flag**（图零改动）。尾延迟代价 <10%，先不做 per-tier flag 的图内轮询。
- 行所有权：kind 分类从"peer==2 / CPU==-1"扩为 per-tier `res_` 查询，互斥性由 claimed 保证。

### D. PLE 独立卡（~200 行，装载账本超了才做）
- PLE 的哈希+gather 在 host 侧（`ple_stage_token`），设备半在 `ple_block`（layer.cpp:1142 起）。把 `blk.1.ple_*` 放卡1，ple_block 在卡1 跑，结果 P2P 写回卡0 `emb_dev`。

### E. 构建与平台
- `-DSTRATA_EXPERIMENTAL_SM60=ON -DCMAKE_CUDA_ARCHITECTURES=70`，CUDA 12.4 工具链，ppc64le 直接 CMake，**跳过 setup.py 一键安装器**。
- 运行时设备准入已由 `src/core/device.cu` 的 `STRATA_EXPERIMENTAL_SM60=1` 宏放行。

### F. 二期可选
- adapt refill 改卡间 P2P 直拷（现在走 host→device，peer_experts.cpp:304）。
- QSA 打分器写 fp16 mma 版（sm_70 tensor core；现走 fp32-FMA 回退，见风险②）。

## 4. 风险与地雷（按杀伤力排序）

1. **CPU 专家池 kernel 只有 AVX2/AVX512 路径**（`expert_layout.hpp`："AVX2: the floor of every expert kernel"）——POWER9 是 VSX，**CPU 行算不了任何行**。这把"全显存覆盖"从优化项变成**硬前提**：必须关 `--resident-cpu-experts`，专家层容量必须 ≥ experts.bin，否则溢出行无处可算。将来跑 >96GB 的 pack 需写 scalar/VSX 回退（数天，先不做）。
2. **QSA 的 tf32 mma 回退**：V100 无 tf32，QSA 12 层走 fp32-FMA，变慢多少未知。先测再决定写 fp16 mma。
3. **V100 兼容性非零风险**：#236 跑通的是 x86 + 32GB V100；ppc64le host 代码 + 16GB 卡的槽位预算是新的。卡上 reserve 600MiB 可能要上调（CUDA context ~500-600MB）。
4. **HOST 单核弱**：POWER9 单核 < Ryzen 5 7600，池线程的 plan Meta 构建（launch() 里有 O(n²) 去重环，n≈T×10≈80，无碍）和逐层 staging memcpy 是串行热点。若成瓶颈，二期把 launch 分派给 per-tier 线程。
5. **32GB RAM**：pack 源文件 mmap（ExpertSource blob）+ pinned 缓冲，几十 MB 级，无压力；冷启动顺序读 73GB @ NVMe ~1-2GB/s ≈ 40-70s。

## 5. 速度预估（带推导）

锚点数据：
- RTX 5070 12GB + 64GB RAM（**专家在内存里流式**，miss 走 PCIe4 zero-copy + x86 CPU 算行）：IQ3_XXS **62 tok/s** decode / **1750** tok/s prefill；3090 24GB 估 100-140。
- AC922 llama.cpp 线（六卡行分片，全 VRAM，同模型同量化）：合成 decode **113**，真实热缓存 **83**，prefill **966**。1Cat-vLLM 窄候选 **31**。
- 关键差异：本方案全 VRAM 覆盖（5070 是内存流式），NVLink 互联 + 五卡并行算行（5070 单卡 + CPU 分担）；POWER9 池线程单核偏弱（负项）。

**decode（IQ3_XXS，全 VRAM）**：
每步临界路径 ≈ 48 层 × [卡0 注意力 20-40µs ∥ 外部层行往返（launch 10-20µs + 4 卡并行行计算 10-30µs + NVLink scatter + spin）40-70µs] ≈ 2.9-5.3ms/窗口；MTP 验证窗接受率 55-82% → 3-3.5 token/窗 → **预估 90-140 tok/s，中位 ~115**。
对齐：llama.cpp 线真实 83（合成 113 是同卡同量的引擎上限参照）→ **相当到 +40%**；比 5070 的 62 高 50-120%（我们把它的内存流式整个消掉了）。

**prefill**：吞吐型。51 GFLOP/token 专家行 / 5 卡 × 15-20 TFLOPS 有效 MMQ ≈ 0.7-1ms/token → **预估 700-1500，中位 ~1000**，与 llama.cpp 线 966 相当；5070 的 1750 靠新代际 tensor core，我们靠多卡并行大致追平。

**冷启动**：experts.bin mmap 顺序读 + 槽位填充 ≈ **1-2 分钟**（对比 llama.cpp 线历史上 73GB 冷读的 6 分钟 → R21 优化后的 ~2 分钟，持平或略优）。

**判断**：这套改造的真正价值不是单点刷新 decode 纪录，而是：① 全 NVLink 五卡专家层 + 自适应命中，长对话稳态好；② 128K-262K 上下文可用（GDN 混合 KV 极小）；③ 引擎为本模型量身打造，没有 parity 仗要打；④ 与 llama.cpp 线互为备份，量化质量不降级（同 IQ3_XXS）。

## 6. 施工顺序

1. **第 1 步（0.5-1 天）**：A + C（方案1 同步）+ E，1 主 + 4 静态层跑通 IQ3_XXS，拿到 AC922 第一组 tok/s。此步顺带完成 sm_70/ppc64le 的首次编译——它是改造的前提而非"验证"。
2. **第 2 步（1-2 天）**：B 自适应化 + 卡0 热层回归。
3. **第 3 步（按需）**：D PLE 卡（若账本爆）；F P2P refill / fp16-QSA。
4. 全程对照物：llama.cpp 线 83/966、5070 公版 62/1750。

## 7. 边界（明确不做）

- 不做主卡 TP（无通信层，负收益）。
- 不写 VSX CPU kernel（全显存覆盖后无 CPU 行）。
- 不动 1Cat-vLLM / llama.cpp 两条现有战线（本方案是第三条独立战线，实验目录独立）。
- 不先跑分再改造；编译通过即接入改造主线，第 1 步末尾一次性拿数字。
