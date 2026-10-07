# Strata 引擎 AC922 (POWER9 ppc64le + 6×V100) 移植与多卡改造实录

日期：2026-10-03 · 执行：ZCode（Codex 无额度，用户指令"你来实施"）
上游：github.com/Niko1221/Strata @ MIT，Qwen3.8-Flash-Next 专用推理引擎（ggml 底座）
源码克隆：本地 `opensource/strata-engine/`（33MB 源码，无模型文件）；服务器 `/home/debian/strata-engine/`

## 最终结果

**六卡全量驻留跑通**：CUDA0（主控+自适应热层 2732）+ CUDA1..5（各 4915 专家层，7.99GiB/卡）
- 总驻留 24571/24576 专家（>99.98%），CPU 池 0 任务，全 GPU 计算
- decode 7.42 tok/s（T=1 窗口、无 MTP 草稿、冷缓存），prefill 1.19 tok/s（磁盘流式 2217 专家 @2.64GB/s）
- 首个输出：`248046 198 248045 74455 198 248068 271 248069`（贪心、数字 prompt）

### 可用命令（服务器）
```bash
cd /home/debian/strata-engine
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 ./build/strata \
  --pack pack/iq3_xxs \
  --native /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8-strata48.gguf \
  --tokens "248045,9707,11,32,64,128,256,512,1024,2048,4096,8192" \
  --spec 2 --prefill 512 --max-new 8 --pcie-frac 1.0 --mmap-experts \
  --expert-cache 2048 --expert-profile expert-profile.bin --remote-mirror-all \
  --expert-cache-device1 4920 --expert-cache-device2 4920 --expert-cache-device3 4920 \
  --expert-cache-device4 4920 --expert-cache-device5 4920
```

## 移植补丁清单（全部编译期守卫，x86 零影响、可上游）

| # | 文件 | 内容 |
|---|------|------|
| 1 | `src/kernels/cpu/x86_stub.cpp`（新） | 非 x86 的 AVX 内核替身：*_supported()=false→ggml 兜底；Q2_0 规范内核 abort 哨兵；bf16_rows_dot(_multi) 真标量实现（保预取） |
| 2 | `src/kernels/cpu/expert_layout.cpp` | cpuid/xgetbv 加 `__x86_64__` 守卫；非 x86 cpu_avx2_ok()=false、cpu_name() 读 /proc/cpuinfo |
| 3 | `src/kernels/cpu/native_expert.cpp` | avx2/iq4nl_mt/kq 分发加 `cpu_avx2_ok() &&`（static bool，一次性检查） |
| 4 | `CMakeLists.txt` | STRATA_CPU_X86 检测；非 x86 换 stub、跳过 -mavx* 旗标；ggml-cpu 在非 x86 放开 CXX_EXTENSIONS（AltiVec vector 关键字需 GNU 扩展） |
| 5 | pool/graph/verify/session | `_mm_pause/_mm_sfence` 宏化（x86 原指令，非 x86 atomics） |
| 6 | `src/program/generate.cpp` | AVX2 启动门槛→警告（native pack 走 ggml）；**专家层 3→5 卡**（array/CLI/循环/统计全套） |
| 7 | `include/strata/core/expert_source.hpp` | `remote[3]`→`remote[5]` |
| 8 | `src/core/native_dense.cpp` | eligible() 排除 blk.48（集成 MTP 层不在 canonical 表） |
| 9 | `RemoteExperts::open` + generate.cpp | **--remote-mirror-all**：remote 层拿全量专家（含 CUDA0 重叠） |

## 模型侧适配

- `tools/iq_pack.py`：加 `--layers N`（截断集成 MTP 层的专家表）+ dense 侧 blk.48 跳过 + tokenizer 导出非致命（服务器无 pip/regex）
- `tools/block_count_patch.py`（新）：GGUF 元数据 block_count 49→48（复制件 `IQ3_XXS-MTP-Q8-strata48.gguf` 上 1 字节补丁，原文件未动）
- pack 产物：`pack/iq3_xxs/`（dense.bin 1.5GB + index.txt 1079 张量/302 原生 + native_experts.txt 48 层）
- PLE 表 28.8GB 留 GGUF 由引擎按需直读（每 token 仅 16 行 ×90B）✓ 无需显存

## 调试战记（关键转折）

1. **40GB arena OOM**：引擎默认匿名 arena（假定 64GB RAM 机器）→ 31GB RAM 填充即死。解：`--mmap-experts`（GGUF 文件映射、页缓存可回收）
2. **block_count=49 拒绝**：集成 MTP 层导致元数据 49 层 vs 引擎 48 → 元数据补丁（复制件）
3. **单 token prompt 0 输出**：native pack 下 spec_pos=0 跳过解码循环 → 多 token prompt
4. **spec 需要 residency 表**：自造全量 profile（STRP 格式，want==n_ranked）
5. **act_quant_q8_1_avx2 哨兵触发**（两轮）：
   - 第一层：remote 分配按"当时 CUDA0 持有"跳过 → CUDA0 自适应层内容变化后出现**无人持有的洞**（layer 21 expert 10 实证）→ CPU 池 → POWER 上 abort
   - 解：--remote-mirror-all（by_device 填充 + RemoteExperts::open 两道过滤都放行）
6. **ggml vec.cpp 编译错误**：严格 c++17 禁 AltiVec `vector` 关键字 → ggml-cpu 目标单独 gnu++17

## 性能优化战记（pool 空等 68ms 之谜）

首跑 7.42 tok/s（135ms/token），其中 "pool 空等 68ms/round" 占一半。逐步拆解：

| 阶段 | 改动 | tok/s | 每 token |
|------|------|-------|---------|
| 基线 | - | 7.42 | 135ms |
| +1 | `--adapt-swaps 0`（全驻留下 CUDA0 热层自适应纯浪费：排序 48ms + 磁盘读 30ms）+ `STRATA_LOOKAHEAD=0`（预取线程抢核） | 9.64 | 104ms |
| +2 | **根因修复：sm_70 关零拷贝** | **25.87** | **38.6ms** |

**根因**（计量学：begin 5.1ms / wait 67.6ms → wait 240 次 × 282µs = GPU 流水线真延迟）：
`RemoteExperts` 默认零拷贝——kernel 直读映射主机内存（`z_x_`）。这在 sm_80+ 是快路径，但 **V100（sm_70）上 kernel 读 host-mapped 内存每次 ~280µs**，5 卡 × 48 层 = 240 次/轮全串行暴露。改 `STRATA_REMOTE_ZEROCOPY=0` 走 DMA 异步拷贝后 wait 崩到 2.3ms——拷贝在 copy engine 上与计算流水，**9.64 → 25.52 tok/s（2.65×）**。
已固化为引擎默认：`remote_experts.cpp` 检测 `prop.major < 8` 自动选拷贝模式（env 仍可覆盖），x86/sm_80+ 行为不变。
途中排除的嫌疑：阻塞式 `cudaStreamSynchronize` 睡眠（改自旋无效，非睡眠问题）。

**确定性验证**：零拷贝与拷贝模式输出 token 序列逐位一致。

**第三步：热画像 + CUDA0 大 cache（+27%，25.87 → 32.75 tok/s）**
`--dump-routing` 拿真实路由 → `make_profile.py --no-base` 生成热排名 → `--expert-cache 5000`（自动扩到 6594 槽 10.74GiB）。
热路由几乎全部变成**图内命中**（不走主机往返）：pool 10.9 → **0.75ms**，remote begin 0.02ms（五张专家卡只兜冷门）。
代价：rings 18.8 → 21.6ms（图内多算专家行，但在 GPU 上便宜）。

**当前 30.6ms/token 结构**：rings 21.6（单 V100 串行稠密层计算，70%）+ host 5.5 + pool 0.75。
参照系：RTX 3090×2 层分片 MTP-on = 96.7 tok/s（接受率 0.69 → 无 MTP 折算 ~35-40）——我们 32.75 已进同量级。
剩余差距构成：单卡 V100 稠密层计算（3090×2 分片摊半）+ 无 MTP。
## MTP 转换器（blk.48 → Strata drafter）— 成功

**链路**：`tools/mtp_from_integrated.py`（新，从集成 GGUF 抽 blk.48 重排为 checkpoint 布局）
→ `mtp_pack.py --experts q2_0`（**mtp_rt 只读 Q2_0 专家**，断言 720/180 块字节）
→ `mtp_rt.py` → `mtp-rt/`（dense.txt/dense.bin/experts.bin，799MiB VRAM）

**三个关键语义**（转换战记）：
1. `eh_proj` 是 [2560 出, **5120 入**]——fc_embedding/fc_hidden 沿**输入维**各 2560 列（e 前 h 后）
2. `shared_expert_gate` 在 checkpoint 是 **2-D [1,2560] → BF16**，不是 1-D norm（mtp.cpp 用 bf16() 取它；1-D 会被 mtp_rt 写成 F32+1 → nullptr 崩溃，错误签名 w=0 n_out=1）
3. 11 个 1-D F32 norm 在 GGUF 里**预加过 1**（llama.cpp 惯例，与 1Cat-vLLM 线的 "norm -1 还原" 互证）——转换时减回原值，mtp_rt 的 Gemma +1 再加回

**结果**（spec=4）：接受率 **0.376**，2.09 tokens/round，**40.48 tok/s**（vs 无 MTP 32.75 = 1.24×）；
**64 token 输出与非 MTP 版逐位一致**——验证路径正确性无损 ✓
接受率低于上游 0.69 的主嫌疑：我们的 MTP 源是 Q8_0 量化版再压 Q2_0（双重量化）；flip-eh/flip-qk 变体用于排除切分顺序歧义。

## flip 判决（映射正确性闭环）

`--flip-eh` 变体（fc_hidden 列在前）：**接受率 0.000**（0/93），decode 跌回 19.61 tok/s。
翻转即归零 + 默认 0.376 ⇒ eh 切分（e 前 h 后）与 qk 顺序全部正确；
0.376 vs 上游 0.69 的差距 = **量化 lineage**（我们的源：Q8_0 → F32 → BF16 → Q2_0 四级 vs 上游 BF16 → Q2_0 单级）。
BF16 源已从 HF 拉取中（tools/mtp_fetch.py，HTTP range 只抓 31 个 mtp.* 张量），拉完走同一链路重打包，预期接受率 ~0.6+、60+ tok/s。

## 累计战果（同日）

| 阶段 | tok/s | 关键改动 |
|---|---|---|
| 六卡首跑 | 7.42 | ppc64le 移植 + 5 专家卡全驻留 |
| 优化 1 | 9.64 | adapt/lookahead 关闭 |
| 优化 2 | 25.87 | sm_70 零拷贝修复（arch 感知默认）|
| 优化 3 | 32.75 | 热画像 + CUDA0 大 cache（图内命中）|
| MTP | **40.48** | blk.48 转换器 + 验证（输出逐位一致）|

## 层分片 serve（双卡稠密计算）— 用户提案实测兑现

**拓扑**：`--serve --layer-split 24 --split-device 1` → CUDA0 跑层 0-23，CUDA1 跑 24-47+head+自带 5489 槽 cache，
CUDA2-5 专家卡 ×4772（remote_dev 自动避开 stage 卡），NVLink 组内每 token 仅传 ~10KB 激活。
**协议**：stdin 行协议（`GEN <max_new> <ids>` → 逐 token `T <id>` → `DONE` 统计行）；FIFO 测试台 /tmp/serve.in。

**结果**（数字流 prompt 预热后）：
| 请求 | tok/s | 接受率 |
|---|---|---|
| 冷启动 | 35.7 | 0.33 |
| 预热 1 | 54.6 | 0.578 |
| 预热 2 | 67.2 | 0.667 |
| **64 token 全长** | **66.1** | **0.662**（达上游 0.69 水平）|

vs 非分片 generate 40.48 = **+63%**；serve 模式 spec 自动升到 6（mtp_max 4 + lookup 3）。
注意：serve 默认 stop_eos（数字 prompt 常早停）；serve 有每请求统计（CUDA%d host staging/wait ms）。

## 下一步
- BF16 MTP 源重打包（下载断流已续传，自动链挂 /tmp/mtp_bf16_chain2.log）→ 接受率 0.66 → 0.69+
- prefill DMA 路径（当前 ~1.4 tok/s，冷启动 TTFT 8-11s 的主因）
- 真实文本 tokenizer（补 regex 模块或手写 BPE 导出）→ 完整 chat 服务

## 与上游基准对照

RTX 5070 12GB（IQ3_XXS）= 62 tok/s decode（有 MTP 草稿+AVX2 CPU 池分担）。
我们 25.87 tok/s 的差距构成：无 MTP（T=1 窗口零接受——"drafts accepted 0 of 15"）、rings 18.8ms 的 CUDA0 串行层计算。
下一步提速路径：MTP 转换器（blk.48→mtp GGUF，映射表已备好）→ spec-2+ 草稿接受。

## 资产位置

- 服务器：`/home/debian/strata-engine/`（源码+build+pack+profile）、`models/.../IQ3_XXS-MTP-Q8-strata48.gguf`（补丁副本 78.6GB）
- 本地：`opensource/strata-engine/`（与服务器同步的补丁源码）
- Mac 磁盘占用：仅 ~33MB 源码，模型/构建产物全在服务器
