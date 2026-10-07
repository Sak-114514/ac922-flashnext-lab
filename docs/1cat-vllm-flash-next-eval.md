# 用 1Cat-vLLM 跑 Qwen3.8-Flash-Next GSQ-RCO：可行性评估（含缺口清单）

日期：2026-10-02 深夜 · 依据：本地源码逐项核实（`~/Desktop/ssh/opensource/super-ac922-clone`，即 1Cat-vLLM 完整克隆）+ 今晚 llama.cpp 线 R18-R21 实测

---

## 0. 一句话结论

**可行，且比直觉近得多**：架构骨架（GDN 混合层、MoE、MTP、DFlash2、GGUF i-quant 加载、SM70/V100 特化内核）在 1Cat-vLLM 里全部现成；**唯一的大缺口是 PLE 稀疏查找表**（16 头 × ~2000 万行/头，~27GB，GSQ-RCO 私有结构）。增量工程预估 **数天到一周**（PLE 加载器 + gather 前向 + 验证），不是重写整个模型。

---

## 1. 为什么值得做（两条线的真实差距）

| 维度 | llama.cpp fork（今晚主线） | 1Cat-vLLM | 差距来源 |
|---|---|---|---|
| 单流 decode（真实文本） | 83 tok/s @63% 接受率（MTP 头） | 预期 >100 tok/s @82% 接受率（DFlash2 独立草稿） | **草稿质量是结构性差距，最大单点** |
| prefill（真实 8K） | 966 tok/s（PLE 六卡分片后） | 未知（PLE 未实现，读盘问题会原样存在） | llama.cpp 线已有分片方案 |
| 并发吞吐 | 未实现（单序列 FIFO） | PagedAttention + continuous batching，已验证四并发零错误 | vLLM 核心优势 |
| KV 效率 | 连续池，有碎片 | 页共享（前缀缓存免费获得） | 32GB 卡上差距更大 |
| 数值确定性 | 今晚做了确定性 top-k | 无（GDN batch-invariant 上游都是 open issue） | llama.cpp 线领先 |

**注意**：两条线跑的模型不同（27B AWQ vs Flash-Next GGUF），110 vs 83 不能直接横比；83 是 167B/6B 激活模型的成绩，效率其实更高。

## 2. 1Cat-vLLM 就绪度清单（本地源码核实）

✅ **现成的**：
- `vllm/model_executor/models/qwen3_5.py`（1327 行）：GDN 线性注意力完整实现，含 **SM70 专用 Triton split kernel**（`_sm70_qwen35_gdn_split_kernel`）
- `qwen3_next.py` / `qwen3_next_mtp.py` / `qwen3_5_mtp.py`：MTP 投机解码全套
- `qwen3_dflash2.py` + `vllm/v1/worker/gpu/spec_decode/dflash2/`：DFlash2 独立草稿栈（82% 接受率的来源）
- `vllm/model_executor/layers/quantization/gguf.py`：**GGUF loader 明确支持 IQ3_XXS / IQ3_S / IQ4_XS / IQ4_NL**（MMQ kernel 路径）——GGUF 量化不是障碍
- SM70 特化面：`VLLM_SM70_QUANT_BACKEND`（marlin/turbomind）、`awq_sm70_moe.py`、fused MoE 的 SM70 legacy config、dense f16 fast path
- GitHub CI 已有 "Qwen3.8-Flash-Next on 4x V100-32G" workflow；RELEASE 1.5.0 即 "Qwen3.8 DFlash2 实用化"

❌ **缺口**：
1. **PLE 稀疏查找表**（唯一大项）：
   - 结构：16 头 × ~2000 万行/头（`ple.head_vocab_sizes`），每 token 运行时按索引 gather
   - 需要：GGUF meta 格式的 PLE 张量解析 + 查表前向（gather → 投影）+ 与 Qwen3.5 主干的接线
   - 已知深坑（来自今晚 llama.cpp 线）：PLE 查表反量化本身能做对（393 万值逐位一致），**端到端分岔出现在查表之后的运算**，根因未定位——vLLM 实现会原样面对这个问题
2. **PLE 读盘**：27GB 表 mmap 按需分页是 prefill 冷/热 5 倍差距的根源，与框架无关。起步可先用 CPU 侧 mmap + 并行 gather；进阶再做 GPU 分片（vLLM 的 device mesh 体系里比 llama.cpp 的 meta backend 别扭）

## 3. 建议路径

1. **第一步（半天）**：在 1Cat-vLLM 上以 `--model` 指向 GGUF 试载，确认 loader 对 IQ3_XXS-MTP-Q8 文件的行为（预计卡在未知架构 qwen4exp，需注册架构名）
2. **第二步（2-4 天）**：PLE 加载器 + 查表前向（可先 CPU gather 起步，对照 llama.cpp 线的 fixture 方法论：逐张量 dump、重放 12 次验确定性）
3. **第三步（1-2 天）**：DFlash2 接入 + 与 TP3+DFlash2 生产线相同的验收口径（贪心一致 + 吞吐）
4. **战略价值**：vLLM 的独立 PLE 实现 = 今晚"后续运算分岔"问题的第二个交叉验证参照；两条线互相钉数值

## 4. 今晚 llama.cpp 线成果快照（背景）

- 修 3 个正确性 bug：chain 跨流同步（接受率 26%→100%）、pending_h 跨请求残留、采样非确定性（确定性 top-k）
- 合成 decode 100.04→**113.13**；真实热缓存 48.7→**83.03**（parity 全过）
- PLE 六卡分片：fixture 48 轮逐位一致、132K 配置首次完整加载、真实 prefill 207→**966**（4.7 倍）——但端到端与 CPU 参考在第 63 token 分岔，**GPU PLE 暂缓转正**，当前走"CPU PLE + 多线程读取（R21）+ 紧凑行缓存"路线
- 判负记录：GPU 草稿链真实语料无提速（111 降级为回归提示词成绩）；host 侧"27ms 可移除开销"证伪（真实只有 13%）

## 5. 风险与保留

- 16GB V100 × 6 = 92.8GB 可用（ECC 后），73.2GB 权重 + 132K KV 下每卡余量 <1GB；32GB 卡（CI 用的 4×V100-32G）会宽裕很多——若手里有 32GB 版，优先用那条配置
- DFlash2 的 82% 接受率是在 27B 模型上测的；对 Flash-Next 的接受率需要实测，不能直接外推
- ppc64le 上 SGLang/TensorRT-LLM 等其余框架均不可行（依赖栈无 POWER 支持），不用再评估
