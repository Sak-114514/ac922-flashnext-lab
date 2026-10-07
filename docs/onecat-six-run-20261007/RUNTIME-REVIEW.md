# 六卡 1Cat-vLLM runtime 适配 review

结论：发现 1 个 PP2 启动阻断；本次只读审查未确认其他明确的 TP3 数值或张量布局丢权问题。

## 阻断项

### PP 预过滤把全局 rank 当作 PP 组内 rank

- 位置：[gguf_loader.py](~/Desktop/ssh/opensource/super-ac922-upstream-sync-20261007/vllm/model_executor/model_loader/gguf_loader.py:520)，日志也在该函数的第 529 行打印同一个 rank。
- 证据：这里将 `get_pp_group().rank` 传给 `get_pp_indices`。`GroupCoordinator.rank` 明确定义为全局 rank（`parallel_state.py:485-496`）；同一仓库的 `make_layers` 则传 `get_pp_group().rank_in_group`（`models/utils.py:723-725`）。TP3×PP2 时全局 rank 可为 2–5，而 `get_pp_indices(..., pp_size=2)` 只接受 PP 组内 rank 0 或 1；因此启用 `qwen4exp_pp_stage_prefilter` 后，部分 worker 会在载入前因分区索引越界退出，或按错误阶段筛权重。
- 最小修法：取一次 `pp_rank = get_pp_group().rank_in_group`，将它传给 `get_pp_indices`，并用于日志中的 PP rank。
- 适用条件：`qwen4exp_pp_stage_prefilter=true` 且 PP>1；TP3×PP2 的 25+23 目标满足该条件。

## 其他检查范围

- QSA `[0,0,0,1,1,1]` 映射与每 rank 的两头分组相符；GDN Q/K/V 分段补齐保持 Q、K、V 各段边界；640→768 后每 TP rank 宽 256，适配 Q2_0 的 64 元素块；逻辑词表保持 248320，192 是 embedding storage 的对齐粒度。静态代码路径中未看到这些变换明确丢失源数据。
- MTP HF helper 先按源配置重映射，再对 QSA KV 和 expert/shared-expert 结构补齐；FC 参数走 replicated 分支。没有确认额外启动或形状错误。
- 此结论是源码静态审查；未在本 lane 运行 Python、测试、模型加载或推理。已有 CPU fixture 报告的通过结果不等同于六卡启动或输出 parity 证据。

## Leader resolution

The new PP prefilter now uses `get_pp_group().rank_in_group` for both layer partitioning and its log. The corrected Python file was copied to the private server source. `PP-RANK-REGRESSION.json` verifies all six global ranks resolve to PP-local 0/1 and ranges [0,25)/[25,48); source AST and rejection of the old accessor passed. No GPU/model run is implied by this fix.
