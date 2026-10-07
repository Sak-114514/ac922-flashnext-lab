# PLE, GGUF and 3+3 carry-forward audit — 2026-10-07

## Decision

Carry forward only the measured, model-specific TP3 shape rules, pipeline-stage weight prefiltering/staged loading, and a host-memory budget with a disk fallback. Keep these opt-in until a full 512-token output matches an independent oracle on the target runtime.

Do not carry the old 60 MiB row LRU as the PLE design, apply its IQ4_NL inverses to a new canonical GGUF, or treat the 64-key × 8-token GGML tile as a vLLM patch. The materialized local sync snapshot already has (a) CPU GGUF row deduplication and (b) a device / pinned-host / disk PLE placement path. Its special pinned decoder is qualified only for SM70 FP16, TP4, PP1, V2 full-graph mode and at most four sequences. That does not admit the existing six-card PP2×TP3 arrangement.

## What was actually validated on the six V100s

The private 1Cat experiment used two NVLink islands: GPU 0–2 handled layers 0–24 as TP3, GPU 3–5 layers 25–47 as TP3; PLE was an IQ4_NL file-backed CPU mapping. It ran a real 8,192-token prompt and 512-token output with `pipeline_parallel_size=2`, `tensor_parallel_size=3`, `max_model_len=9216`, no MTP/DFlash, and a compatibility GGUF that re-encoded 38 Q2 tensors to Q4. The encoding-only compatibility report says its sampled tensors dequantize bit-exactly and the source file is unchanged; it does not prove model-level layout or inference correctness.

| Run | Output relative to saved llama.cpp IDs | Decode rate | Finding |
| --- | --- | ---: | --- |
| `g`, eager | first mismatch at token 0: 15406 vs 198 | 4.89 tok/s | Full request, incorrect |
| `h`, compiled graph | first mismatch at token 0: 85 vs 198 | 27.18 tok/s | Graph path changed output relative to `g` |
| `i`, no-compile decode graph | all 512 IDs equal to `g`, still wrong at token 0 vs oracle | 26.70 tok/s | Graph replay was not the remaining `g` drift |
| `j`, partial inverse | first mismatch at token 0; 20 unique output IDs | 24.65 tok/s | Incorrect, reverted |
| `k`, norm/A inverse only | token 0 matches; token 1 differs, then output collapses | 30.94 tok/s | Incorrect, reverted |
| `l`, HF-byte-range-verified transforms | first 5 IDs match; mismatch at index 5 is 760 vs 1596 | 29.19 tok/s | Best old vLLM PP2×TP3 result, parity false |

The `l` run peaked at 14,497/14,437/14,436/13,935/13,935/13,934 MiB on the six 16 GiB V100s. Host `MemAvailable` bottomed at 11.70 GiB, swap peaked at 3.65 GiB and temperature at 50°C. Thus this was a near-capacity single-request run; it did not validate 64K or 262K input, concurrent requests, or MTP. Its configured `max_model_len`/reported KV allocation must not be described as an actual 144K input.

Some subcomponents passed narrower checks: TP3 QSA/GDN/MoE fixtures, GGUF QSA/GDN/MoE payload checks, PLE row-ID equality over 131,072 IDs, and a chunked PLE short-conv fixture. The saved state explicitly leaves live PLE state and QSA selection unproven. A separate 8K/64K report from the llama.cpp sync tree has its own 512-ID regression results; those are not vLLM PP2×TP3 results or an independent HF-oracle parity pass.

## Low-level GGUF formats are not model support

The old fork’s [`vllm/model_executor/layers/quantization/gguf.py`](~/Desktop/ssh/opensource/super-ac922-clone/vllm/model_executor/layers/quantization/gguf.py) lists IQ4_NL in `DEQUANT_TYPES` and `MMVQ_QUANT_TYPES`, but not `MMQ_QUANT_TYPES`. That only says certain linear/GEMV operations can handle the format. It does not implement Qwen4Exp’s tensor names, GGUF hash metadata, PLE row lookup, QSA/GDN layout transforms, tokenizer, MTP block, or PP placement.

The public vLLM Qwen3.8 model PR was merged with reported BF16/FP8/NVFP4 validation on GB200/GB300/H200-class hardware; its PLE offload matrix covered BF16/FP8, not NVFP4. The public recipe points to a dedicated Qwen3.8 image and host PLE offload. Neither establishes support for this IQ3_XXS-MTP-Q8 GGUF or for V100/POWER9. The current GGUF plugin’s generation-tested model list includes other Qwen generations, not Qwen3.8. [Qwen3.8 model PR](https://github.com/vllm-project/vllm/pull/53896), [vLLM Qwen3.8 recipe](https://github.com/vllm-project/recipes/blob/main/models/Qwen/Qwen3.8-Flash-Next.yaml), [GGUF plugin](https://github.com/vllm-project/vllm-gguf-plugin).

The public disk-backed PLE proposal and PP issues describe their own restrictions. In the current 1Cat sync source, `check_ple_layers_on_first_pp_rank` specifically rejects PLE layers outside the first pipeline stage, not every PP>1 configuration. The AC922 checkpoint has `ple_layer_ids=[2]` (1-based); it passes the exact upstream guard with a 25+23 PP2 partition. The pinned local-TP4 specialization separately requires PP1. See the leader CPU-ADMISSION-TESTS.json for the executed positive and negative cases.

## What the new local sync snapshot already replaces

The parent-provided sync snapshot’s `vllm/model_executor/model_loader/gguf_adapters/qwen4exp.py` references the canonical `llama.cpp` Qwen4Exp converter revision, preserves integer PLE hash constants, maps the PLE tensor directly, and keeps its packed storage file-backed. Its `tests/models/gguf/test_qwen4exp_adapter.py` covers metadata/hash validation, split indexer Q/K projection assembly, PLE norm/conv handling, and packed PLE table retention. Use this adapter’s converter-qualified rules for new GGUFs; the old `j/k/l` inverse loader belongs to the 2026-10-03 compatibility run and did not close the end-to-end mismatch. Do not stack its inverses blindly on the new adapter.

Mapping the parent’s #962/#967/#988/#996 labels onto the files inspected: #962 GDN restore/layout is in `gguf_adapters/qwen4exp.py` plus inherited `gguf_adapters/qwen35.py`; #967 packed PLE is in `qwen4_exp/nvidia/gguf_embedding.py` and `csrc/ple_disk_rows.cpp`; #988 hash/input preparation is in `qwen4exp.py`, `qwen4_exp/nvidia/sm70_ngram_input.py`, and `tests/models/qwen4_exp/test_sm70_ngram_input.py`; #996 pinned TP4 admission is in `model_executor/kernels/ple/gguf_pinned.py` and `qwen4_exp/nvidia/ple_layer.py`, with policy coverage in `tests/models/qwen4_exp/test_gguf_pinned_policy.py`. These source-level fixes replace the old model-specific integration assumptions, but they do not constitute a full PP2×TP3 output-parity result.

`vllm/transformers_utils/gguf_rows.py::PackedGGUFRowReader.lookup` already computes `unique, inverse = np.unique(ids, return_inverse=True)`, dequantizes only those unique rows, and restores original order. That is the reusable batch-dedup behavior. The custom GGML CPU candidate’s raw IQ4_NL row cache and its ~36 MiB LRU + 24 MiB scratch cap are tied to `GET_ROWS`, 90-byte rows, O_DIRECT and GGML file mappings. The latest cache-on micro had exact GET_ROWS/dequant float parity and an active cache log, but only 416 sampled row requests; its counters were 0 hits, 1 resident, 231 direct, 0 fallbacks. It made no full-model or speed claim. Do not add a second LRU to vLLM. If large prefill batches later show excessive `np.unique` temporary memory, transfer the old explicit scratch-budget/fallback idea around the existing row reader; do not duplicate its dedup implementation.

The new PLE path in `vllm/models/qwen4_exp/common/ple.py`, `vllm/models/qwen4_exp/nvidia/ple_layer.py`, `vllm/model_executor/kernels/ple/gguf_pinned.py` and `csrc/ple_disk_rows.cpp` has a device / pinned-host / disk tier plan and host-budget accounting. The GGUF disk reader gathers only selected pages/rows; its native path sorts/uniques the page-advice list, not the row payload list. The special IQ4_NL pinned decode capability requires all of: SM70 + FP16 output, TP4 local workers, PP1, V2 dual-compile full graphs, `max_num_seqs <= 4`, exactly one type-20 PLE table, `K=160`, and row count divisible by TP. The observed padded GGUF count is 320,001,536 rows: it divides TP4, but has remainder 2 under TP3. Its CPU row reader already deduplicates; its pinned GPU gather is separately calibrated for small decode batches.

The root's CPU admission report adds two AC922-specific blockers: all 15 four-GPU subsets of the six-card machine fail the required 4-GPU ring topology; TP3 is rejected as `requires_local_tp4_workers`. Even a hypothetical TP4 placement fails `insufficient_fair_host_capacity` on the observed host memory. The root-reported evidence is `CPU-ADMISSION-TESTS.json`; that file was not present in the local or `/home/debian/ac922-vllm` trees I could inspect, so these three admission results are marked parent-reported in `evidence.json`.

A read-only AC922 `/proc/meminfo` sample reported `MemTotal=33,057,600 kB` (~31.5 GiB) and `MemAvailable=29,972,672 kB` (~28.6 GiB). This is CPU host memory, not the aggregate HBM of six 16 GiB V100s. The IQ4_NL table is 28,800,138,240 bytes (~26.82 GiB). With the new code’s default host reserve of 25% and four TP ranks, a TP4 pinned share would be about 5.2 GiB per rank, below the table shard requirement of 7,200,034,560 bytes (~6.7 GiB). The pinned path should therefore reject this host snapshot and leave overflow to disk. The budget reader uses global `/proc/meminfo`, not cgroup `memory.max`; container memory limits still need a separate cap.

## Port map

| Candidate | What to carry | What not to assume |
| --- | --- | --- |
| TP3 shape handling | Keep QSA 2-KV-head replication, GDN 16/48 head mapping, expert-width padding, and the checks in the old TP3 adapter as model-specific loader logic. Retest against the canonical adapter and preserve exact zero contribution for padded channels. | Constructor/load success is not 512-token parity; the `l` output still diverges. |
| PP prefilter + staged load | Carry the pattern: choose local layer range before touching payloads, stage the two islands sequentially, keep PLE on the first pipeline stage, and use `vllm/model_executor/model_loader/gguf_loader.py`'s `skip_weight` and `vllm/model_executor/model_loader/direct_io.py`'s `decoder_layer_filter` where applicable. The old overlay’s `filter_name_map_for_pp` and `tp3_gguf_stream.py` are design references. | Keep the requested PP2×TP3 topology. This checkpoint places PLE on stage 0 and passes the current source placement guard; engine/loader/state parity still needs validation. The pinned TP4 specialization is not admitted. |
| Host budget | Prefer the new per-rank host reserve/share + VRAM/disk cascade over a fixed 60 MiB cache as the table-placement policy. | `MemAvailable` is not the container budget; account for cgroups before admission. |
| 60 MiB CPU row cache | Do not port the cache wholesale. Retain only its explicit scratch-cap/fallback idea if full-prefill measurements show the new dedup reader needs it. | Its CPU-only sample does not show full-model benefit; cache-on sample had no hits. |
| 64-key × 8-token indexer tile | Treat it as a kernel design/reference, not a vLLM patch. The tile is merged in [ggml-org/llama.cpp#29901](https://github.com/ggml-org/llama.cpp/pull/29901); that PR stages keys once for four QSA heads and keeps the vector path below an 8-token tile. The local `codex-upstream` work reported 9/9 fixtures, but that is GGML/llama.cpp evidence. | [vllm-project/vllm#29901](https://github.com/vllm-project/vllm/pull/29901) is an unrelated Turing Marlin PR. vLLM’s target is `vllm/models/qwen4_exp/nvidia/indexer_qsa.py` plus `nvidia/ops/qsa.py` and the SM70 tuning module, with different tensor/dispatch contracts. Reimplement only against that contract after profiling and deterministic top-k tests. The public [persistent_topk report](https://github.com/vllm-project/vllm/issues/54521) is a separate sm121 issue, not a diagnosis of the V100 run. |
| Old inverse transforms | Preserve only as test vectors for the old GGUF source. | Do not apply them over the new canonical converter; qualify each transform by converter revision and exact tensor evidence. |

The local sync Qwen4Exp GGUF adapter and PLE tests are source-level/unit fixtures, not a new 3+3 server pass. The new pinned route is TP4/PP1; the old 3+3 PP2×TP3 route remains an isolated, incorrect-output experiment. No new GPU/service run was performed for this audit.

## Evidence files

See [evidence.json](evidence.json) for path/SHA inventory, remote run hashes, source snapshot status, official support links, the read-only P9 memory sample and the distinction between tested facts and proposed carry-forward.
