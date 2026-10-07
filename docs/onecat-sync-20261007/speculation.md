# 1CatAI/1Cat-vLLM PR #927–#936: MTP, CUDA Graph, and batching

## Scope and source snapshot

Read-only review of GitHub PR metadata, changed-file lists and PR bodies, captured 2026-10-06 16:09 UTC (2026-10-07 in Shanghai). Of the ten PRs, seven are merged, #930 and #934 were closed without merge, and #932 remains open/draft. The current `main` snapshot queried was `bc01f289` (merge #1000; its first parent is the #1001 merge). Each PR’s state below comes from that PR’s REST record; code links are pinned to its merge commit or, for unmerged work, its head commit.

PR-body test numbers are author-reported; I did not rerun their tests. The full metadata, head/base/merge SHAs, file list, source permalinks and evidence notes are in [speculation.json](speculation.json).

## PR ledger

### #927 — GDN projection tail copies

**Merged 2026-10-04 19:22:55 UTC** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/927) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/927/files)

Files: `benchmarks/kernels/benchmark_sm70_gdn_projection_tails.py`, `docs/design/sm70_gdn_projection_tails.md`, `tests/kernels/test_sm70_gdn_projection_tails.py`, `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py`.

Fuses copies of z/b/a at the GDN input/core boundary for measured FP16 M5/M20 shapes. It leaves QKV arithmetic and the QKV view intact. The PR reports four V100 checks passing and bitwise payload/layout equality. The 36-layer copy-chain reduction (274.145→48.763 μs at M5; 274.432→72.663 μs at M20) is an operator result, not a model TPS claim.

### #928 — row-strided mixed QKV for MTP recurrence

**Merged 2026-10-04 19:36:05 UTC** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/928) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/928/files)

Files: `benchmarks/kernels/benchmark_sm70_mtp_strided_qkv.py`, `docs/design/sm70_mtp_strided_qkv.md`, `tests/kernels/test_sm70_mtp_gdn_mixed_qkv.py`, `tests/kernels/test_sm70_mtp_strided_qkv.py`, `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`, `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`.

Allows the fused MTP GDN recurrence to read feature-contiguous mixed-QKV using its physical row stride for the measured M5/M20 shapes; other layouts retain a fallback. The PR reports 18 V100 cases passing, six unmeasured fixtures skipped, and bitwise output/conv-cache/FP32-SSM-state equality. Its 36-layer recurrence-chain timings are operator-only.

### #929 — reuse MTP GDN verifier output

**Merged 2026-10-04 19:51:21 UTC** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/929) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/929/files)

Files: `benchmarks/kernels/benchmark_sm70_mtp_gdn_output.py`, `docs/design/sm70_mtp_gdn_output.md`, `tests/kernels/test_sm70_mtp_gdn_direct_output.py`, `tests/kernels/test_sm70_mtp_gdn_mixed_qkv.py`, `vllm/model_executor/layers/fla/ops/fused_sigmoid_gating.py`, `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`.

Adds a validated caller output for pure MTP verification at measured M5/M20 geometry, avoiding a temporary recurrent output and merge/copy. The PR reports bitwise output and recurrent-state checks. The 36-layer output-chain reductions (550.902→470.876 μs at M5; 1081.477→1021.399 μs at M20) are not model round latency.

### #930 — HC weight lookahead screen

**Closed unmerged 2026-10-04 20:25:22 UTC; draft** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/930) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/930/files)

File: `docs/design/sm70_hc_weight_prefetch_screen.md` only. The author reports bitwise equality but a negative M20 timing, so the original schedule was retained and the candidate reverted before production dispatch. There is no merged runtime implementation.

### #931 — canonical GGUF expert loader

**Merged 2026-10-04 20:36:31 UTC** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/931) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/931/files)

Files: `docs/design/gguf_turbomind_model_moe.md`, `tests/kernels/quantization/test_gguf_turbomind_model_moe.py`, `tests/models/gguf/test_small_routing_integration.py`, `vllm/model_executor/layers/quantization/gguf.py`, `vllm/model_executor/layers/quantization/gguf_turbomind_moe.py`, `vllm/model_executor/model_loader/gguf_loader.py`.

Restores canonical TurboMind GGUF expert preparation and quantization-method selection, including retaining Q2_0 packed source through TP slicing. It is model/weight integration, not a speculative controller or a GPU hidden/token feedback change. The PR reports four CPU and four V100 checks passing; it makes no isolated speed claim.

### #932 — long-context attention candidates

**Open/draft; last updated 2026-10-04 20:51:32 UTC** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/932) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/932/files)

Files: `benchmarks/kernels/benchmark_sm70_dflash_sliding_candidate.py`, `benchmarks/kernels/benchmark_sm70_long_context_attention.py`, `benchmarks/kernels/build_sm70_dflash_sliding_candidate.py`, `benchmarks/kernels/build_sm70_long_context_attention.py`, `docs/design/sm70_v100_migration_control.md`.

This is a benchmark/build candidate lane, not a production MTP or graph-dispatch change. The PR body reports CPU page/causal checks and SM70 candidate compilation, with GPU screening pending; it explicitly says no experimental kernel is enabled.

### #933 — measured GGUF expert down-vector dispatch

**Merged 2026-10-04 20:48:53 UTC** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/933) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/933/files)

Files: `docs/design/sm70_gguf_expert_down_dispatch.md`, `tests/kernels/quantization/test_gguf_expert_down_dispatch.py`, `tests/kernels/quantization/test_gguf_expert_down_graph.py`, `vllm/model_executor/layers/quantization/gguf_turbomind_moe.py`.

Connects measured IQ4_NL/Q2_0 down-vector routes to canonical GGUF expert banks while retaining grouped-GEMM fallback elsewhere. It is not MTP sequencing. The PR reports 12 CPU dispatch tests and ten V100 checks passing; the cited 0.342 ms saving is from an earlier operator sweep, not an end-to-end model result.

### #934 — router selection negative screen

**Closed unmerged 2026-10-04 20:58:17 UTC; draft** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/934) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/934/files)

File: `docs/design/sm70_router_selection_screen.md` only. The candidate was reverted before GPU testing because the target path already used FP16 logits and lossless half keys/top-16 selection. It adds no runtime route or performance claim.

### #935 — DFlash2 Q/K preparation and paged attention layout

**Merged 2026-10-04 22:25:52 UTC** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/935) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/935/files)

Files: `benchmarks/kernels/benchmark_sm70_dflash2_qk_rope.py`, `flash-attention-v100/flash_attn_v100/flash_attn_interface.py`, `flash-attention-v100/include/fused_mha.h`, `flash-attention-v100/kernel/fused_mha_api.cpp`, `flash-attention-v100/kernel/fused_mha_forward_paged.cu`, `tests/kernels/core/test_sm70_dflash2_bmhd.py`, `tests/kernels/core/test_sm70_dflash2_qk_rope.py`, `vllm/model_executor/layers/attention/sm70_dflash2_qk_rope.py`, `vllm/model_executor/models/qwen3_dflash.py`, `vllm/model_executor/models/qwen3_dflash2.py`, `vllm/v1/attention/backends/flash_attn_v100.py`.

Adds a shape-gated FP16 Q/K RMSNorm + NeoX RoPE + KV-publication op and direct BMHD attention output for a supported Qwen3.8-27B NVFP4, DFlash2 draft7, TP4 path. The PR reports 21 GPU tests and bitwise full-layer replay. It also reports C1/C4 timing, with overlapping acceptance intervals. **This is 27B NVFP4+DFlash2 evidence, not Flash-Next evidence; neither acceptance nor throughput should be extrapolated to Flash-Next.**

### #936 — default-forward GDN tail-copy route

**Merged 2026-10-04 21:28:54 UTC** · [PR](https://github.com/1CatAI/1Cat-vLLM/pull/936) · [files](https://github.com/1CatAI/1Cat-vLLM/pull/936/files)

Files: `docs/design/sm70_gdn_projection_tails.md`, `tests/kernels/test_sm70_gdn_default_projection_tails.py`, `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`.

Extends the GDN z/b/a projection-tail copy from the optional input/core boundary to the default MTP full-forward path for measured M5/M20 shapes. The PR reports two GPU checks with bitwise Z/QKV/b/a and QKV alias equality. The 36-call graph timing excludes GEMM and recurrence; it is not end-to-end.

## What the PR set does and does not add

The MTP-related kernel work is tightly scoped to GDN data motion and layout: fuse tail copies (#927, #936), read row-strided QKV (#928), and reuse verifier output (#929). Those changes improve existing model operations. None modifies the speculative scheduler, acceptance/rejection logic, proposer state machine, or general CUDA Graph dispatcher.

The wider `main` snapshot already has a multi-request GPU draft chain before these PRs. The Qwen4Exp MTP proposer subclasses `EagleProposer`; the V1 Eagle speculator keeps per-request `draft_tokens` and hidden-state buffers, samples on device, and uses a Triton update kernel to put the sampled token and that request’s hidden row into next-step inputs. It captures model forward + sample + input update for one draft step and replays that graph for subsequent steps. The PRs reviewed here do not introduce that base behavior. [Qwen4Exp proposer](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/v1/spec_decode/qwen4_exp.py), [Eagle speculator](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/v1/worker/gpu/spec_decode/eagle/speculator.py).

CUDA Graph replay and padding are also broader, pre-existing V1 behavior. The Eagle speculator captures the one-step draft routine; the dispatcher pads token/request descriptors to configured graph capture sizes. That differs from `LLAMA_SPEC_PAD_MULTI`, which appends explicit dummy `llama_batch` rows above real positions and waits until accept/rollback before trimming their KV cells. #927–#936 do not add this server-side verify-pad mechanism. [CUDAGraph dispatcher](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/v1/cudagraph_dispatcher.py).

There is no adaptive native Qwen MTP depth feature in #927–#936. The GDN fast paths use exact M5/M20 geometries, and #935 gates DFlash2 draft7. Current `main` has a separate optional DFlash2 lookup controller that chooses a short `draft_block` or full configured width, and Suffix Decoding has a separate dynamic token count. Neither is native Qwen MTP depth adaptation. [DFlash2 lookup controller](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py), [suffix decoding config](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/config/speculative.py).

## Parent-requested context: Native Flash-Next MTP4 and LUT4

These two merged PRs are outside the formal #927–#936 ledger but are the most useful priority anchors for Flash-Next.

- [#821 — qualified full-vocabulary Flash-Next MTP4 defaults](https://github.com/1CatAI/1Cat-vLLM/pull/821), merged 2026-10-05 11:59:25 UTC. It admits full-vocabulary native MTP4 with draft-only QPN8 head storage, FP16 compute, TP4 local top1/compact value-ID transport and guarded fallbacks. The reported C1 complete-round mean is 21.198709 ms vs 23.215921 ms frozen main, with 50.2035% eight-prompt acceptance and zero target/draft teacher-forcing KL. Its measured hardware and model path are TP4 V100-SXM2-32GB. C4 8K prefill fails a 160 MiB allocation; no C4 throughput claim is available. The [default qualification doc](https://github.com/1CatAI/1Cat-vLLM/blob/7413fad39a4c76a64f4ddfeabae31a65b37357b6/docs/design/sm70_flash_next_mtp4_15ms.md) explicitly preserves the C4 gap.
- [#1000 — canonical LUT4 expert route](https://github.com/1CatAI/1Cat-vLLM/pull/1000), merged 2026-10-05 22:00:47 UTC. This is directly relevant to the Flash-Next final expert layer’s IQ4_XS gate/up and IQ4_NL down. It enables the measured four-lane Q8_1/DP4A route only for TP4 M5/M20; all other shapes keep canonical execution. The PR reports C1 pooled mean 18.996→18.534 ms and C4 mean 44.754→43.853 ms, with quality gates recorded. Its four-lane route is retained; eight- and sixteen-lane variants remain operator screens, and sixteen lanes regressed. See the [qualification record](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/docs/design/gguf_lut4_q8_experts.md).

The PR #821 head tests are useful for defining the local-head gate: they cover 30 finite/tie/NaN/padding cases and test value/ID packet equivalence at M1/M4/M5, but the fresh real-checkpoint head check is TP0 only because the fixture contains one checkpoint shard. This is not a six-GPU 3+3 PP head qualification.

The same #821 qualification doc keeps several research probes out of the default: for example, grouped QPN8 draft experts had maximum KL 0.091476, and row-scaled INT8 draft experts had maximum KL 0.062439 against a 0.05 maximum gate. Those results are examples of rejected quality routes, not kernels to reproduce as a default. #1000 similarly keeps the four-lane LUT4 model route while eight/sixteen-lane results remain operator screens and sixteen lanes regress.

The AC922 source record helps rank the next work. The existing island path has GPU0–2 as TP3 for the first stage and GPU3–5 as TP3 for later layers/output/MTP; its 1×8K C and C+H paths record exact 512-token outputs. The tested `LLAMA_DRAFT_DETP=1` single-GPU draft-head path is separate: local notes make de-TP and `LLAMA_ISLAND_SPLIT` mutually exclusive, and the R19 handover says DETP disables island mode. The separate `isolated-draft` config sets both, while its log shows DETP active on CUDA0 with trunk weight 1.0; by that guard it qualifies the de-TP branch, not a concurrent 3+3 local head. R19’s status file says keeping the target mapped 3+3 while moving MTP to one GPU had not yet been implemented and still needed placement/headroom preflight. A TP3 native-MTP trial did verify seven BF16 tensor transforms, but it used Qwen3.8-27B AWQ and is not a Flash-Next 3+3 PP test. See local `GPT-HANDOVER.md`, `patches/draft-detp-NOTES.md`, `final-results/ROUND19/STATUS.md`, and `opensource/super-ac922-clone/ac922/tp3-mtp4-trial/results/MTP-TP3-1K-64K.md`.

### Seven-step adaptation priority

1. **P0 — qualify head ownership/top1 on the actual 3+3 island topology.** Verify which second-island ranks own the MTP vocab shards and how local value/ID selection reaches the proposer. Do not enable `LLAMA_DRAFT_DETP` together with the 3+3 split: the current source says those modes are mutually exclusive.
2. **P0 — reproduce the #821 native MTP4 quality contract on the local Flash-Next build.** Treat the papered TP4 C1 result as a reference; preserve exact IDs/teacher forcing and keep C4 unqualified until it passes its own headroom gate.
3. **P1 — evaluate #1000 on the matching final IQ4 layer only.** Map the TP3 local M5/M20 shape and memory budget first, then compare exact output/quality and complete-round time; the observed route affects one expert layer, not every layer.
4. **P1 — test #927/#936 copy-tail fusion if local z/b/a/QKV geometry matches.** It is copy-only and leaves arithmetic intact, but retain it only if a matched full round improves after the exact local layout gate.
5. **P2 — test #928/#929 only where actual row stride and output ownership match.** Their operator-chain wins do not establish target/draft model improvement.
6. **Reference, do not duplicate — use the upstream per-request GPU token/hidden chain and FULL per-step graphs as the multi-slot design reference.** The custom R19 chain and FIFO are single-sequence. Prioritize request-local state/reset semantics and matched parity over another single-sequence fast path.
7. **Defer/no-port — #935 DFlash2 attention prep is a different 27B drafter; #932 is GPU-pending; #930/#934 are negative screens.** Do not copy rejected shortlist/quantization experiments. On #1000, four lanes are model-qualified; eight lanes are operator-only and sixteen regressed.

## Comparison with the AC922 llama.cpp work

| Area | 1Cat-vLLM baseline / PR #927–#936 | AC922 custom path and evidence | Gap or distinction |
|---|---|---|---|
| Multi-request MTP state | Existing V1 Eagle/Qwen MTP holds per-request token and hidden feedback on device and replays a per-step CUDA Graph; #927–#936 optimize GDN kernel data movement. | R19 `LLAMA_MTP_GPU_CARRY`, `LLAMA_MTP_GPU_CHAIN`, GPU embedding and MTP FIFO source guards are single-sequence. | 1Cat’s general multi-request proposer is broader than the original llama.cpp chain. The private llama.cpp 2/4-row hidden-carry candidate was activated in a later integration run, but parity is not established. |
| Hidden feedback semantics | Qwen4Exp proposer returns a feedback hidden tensor and the Eagle step update reuses each request’s own row. GDN SSM state is separately per-request and accepted-state selected. | `mtp_gpu_carry_input` stores the Qwen4Exp `h_nextn` row and feeds it back; partial layouts flush to that request’s CPU row. | GDN SSM recurrence state in #928/#929 is not the same object as the llama.cpp `h_nextn` GPU-carry buffer. |
| Token chain | Existing Eagle/MTP device buffers and update kernel already feed sampled token IDs into the next draft step; this is not added by these PRs. | The custom greedy token chain keeps token/hidden feedback on GPU until chain completion, but its main fast path is single-sequence and excludes adaptive MTP. | There is no base token-chain capability gap to attribute to #927–#936; the concrete custom difference is GGML backend integration and guard behavior. |
| FIFO | Uses V1 graph-manager/stream execution and graph replay; no 1Cat `LLAMA_MTP_FIFO`-equivalent switch appears in these PR diffs. | `LLAMA_MTP_FIFO` calls the GGML scheduler FIFO path only for MTP, one sequence, embedding replica enabled, no pipeline parallelism. | Similar scheduling goal, different runtime primitive. Compare graph replay and host synchronization before treating as a missing upstream capability. |
| Verify padding / graphs | Existing dispatcher pads batch descriptors to captured graph sizes; Eagle supports FULL decode graphs. #935 adds a route-specific fused DFlash2 attention path. | `LLAMA_SPEC_PAD_MULTI=24` pads pure decode/verify server batches with dummy rows and trims after acceptance. Local DEBUG5 logs show 13 pad markers in control and 16 in carry run; neither run had retry/cleanup warnings. | Both stabilize graph shapes, but the padding and state-cleanup contracts differ. The PR set does not add the custom dummy-row policy. |
| Dynamic draft width | Fixed native MTP step count in the Eagle proposer; optional adaptive DFlash2 lookup width and dynamic Suffix Decoding are separate methods. | `mtp_adaptive` uses per-sequence EMA thresholds for depth 2/3, but GPU token-chain fast path and multi-sequence carry candidate exclude adaptive mode. | Custom dynamic native MTP exists, but is not combined with the GPU chain/carry paths; DFlash2 width adaptation is not a like-for-like native MTP replacement. |

### Local integration evidence boundary

The private `phase2-combined-carry` artifact reports completion and real 4-row activation. The matched `phase2-combined-control` and carry profiles load the same runtime-combined-v1 library hashes; the carry opt-in differs. All 12 matched output lists differ and the candidate is not faster. A separate frozen V1 short control uses a different binary and confirms that repeated real-text multistream outputs already differ there. That does not isolate the mismatch to the new carry. These are activation/performance observations, not a parity proof or a transferable optimization.

The earlier R19 chain-order log has 408/408 sampled token checks and zero reported hidden max error, but its status file labels that chain diagnostic/local and says real-text parity was still under test. That is internal chain evidence, not Flash-Next acceptance evidence.

## Reading the speed claims safely

The copy/stride/output-buffer numbers in #927–#929 and #936 are operator or graph-chain timings. #935 includes a full-model Qwen3.8-27B NVFP4+DFlash2 draft7 measurement, but still measures a different model, quantization and drafter from Qwen3.8-Flash-Next. #821 and #1000 do include Flash-Next, but on TP4/32GB V100 shapes; #821 explicitly has no C4 result. None is a result for our six-16GB V100 3+3 topology until rerun there. Do not generalize the 27B acceptance or throughput numbers to Flash-Next.
