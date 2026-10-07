# Prefill investigation — code facts and unmeasured hypotheses

Current cold single-flow prefill is339.94/341.67/328.98tokens/s at8K/64K/128K. Two/four-flow rates divide shared machine work; per-flow86tokens/s is not total machine throughput. The matrix continues unchanged during investigation.

## Verified code facts

1. UB128 routes128×10 assignments among512experts: average2.5tokens/expert. V100 MMQ policy (`ggml/src/ggml-cuda/mmq.cuh:1489`) already caps MoE J at32 for small loads. UB256/512 raise average to5/10, still within the documented J32 policy. Six-way tensor splitting makes expert projections narrow on each GPU.
2. GDN prefill already uses the fused path: default `fused_gdn_ch=true` (`src/llama-context.cpp:239`), fused selector in `src/models/delta-net-base.cpp`. Optional X4 reduces grid size but not arithmetic. Combined new kernels did not improve prefill and lowered short decode, so both remain off in the selected profile.
3. `ggml/src/ggml-cuda/ggml-cuda.cu:1993` falls back BF16→F32 on Volta at input width>8. Metadata contains BF16 HC/recurrent-gate/PLE projections. Full-attention Q/K/V are quantized; calling the entire attention path BF16 would be inaccurate.
4. Cross-island GPU peer access is unavailable. Six-way reductions use NCCL; LL is forced externally. A UB128 hidden tensor2560×128 is1.25MiB F32, above the fork's BF16-wire threshold. Conversion/two-island synchronization is a candidate cost, not a measured dominant cause.
5. PLE is one layer/sixteen heads, currently gathered on CPU. It adds gather/transfer per ubatch. Earlier GPU-PLE prefill figures overlapped model-merge I/O and cannot establish isolated benefit. GPU-PLE MTP132K failed per-device graph allocation.

## Long multi-flow QSA difference

`src/llama-memory-hybrid-idx.cpp:88-100` creates the pooled indexer block cache only when `n_seq_max==1`. Single long requests can select pooled blocks and use block sparse attention (`src/models/qwen4exp.cpp:1592-1609`). Multi-flow builds block keys/scores through `build_qsa_top_k()` and `build_attn_qsa()`. Context-sized indexer/group/mask work remains without the persistent pooled cache. MTP2×64K measured124.18prefill and17.34/17.49decode per flow;4×64K measured52.23–52.37prefill and5.75–6.41decode. These measurements do not quantify CPU or individual CUDA kernel shares.

Correction to the initial diagnosis: unified KV has `n_stream=1` even with four sequences. Ordinary top_k has shape[width,n_tokens,1,1], so multi decode with at most16tokens can satisfy the existing sparse-gather guard. It is incorrect to say multi decode always falls back to full dense attention. PrefillUB128 exceeds that guard; actual decode eligibility requires graph/guard tracing. Sequence identity is per-query masks and selected cell IDs, not a separate tensor stream axis in unified mode.

`LLAMA_MTP_PAD` only applies when folded draft mode is active, and fold is limited to `n_seq==1`. PAD5 therefore does not turn four-flow target verification into20tokens; do not blame the16-token gather limit on that unsupported assumption.

Removing the single-sequence gate would mix sequence visibility. A future optimization must provide separately keyed block cells/positions, validity, rollback and updates per sequence; it is not a one-line gate change.

## Bounded probes after matrix

Retain CPU PLE/MTP3/PAD5/graphs, exact8K input/512output, ctx132096. Compare UB256 to UB128 reference; require exact counts,512-output parity, temperature bounds and decode retention. Consider UB512 only if256 helps with useful headroom. Test NCCL default versus LL separately if timings implicate collectives.

Global `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16` overrides even explicit F32 precision requests. Do not promote it as a safe fix without an independent numerical audit.

Existing observables: `LLAMA_STEP_TIMING=1` (host stages every200ubatches), `GGML_FT_HOST_TIMING=1` (capture/replay/synchronize), `GGML_FT_GRAPH_CENSUS=1` (operation census). Process-start flags wait for planned profile transition. Host times are not isolated CUDA kernel timings.

Relative bottleneck shares are not yet measured. This records source evidence and the smallest useful follow-up, not projected gains as actual measurements.
