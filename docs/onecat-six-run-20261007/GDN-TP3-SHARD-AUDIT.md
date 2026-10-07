# TP3 GDN QKV and conv shard audit

## Result

The real TP3 shard helpers preserve the intended GDN channel order when the global tensor is padded inside each Q/K/V segment before loading:

- Source GDN dims are K16/V48 with head dim128, so source Q/K/V row counts are `2048/2048/6144`.
- Candidate TP3 dims K18/V54 make target Q/K/V row counts `2304/2304/6912` and per-rank rows `768/768/2304`.
- Required structural padding is Q `+256` rows, K `+256`, V `+768`, inserted after each original segment. Local rank order remains `[Q_rank, K_rank, V_rank]`.

A single tail pad to reach total 11520 rows is wrong. Both actual helpers compare unequal against the expected per-segment rank layout for all 3 ranks.

## Source proof

The read-only AC922 `runtime-357d` source used by the CPU fixture reports these source hashes:

- `vllm/model_executor/layers/linear.py`: `2f4976f238dcc8bc39ce29243deeb3bb1d035312ffd49db2d75cb2aeedfb150f`
- `vllm/model_executor/layers/mamba/mamba_mixer2.py`: `de92464601f291447fa4a4717464a5c8ad7a62f07e254a08c7950e5b0b29d20e`
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`: `3e241bcf133ba0274ff8549d1840e7f6454aa2251f08241a4060c582d83f468a`
- `vllm/models/qwen4_exp/nvidia/model.py`: `ee27e5d9668cc3c350e94b52f0f8f45889b7d53920926caf33633f573659fa5c`

On that runtime, Qwen4Exp constructs GDN with `gqa_interleaved_layout=False`; the GDN class computes `conv_dim=2*key_dim+value_dim`, creates Q/K/V/Z output sizes independently when non-interleaved, and installs the conv loader with shard specs `(key_dim,0,false)`, `(key_dim,0,false)`, `(value_dim,0,false)`. The Qwen3.5/Qwen4Exp checkpoint mapper sends `in_proj_qkv` into `in_proj_qkvz` using shard tuple `(0,1,2)` and `in_proj_z` as shard3.

`MergedColumnParallelLinear.weight_loader` processes the QKV tuple by splitting the loaded tensor at its configured output-size boundaries. Each projection is then narrowed by its own TP shard offset. `mamba_v2_sharded_weight_loader` likewise walks the three full-dimension shard specs, advancing separate loaded and destination boundaries. Neither helper should divide the entire 11520-channel tensor as one contiguous TP chunk.

## CPU evidence

The checked-in fixture is [gdn_tp3_shard_cpu.py](<~/Desktop/ssh/reports/onecat-six-run-20261007/fixtures/gdn_tp3_shard_cpu.py>). It constructs tiny CPU row-ID tensors, invokes the actual installed `MergedColumnParallelLinear.weight_loader` non-quantized path and the actual `mamba_v2_sharded_weight_loader`, and compares each rank against the expected segment slices. It also has an optional `AC922_TP3_LOCAL_WEIGHT_LOADER_B64` mode that executes the exact current worktree `MergedColumnParallelLinear.weight_loader` source body in-memory, allowing the GGUF tuple/data-container branch to be checked without copying files to AC922.

Verified output on AC922:

- `torch=2.10.0`, `CUDA_VISIBLE_DEVICES=''`, `cuda_initialized=False`.
- Runtime MergedColumn and conv helpers: all `3/3` rank layouts matched for segment-padded input.
- Tail-only padding: `3/3` QKV rank layouts and `3/3` conv rank layouts mismatched.
- Current worktree GGUF tuple path: all `3/3` Q/K/V data-container shards matched; tail-only padding mismatched `3/3` ranks.
- The exact current worktree MergedColumn method source hash used by the optional GGUF-path run was `bc29897a7c7dc2ff2c5a3afd4c94092e83a696dfb3d6fa604ba4f7c6a1ade84f`.

The remote `runtime-357d` MergedColumn implementation predates that worktree GGUF tuple support and raises `NotImplementedError` for GGUF multi-shard tuples. The optional fixture path therefore injects the exact current worktree method source into the remote CPU process; it performs no remote file write. The conv helper hash matches the local worktree helper.

The Python import root needed `latest-build-site`, `runtime-357d`, plus `runtime-525` and the existing venv site-packages for dependencies such as `typing_extensions`; no packages were installed. The import emitted a no-CUDA-runtime warning and did not initialize CUDA.

## Boundary

This verifies row/data motion through the actual shard helpers, including the current GGUF tuple branch. It does not validate the adapter's quantized row-padding bytes, dequantized zeros, GDN numerical output, or full model parity. The adapter owner has been notified that Q/K/V padding must be segment-aware.
