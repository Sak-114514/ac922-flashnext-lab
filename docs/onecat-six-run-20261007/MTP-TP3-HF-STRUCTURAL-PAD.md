# TP3 HF MTP structural weight adapter

Status: implemented behind the default-off `_ac922_tp3_enabled` config-view marker. CPU mini fixture passed through the current helper source and actual vLLM QKV/FusedMoE loaders. No model, GPU, CUDA, or service was run.

## HF sidecar path

`MTP-HF-SOURCE.json` identifies `Qwen/Qwen3.8-Flash-Next` revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Its MTP tensor names include `mtp.layers.0.self_attn.{q,k,v}_proj.weight`, shared-expert gate/up/down, and routed `experts.gate_up_proj` / `experts.down_proj`; it contains no MTP `linear_attn`/GDN weights. The model config supplies hidden size2560, QSA KV2/head-dim256, and MTP routed/shared expert width640.

## Opt-in weight handling

`Qwen4ExpMTP.load_weights` remaps checkpoint names first, then calls `pad_ac922_tp3_mtp_hf_weights` only when the loader-installed TP3 config marker is present. The default path passes every weight object through unchanged. Checkpoint FP8 MTP prefixes fail closed under this TP3 HF adapter; the intended sidecar is half precision.

The adapter applies these shape-only transforms:

- QSA K/V rows repeat by source head as `[0,0,0,1,1,1]`, preserving the query groups and providing six target KV heads.
- Shared gate/up matrices pad their 640 output rows to768 with zeros; shared down matrices pad their 640 input columns to768.
- The routed fused bank `[experts, 2*640, hidden]` is split into per-expert gate/up views, each padded separately and yielded under the existing `experts.{id}.{gate,up}_proj.weight` names. Routed down is similarly yielded per expert after padding its input width. The adapter never creates a second padded full expert bank.
- FC matrices, QSA query matrices, router, norms, and HC tensors pass through unchanged.

The existing `Qwen4ExpSparseMoeBlock` mapping includes per-expert names and `FusedMoE.load_weights` already accepts those 2D tensors. No shared FusedMoE code was changed.

## CPU mini fixture

Fixture: [mtp_tp3_hf_weights_cpu.py](<~/Desktop/ssh/reports/onecat-six-run-20261007/fixtures/mtp_tp3_hf_weights_cpu.py>).

It uses 2 experts, hidden width8, source expert width4, target width6, QSA head dim2, and TP3. It calls the helper from the current worktree source, then invokes actual `QKVParallelLinear.weight_loader` and actual `FusedMoE.make_expert_params_mapping` / `FusedMoE.load_weights` on CPU. It verifies:

- TP3 QSA K/V rank mapping is source heads `(0,0)`, `(0,1)`, `(1,1)`.
- Shared and routed source values remain exact; padded regions are zero.
- Routed outputs are per-expert 2D tensors and the full padded 3D bank is never produced.
- The actual FusedMoE loader writes each local gate/up/down slice to the expected expert slot on all three TP ranks.
- The disabled marker preserves source names and tensor identity.

AC922 run result: Torch2.10.0 CPU; all assertions passed; `CUDA_VISIBLE_DEVICES=''`; `torch.cuda.is_initialized()==False`. The existing `runtime-357d` imports were used with its `runtime-525` and venv dependency roots. No packages were installed.

For canonical half weights, the adapter's extra temporary is at most two expert matrices at a time: each padded matrix is `768 x 2560 x 2 B` (about3.75 MiB), rather than a padded `[512,1536,2560]` bank. Full checkpoint source tensors are still held by the safetensors loader while their expert slices stream through.

## Remaining gates

This proves mini shape/data motion and loader ownership, not the full sidecar load, numerical parity, or end-to-end model output. The production loader must install the same TP3 config view on the separate draft `ModelConfig`, and the actual HF sidecar load remains a parent-owned integration/parity gate. The GGUF target path remains separate and unchanged by this HF MTP transform.
