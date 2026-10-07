# Flash-Next GGUF target + MTP4 load contract

Checked against upstream sync commit `bc01f2899db721a99788f3bc2abc041e4b9a7fd0`, the #940 benchmark/design record, local conversion artifacts, and a read-only listing of `/home/debian/models`. No GPU, CUDA build, service, source edit, or original weight payload read/write was performed.

## Answer

#940 runs **two model paths in one vLLM instance**:

- `MODEL.gguf` is the quantized Flash-Next target. The benchmark recognizes the `.gguf` suffix and sets `quantization="gguf"`; `GGUFModelLoader` selects `Qwen4ExpAdapter`.
- `MTP_MODEL` is a separate draft checkpoint loaded with `load_format="safetensors"`, `method="mtp"`, four speculative tokens, and greedy draft sampling. The proposer constructs a separate `draft_model_config` from that path and passes its independent `draft_load_config` to `get_model()`.

The #940 benchmark code is the concrete loader recipe. Its public report leaves `MODEL.gguf` and `MTP_MODEL` as placeholders; it gives no draft path, source revision, or checkpoint hash. [#940 PR and result](https://github.com/1CatAI/1Cat-vLLM/pull/940) · [benchmark config](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/benchmarks/benchmark_gguf_model.py#L101-L129) · [separate draft-config forwarding](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/v1/spec_decode/llm_base_proposer.py#L2142-L2151)

A load-config sketch matching that source is:

```python
config = {
    "model": "MODEL.gguf",
    "quantization": "gguf",
    "dtype": "half",
    "speculative_config": {
        "method": "mtp",
        "model": "MTP_MODEL",
        "num_speculative_tokens": 4,
        "draft_load_config": {"load_format": "safetensors"},
        "draft_sample_method": "greedy",
    },
}
```

The same #940 benchmark command is:  

```bash
python benchmarks/benchmark_gguf_model.py MODEL.gguf \
  --mtp-draft MTP_MODEL --ring auto --require-installed \
  --prompts-json PROMPTS.json --kv-cache-dtype float16 \
  --ssm-state-dtype float32 --widths 1 4 \
  --input-len 8192 --output-len 256 --concurrent-input-len 128 \
  --concurrent-output-len 1024 --prefill --max-model-len 8704 \
  --max-seqs 4 --max-batch 512 --repeats 3 \
  --gpu-memory-utilization 0.9 --output RESULT.json
```

`MODEL.gguf`, `MTP_MODEL`, and `PROMPTS.json` remain placeholders in the public record.

The draft path must resolve to the registered `Qwen4ExpMTP` architecture and its HF-style checkpoint names; vLLM constructs the separate draft `ModelConfig` from that path. The exact sidecar config is not included in #940. The upstream model registry includes `Qwen4ExpMTP`; its loader remaps MTP/shared-head names and filters non-MTP weights. [MTP registry entry](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/model_executor/models/registry.py#L648-L654) · [Qwen4Exp MTP name mapping](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/models/qwen4_exp/nvidia/mtp.py#L109-L136) · [head aliases](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/docs/design/speculative_draft_load_config.md#L1-L19)

## Why the target GGUF's `blk.48` is not the draft input

For GGUF metadata `block_count=49` and `nextn_predict_layers=1`, vLLM derives `num_hidden_layers=48` plus `num_nextn_predict_layers=1`. The target adapter then intentionally skips `blk.48.*`: the target backbone does not execute nextn and the draft loader owns those weights. `Qwen4ExpAdapter` delegates to this target name-map logic. [GGUF config derivation](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/transformers_utils/gguf_config.py#L79-L108) · [target map's `blk.48` skip](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/model_executor/model_loader/gguf_adapters/qwen35.py#L84-L103) · [Qwen4Exp adapter delegation](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/model_executor/model_loader/gguf_adapters/qwen4exp.py#L60-L73)

The GGUF adapter registry maps `qwen4_exp_text` to `Qwen4ExpAdapter`; it does not register an MTP GGUF adapter. `Qwen4ExpMTP` is a separate model class, not a GGUF weight adapter. So appending `blk.48.*` to a target GGUF does not supply the draft to #940. [GGUF adapter dispatch](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/model_executor/model_loader/gguf_adapters/__init__.py#L6-L25)

## Existing sidecar format and local availability

The upstream Qwen4Exp MTP loader prefers four compact `model-bf16-*.safetensors` shards; generic safetensors/bin/pt layouts remain supported. PR #389 says this avoids rescanning all 206 target shards. If only the full compatible HF checkpoint is available, `Qwen4ExpMTP.skip_checkpoint_weight()` filters to recognized MTP/shared-head names, but the loader still has to scan that checkpoint. The #940 benchmark config uses `dtype="half"`; its design record says MTP weights remain FP16, but does not publish the sidecar file/hash. [compact-shard patterns](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/models/qwen4_exp/nvidia/mtp.py#L538-L549) · [MTP-only filter/load](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/vllm/models/qwen4_exp/nvidia/mtp.py#L681-L709) · [#389 four-shard note](https://github.com/1CatAI/1Cat-vLLM/pull/389)

The exact #940 sidecar's provenance is **not published**. #389 documents compact BF16 safetensors from the Qwen4Exp checkpoint; it does not establish that #940's `MTP_MODEL` was derived from a particular Q8 GGUF. The #940 benchmark itself only pins its sidecar *loader format* to safetensors.

Read-only checks found no compact `model-bf16-*.safetensors` or other MTP safetensors file under the local `onecat-flashnext-experiment` / `flashnext-llama-experiment` workspaces or `/home/debian/models`. The remote MTP directory contains one separate `shared-Q8_0.gguf` and its small verification JSON. The local [download manifest](<~/Desktop/ssh/flashnext-llama-experiment/downloads.json:20>) records the IQ3_XXS target shards from `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF@ed59f920…`, while that MTP GGUF is separately downloaded from `peasantsmith/qwen38-flash-next-v100-recipe@e7208cdae7be29418f38e611ede90a35726e888f`. This proves separate input files/repos; it does not reveal the upstream origin used to make the sidecar.

## Existing GGUF preparation tool: useful for GGUF consumers, not the #940 loader

The local [`append_mtp_gguf.py`](<~/Desktop/ssh/flashnext-llama-experiment/append_mtp_gguf.py:1>) already accepts two target GGUF shards plus one standalone MTP GGUF. It requires `qwen4exp`, `block_count=48` target / `49` MTP, `nextn_predict_layers=1`, `nextn_shared_target_tensors=true`, matching metadata and `blk.48.*` tensors; its CLI also requires a verified input manifest for a non-dry-run. [validator](<~/Desktop/ssh/flashnext-llama-experiment/append_mtp_gguf.py:142>) · [input/output contract](<~/Desktop/ssh/flashnext-llama-experiment/append_mtp_gguf.py:303>) · [CLI](<~/Desktop/ssh/flashnext-llama-experiment/append_mtp_gguf.py:422>)

The existing [append verification record](<~/Desktop/ssh/flashnext-llama-experiment/mtp-append-verification.json:2664>) is `merged-byte-verified` for 1,224 target tensors plus 32 MTP tensors and explicitly limits proof to metadata/raw bytes—no inference claim. It produces an integrated `.gguf`, not a safetensors draft. The target adapter will still skip `blk.48`, so this artifact is not a drop-in `MTP_MODEL` for upstream vLLM. No GGUF→Qwen4ExpMTP-safetensors exporter was found in the checked upstream/local tools.

## Recommended reuse path

1. Keep the #940 split: the quantized GGUF stays the target; obtain the matching compact BF16 MTP safetensors sidecar and its config/index/hash from the original Qwen4Exp checkpoint bundle.
2. Pass that sidecar directory as `--mtp-draft MTP_MODEL`; retain `draft_load_config.load_format="safetensors"`, four speculative tokens, and the ordinary `Qwen4ExpMTP` loader. This avoids rewriting a loader and avoids loading the full 206-shard target a second time.
3. Do not pass the current `shared-Q8_0.gguf` to #940's safetensors draft slot, and do not treat the appended GGUF as an MTP sidecar. If the compact HF sidecar cannot be recovered, the missing prerequisite is an exported MTP safetensors artifact with the expected Qwen4ExpMTP config/name layout; the checked repository contains no converter from this Q8 GGUF into that artifact.

The #940 benchmark run is full-model MTP4 evidence, but its published command leaves `MTP_MODEL` and model hashes as placeholders. A compact HF sidecar's exact identity must be confirmed before claiming the local IQ3_XXS target reproduces that run.
