# Official BF16 MTP sidecar download

## Status

Complete and independently verified. The sidecar contains only the 31 original BF16 `mtp.*` tensors from the official `Qwen/Qwen3.8-Flash-Next` checkpoint pinned at `de4b8e4d43b917e7706784d8bb445c9af86a3540`. The selected tensors span 28 source shards; no full target shard or model was downloaded.

Remote artifact directory:

`/home/debian/ac922-vllm/experiments/upstream-six-20261007/mtp-hf-native/`

Main files:

- `model-bf16-00001-of-00001.safetensors` — 5,214,305,456 bytes, SHA-256 `d50d273a87f51d9b7de3142a3f93f2c2823cd36a6fbf0f53f08303ce33029c17`
- `model.safetensors.index.json` — maps the 31 selected MTP keys to that shard.
- `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json` — copied from the same pinned public revision without edits.
- `mtp-source-provenance.json` — per-tensor source shard, offsets, shape, dtype, byte count, SHA-256, official file hashes, and verification record.

The payload is 5,214,301,696 bytes (4.856197 GiB). The safetensors header is 3,752 bytes, padded to an 8-byte boundary; tensor data offsets are contiguous and end exactly at the payload size.

## Download contract

The downloader is [download_mtp_sidecar.py](~/Desktop/ssh/reports/onecat-six-run-20261007/download_mtp_sidecar.py). It uses the Python standard library and streams each selected source range directly into the remote output file. Each range requires HTTP 206, an exact matching `Content-Range`, the pinned source file length, and the expected byte count. Four workers and 16 MiB range chunks were used. Tensor payload SHA-256 values were calculated as streamed, then compared with each output tensor during the final sequential verification pass.

Exact remote invocation:

```sh
cd /home/debian/ac922-vllm/experiments/upstream-six-20261007/mtp-hf-native
export HTTP_PROXY=http://127.0.0.1:19197
export HTTPS_PROXY=http://127.0.0.1:19197
export ALL_PROXY=http://127.0.0.1:19197
export http_proxy=http://127.0.0.1:19197
export https_proxy=http://127.0.0.1:19197
export all_proxy=http://127.0.0.1:19197
/home/debian/ac922-vllm/experiments/upstream-six-20261007/source/.venv/bin/python \
  download_mtp_sidecar.py \
  --source-json MTP-HF-SOURCE.json \
  --index-json hf-index.json \
  --model-info-json hf-model-info.json \
  --dest . --workers 4 --chunk-mib 16
```

The remote loopback proxy port was a temporary SSH reverse-forward to the workstation's already configured proxy. No proxy configuration was changed; the temporary tunnel was closed after download and validation.

## Loader and precision notes

The sidecar's `config.json` is the official target config (`model_type: qwen4_exp`) and was not rewritten. In the pinned vLLM tree, MTP config normalization maps `qwen4_exp` to `qwen4_exp_mtp` and selects `Qwen4ExpMTP` for MTP draft loading (`vllm/config/speculative.py`); the model's loader accepts `model-bf16-*.safetensors` and the standard safetensors index (`vllm/models/qwen4_exp/nvidia/mtp.py`).

These are original HF BF16 MTP weights. The intended target backbone is separately loaded from a quantized GGUF, so this is a mixed-precision recipe; it is not a same-precision checkpoint pair. Artifact integrity is established, but this task did not start a model server or run an inference/parity test.

## Verification

- Local Python syntax check: `python3 -m py_compile download_mtp_sidecar.py` passed.
- Remote run: `COMPLETE tensors=31 payload_bytes=5214301696 file_bytes=5214305456 sha256=d50d273a87f51d9b7de3142a3f93f2c2823cd36a6fbf0f53f08303ce33029c17`.
- Independent remote stdlib audit reread the final shard, parsed the safetensors header and index, checked all 31 tensor layouts and payload hashes, verified the whole-file hash and all four pinned official-file hashes: `INDEPENDENT_AUDIT_PASS tensors=31 payload_bytes=5214301696 file_bytes=5214305456 file_sha256=d50d273a87f51d9b7de3142a3f93f2c2823cd36a6fbf0f53f08303ce33029c17 official_files=4`.
- No GPU, model service, or source-tree write was used.

The local mirror includes the downloader, source map/index/model metadata, safetensors index, and provenance JSON. The 5.2 GB weight shard remains only in the requested remote private lane.
