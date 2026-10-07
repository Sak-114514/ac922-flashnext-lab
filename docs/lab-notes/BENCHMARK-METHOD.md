# llama-server Flash-Next benchmark method

`benchmark_server.py` is a decode-first harness for the local llama-server
HTTP API. It uses only Python's standard library.

## What the harness measures

For every flow it records the server's native `/completion` timings:

- `prompt_n` and `prompt_ms`, used for `prompt_tps`;
- `predicted_n` and `predicted_ms`, used for `decode_tps`;
- client `wallsec`, which includes HTTP and queueing and is not used as the
  decode time;
- the requested and actual input/output token counts, slot ID, full timings
  object, output token IDs plus their SHA-256, the first 200 characters and
  full SHA-256 of `content`, and any error response.

Each wave starts all requests through a `threading.Barrier` and
`ThreadPoolExecutor`. Every request sets `id_slot` to a separate slot,
`cache_prompt=false`, `return_tokens=true`, `temperature=0`, a fixed seed, and
`ignore_eos=true`.
The harness records both aggregate values explicitly:

- `decode_tps_sum_per_flow_server`: sum of each flow's server decode rate;
- `decode_tps_wave_wall`: total generated tokens divided by the whole wave's
  wall interval.

The second value is the completion aggregate throughput. It must not be
reported as the sum of instantaneous per-flow rates.

For MTP versus no-MTP parity, compare runs only when
`prompt_source.token_sha256` matches. Compare `output_token_ids` directly and
use `content_sha256` as the compact text-result check; the full generated text
is intentionally not stored.

A flow is marked successful only when all of these checks pass: `truncated` is
exactly `false`, response `id_slot` matches the requested slot,
`timings.prompt_n` equals the requested input token count, and
`timings.predicted_n` equals the requested output token count. The returned
token ID array must also be valid and have the same count as `predicted_n`.
Early EOS, truncation, prompt cache reuse, and any other count mismatch are
retained as failures together with the measured timings. `prompt_n` is
`n_prompt_processed` in llama-server's `server_slot_stats`; cached tokens are
tracked separately as `cache_n`. A one-token difference is classified as
`bos_plus_one` or `bos_minus_one` for diagnosis, but strict benchmark success
still requires the exact requested count.

## Exact input construction

The prompt is a fixed readable paragraph repeated until `/tokenize` returns at
least the requested number of tokens. The harness takes exactly the first
`--context-tokens` IDs and sends that integer array as `prompt` to the native
`/completion` endpoint. This uses llama-server's documented token-array form,
so the measured prompt is not based on a character-to-token estimate and is
not a zero-filled token buffer. The JSON keeps the selected count, a SHA-256
of the exact ID sequence, and first/last ID samples.

Flow 0 uses the original `PROMPT_UNIT` content unchanged. For flow 1 and later,
the harness prepends a readable unique line such as `Concurrent request
identifier: stream 1.` before the same repeated base paragraph, then tokenizes
that complete content and truncates to the exact requested length. Each flow
has its own prompt source and token SHA-256 in the JSON. This prevents a
concurrent wave from submitting identical token sequences while preserving
flow 0 and cross-configuration MTP/no-MTP parity; changing only the seed would
not provide this separation at temperature zero.

## Server capacity evidence

Before each benchmark invocation the harness captures `/health`, `/props`, and
`/slots`. It retains the raw responses plus the reported per-slot `n_ctx`,
slot count, and slot states. The JSON labels these as llama.cpp server
context/KV capacity reports; they are not a vLLM token-pool measurement. The
llama-server API does not expose per-GPU VRAM, so supply
`--gpu-memory-mib 32768,32768,...` when that value is already measured, or
record the corresponding `nvidia-smi` evidence alongside the JSON. No GPU
query is performed by this script.

The server should be started with its normal `--slots` endpoint enabled. If
the endpoint is unavailable, the harness still records the requested slot IDs
but marks `slot_assignment_verified=false`; such a result does not prove
independent slot placement.

## Matrix

Run one server configuration at a time and use a label that identifies whether
MTP is enabled. The MTP and no-MTP servers must be started separately by the
operator; this harness does not start or stop services.

Required decode-first matrix:

```sh
# Replace the URL and VRAM values with the local server values.
python3 benchmark_server.py --base-url http://127.0.0.1:8080 \
  --label no-mtp-c1 --context-tokens 8192 65536 --concurrency 1 \
  --output benchmark.json --gpu-memory-mib 32768,32768,32768,32768,32768,32768

python3 benchmark_server.py --base-url http://127.0.0.1:8080 \
  --label no-mtp-c2 --context-tokens 8192 65536 --concurrency 2 \
  --output benchmark.json

python3 benchmark_server.py --base-url http://127.0.0.1:8080 \
  --label no-mtp-c4 --context-tokens 8192 65536 --concurrency 4 \
  --output benchmark.json

python3 benchmark_server.py --base-url http://127.0.0.1:8080 \
  --label no-mtp-c1-128k --context-tokens 131072 --concurrency 1 \
  --output benchmark.json
```

Repeat the four invocations with `mtp` labels after starting the MTP server.
128K multi-flow waves are optional because they can consume the available KV
capacity; the required 128K coverage is single flow for both configurations.
Each invocation performs one wave per requested context length, with 512
output tokens by default. It is intentionally not a long-duration stress
test. A failed flow or failed wave returns a non-zero process exit status, and
the complete failure is retained in the JSON.

## API facts used

The source tree's server routes define:

- `POST /tokenize` with `content`, `add_special`, `parse_special`;
- native `POST /completion` with integer token-array `prompt`;
- `GET /props` with `default_generation_settings.n_ctx` and `total_slots`;
- `GET /slots` with per-slot `id`, `n_ctx`, `is_processing`, and timing state.

Native completion responses contain `timings.prompt_n`, `prompt_ms`,
`predicted_n`, and `predicted_ms`; the harness uses those fields instead of
deriving decode speed from network wall time. The source's
`server_slot_stats::to_json()` reports `predicted_n` as `n_gen`, while its
server-provided `predicted_per_second` uses `n_gen - 1` generation steps
because the first generated token comes from prompt logits. The JSON retains
the full native timings object and reports its own token-count-over-time rate
consistently for every flow.
