# 1Cat Flash-Next TP4 reference for the six-card run

**Checked:** 2026-10-07. The accessible sync tree is `bc01f2899db721a99788f3bc2abc041e4b9a7fd0`; upstream `main` now resolves to `b5e92655032f33656b5efb2b781d696edca8c1e6`. At both commits the relevant README, Flash-Next GGUF design record, and benchmark script have identical SHA-256, so the sync-tree data below matches the latest checked `main`.

## Best full-model GGUF/MTP4 reference

The closest official same-family reference is **Flash-Next GSQ-RCO IQ3_S GGUF + FP16 MTP4**, not a pure IQ3_XXS model. The design record says its gate/up weights are mixed: IQ3_XXS in 17 layers, IQ2_S in 20, IQ3_S in 10, and IQ4_XS in one. The separate IQ3_XXS operator PR does not turn this into an all-IQ3_XXS end-to-end benchmark.

| Contract | Original ring-auto composition | Later development composition |
|---|---:|---:|
| C1 · one request · 8,192 input / 256 output | 41.499 ms/round · **75.212 tok/s** | 26.072 ms/round · **92.267 tok/s** |
| C4 · four requests · 128 input / 1,024 output each | 65.624 ms/round · **262.269 aggregate tok/s** | 52.059 ms/round · **280.441 aggregate tok/s** |

These are full model MTP4 results on **4× V100-SXM2-32GB**, TP4, with direct NVLink edges 0–1, 0–2, 1–3, 2–3 and SYS diagonals. The reported stack is driver 580.173.02, Torch 2.10 / CUDA 12.8 / Python 3.12.3, FP16 activation/KV/MTP weights, FP32 SSM state and accumulation, ring policy `auto`, FULL decode graphs, max model length 8,704, four sequence slots, batch budget 512 and GPU memory utilization 0.9. The later result comes from the source-containing `1.5.2.dev915+g3b07077f7` build.

The latest composition improves over the initial ring-auto record by 22.68% at C1 and 6.93% at C4, but the source record cautions that fixed-length output trajectories changed between builds and that operator changes cannot be credited with the whole-model difference. The original C1 has one repeat; later candidates have three. Within the latest build, all three fixed-length C1/C4 output sequences and speculative counters repeat exactly. Four separate natural-greedy checks (Paris, arithmetic, Chinese translation, and a 63-token Rayleigh-scattering explanation) finish at EOS and match the original token IDs; this is a small text-health check, not a broad quality suite.

## Prompt and measurement semantics

The C1/C4 timing cohorts are **synthetic fixed-length decode**, not natural user prompts. The benchmark builds each input from a repeated tokenized string (`Independent stream {index}: numerical methods and reliable systems.`), uses greedy sampling with seed 4201 and `ignore_eos=True`, warms the cohort, then records three fixed-length repeats. C1 times the 8K input; C4 times four simultaneous 128-token inputs, not four 8K prompts. C4's 280.441 tok/s is aggregate across the four streams, not per-request throughput.

The natural-prompt check is a separate path that reads `PROMPTS.json`, respects EOS and records the completed token IDs/text. The public command and report use placeholders for `MODEL.gguf`, `MTP_MODEL` and `PROMPTS.json`; the changed PR files do not include those external model/prompt artifacts or their hashes. The workload contract is reproducible, but a bit-for-bit rerun needs the same checkpoint, draft weights and natural-prompt file recovered independently. The script config also disables prefix caching and requests FULL decode graphs.

**CLI caveat:** the published command spells `--prefill` without any following lengths, while the script declares it as `nargs="*"` with default `[8192, 32768]`; standard `argparse` turns that invocation into an empty prefill list. The reported C1/C4 decode numbers are still the fixed-length decode cohorts. To run the additional prefill probes, pass explicit prefill lengths and record that exact argv.

The upstream reproduction template is:

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

Source enables `num_speculative_tokens=4` with greedy draft sampling when `--mtp-draft` is supplied. Results are complete engine rounds divided by steady decode intervals; prefill/TTFT are excluded. The README and PR keep full-round latency, emitted tokens/s and speculative acceptance as distinct metrics.

## Other official references, kept separate

| Route | Hardware / workload | Reported result | Why it is not the GGUF/MTP4 reference |
|---|---|---:|---|
| Qwen3.8-Flash-Next-NVFP4, no MTP (#415) | 4× V100-SXM2-32GB, TP4, 8K/512, full CUDA Graph | 65.864 control → **80.732 tok/s** | NVFP4 target, no MTP, output length 512. The PR's speed gate is fixed-length/greedy with EOS ignored; GSM8K and natural-stop checks are separate. |
| Qwen3.8-Flash-Next-NVFP4 + MTP4 (#796) | 4× V100-SXM2-32GB, TP4; C1 31,744/256, C4 four 4,096/1,024 requests | C1 79.832 → **92.199 tok/s**; C4 225.400 → **227.778 aggregate tok/s** | NVFP4 target and a 32K-capacity workload, not 8K GGUF. C4 change is only +1.05%; 21/21 paired quality sequences match. |
| IQ3_XXS/IQ3_S shared-activation pair (#966) | SM70 operator, M8/N4352/K5120 | canonical 86.016 → **57.344 μs** (reverse orientation 87.040 → 58.368 μs) | Seven additional mixed gate/up pairs; approximately 0.201 ms/round is a weighted microbenchmark estimate. PR explicitly does not claim end-to-end speed. |
| README 16GB headline | 4× V100 16GB, Qwen3.8-27B-NVFP4 + DFlash2 | ≈260 tok/s | Different target model, NVFP4 weights, DFlash2 drafter and 16GB/card capacity. Do not compare horizontally with Flash-Next GGUF/MTP4. |

The checked first-party sources contain **no full-model pure-IQ3_XXS 4-card throughput result**. The closest full-model GGUF run is the mixed IQ3_S-named Flash-Next checkpoint above. #966 is the relevant IQ3_XXS kernel evidence, and it is operator-level only.

## Practical target for the six-card experiment

Use 75.212 C1 / 262.269 aggregate C4 as the historical 4×32GB ring-auto GGUF/MTP4 control and 92.267 / 280.441 as the later 4×32GB development reference. These numbers are context for a same-machine test, not acceptance targets for six 16GB cards. A 4×32GB system has 128 GB nominal VRAM; six 16GB cards have 96 GB and a different rank/topology layout. The local 2026-10-03 record describes the six-card AC922 as two 3-GPU NVLink islands; this was not re-probed in this read-only pass.

For a defensible six-card result, first establish a control on the exact six-card build and model composition, then keep C1 8K/256 and C4 128/1024-per-request as separate workloads. Report per-request and aggregate rates, round latency, MTP acceptance, EOS/natural-prompt checks, model/draft hashes, and topology. Require at least three fixed-length repeats and compare against that same-hardware control; do not project a linear 4-to-6-GPU gain or use the 260 tok/s 27B DFlash2 headline as a Flash-Next target.

## Evidence links

- [Current README: real-model table and GGUF/Flash-Next references](https://github.com/1CatAI/1Cat-vLLM/blob/b5e92655032f33656b5efb2b781d696edca8c1e6/README.md#L31-L56) and [GGUF composition/results](https://github.com/1CatAI/1Cat-vLLM/blob/b5e92655032f33656b5efb2b781d696edca8c1e6/README.md#L820-L833).
- [Synced-tree workload and topology](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/docs/design/sm70_flashnext_gguf_mtp4_operator_routes.md#L15-L29), [original/current composition, token outputs and caveats](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/docs/design/sm70_flashnext_gguf_mtp4_operator_routes.md#L41-L71), [IQ3_XXS mixed-layer distribution](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/docs/design/sm70_flashnext_gguf_mtp4_operator_routes.md#L97-L101), [final C1/C4 record and command](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/docs/design/sm70_flashnext_gguf_mtp4_operator_routes.md#L258-L328).
- [Benchmark source: synthetic vs natural paths](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/benchmarks/benchmark_gguf_model.py#L3-L7), [TP4/MTP4 and runtime config](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/benchmarks/benchmark_gguf_model.py#L101-L129), [separate natural prompt check](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/benchmarks/benchmark_gguf_model.py#L244-L259), [synthetic prompt construction and fixed-length timing](https://github.com/1CatAI/1Cat-vLLM/blob/bc01f2899db721a99788f3bc2abc041e4b9a7fd0/benchmarks/benchmark_gguf_model.py#L261-L286).
- [#940 merged benchmark record](https://github.com/1CatAI/1Cat-vLLM/pull/940), [#415 NVFP4 target-only gate](https://github.com/1CatAI/1Cat-vLLM/pull/415), [#796 NVFP4/MTP4 admission and matched quality result](https://github.com/1CatAI/1Cat-vLLM/pull/796), [#966 IQ3_XXS operator result](https://github.com/1CatAI/1Cat-vLLM/pull/966).
- [Local historical six-card topology record](<~/Desktop/ssh/onecat-flashnext-experiment/PP3X3-STATE.md:1>).

No GPU, CUDA build, server, or local source was touched for this reference review.
