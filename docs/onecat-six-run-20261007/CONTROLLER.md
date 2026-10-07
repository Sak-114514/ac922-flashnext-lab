# Six-card C1 controller

`benchmark_six.py` is the only runner in this lane. It first validates the
private wheel/site, pinned GGUF and completed MTP safetensors sidecar, then
checks the real-token input/oracle. The default mode refuses to start if any
check fails, if any target GPU is already running a compute process, or if the
host health gates fail.

## Workload and evidence

- TP3 × PP2, `VLLM_PP_LAYER_PARTITION=25,23`, in the UUID order from
  `reports/onecat-sync-20261007/TARGET-SIX-CARD.json`.
- Native MTP4 from `mtp-hf-native/`, safetensors `model.safetensors.index.json`
  pinned to Qwen/Qwen3.8-Flash-Next revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`. Preflight requires the completed
  31-tensor provenance and rejects `.partial` files. The parent-reported final
  sidecar SHA-256 is
  `d50d273a87f51d9b7de3142a3f93f2c2823cd36a6fbf0f53f08303ce33029c17`.
  Preflight also pins the untouched official config SHA, source model-index and
  MTP-map hashes, and the generated 31-entry safetensors index hash.
- The sidecar `config.json` must remain the pinned official target config:
  `model_type=qwen4_exp`, `architectures=[Qwen4ExpForConditionalGeneration]`.
  The pinned vLLM `SpeculativeConfig` normalizer changes the loaded draft view
  to `qwen4_exp_mtp` / `Qwen4ExpMTP` for `method="mtp"`; the worker verifies
  that runtime result and hashes `vllm/config/speculative.py`. It never rewrites
  the sidecar config.
- CUDA graph stays enabled: compilation mode 3, resolved decode `FULL`, mixed
  `PIECEWISE`; no eager-mode switch. Loader extras are `ac922_tp3=true` and
  `qwen4exp_pp_stage_prefilter=true`.
- The worker first checks the saved real 8K prompt's next-token logits against
  the saved oracle. A first-token mismatch ends the worker before timing.
- After that gate, exactly one greedy C1 request runs on the same 8192 saved
  token IDs with 256 output tokens. The result stores input/output token IDs,
  hashes, end-to-end wall time/rate, and separate steady-decode metrics from
  `RequestOutput.metrics.last_token_ts - first_token_ts`. If those request
  metrics are missing, steady decode rate is `null`; end-to-end rate is never
  labeled decode throughput.
- `cudagraph_metrics` is enabled for the timed request. The worker observes
  scheduler graph stats and requires an actual FULL MTP4 step with five
  unpadded tokens whose padded graph size appears in every rank's pre-request
  FULL capture list. A valid padded graph shape can therefore be 8 without
  hard-coding 5 as the capture size.
- The 92.267 tok/s upstream result is a TP4/PP1 repeated-piece synthetic
  prompt reference. This run uses saved real-corpus token IDs, so the script
  marks the rates as non-comparable. A matched synthetic run is a separate
  follow-up and is deliberately outside this short run.

The worker records loaded `vllm` and `_C` paths/hashes, the installed patched
Python source hashes, PyTorch/CUDA/Python versions, resolved graph policy,
worker graph routes, model paths and MTP source/index provenance. vLLM cache
and Hugging Face cache are isolated under each run's `stage-cache/` directory.

## Safety and cleanup

- Before launch: require at least 8 GiB available RAM, at most 5 GiB used swap,
  readable host thermal sensors below 58 °C, every target V100 below 58 °C,
  at least 512 MiB free on each GPU, and no active GPU compute process.
- Hold `.onecat-six-run-20261007.gpu.lock` for the full run. A competing run
  that follows this lane cannot acquire it.
- Start only a unique `systemd --user` unit with `RuntimeMaxSec=1200s` and
  `KillMode=control-group`. Its `ExecStart` first invokes
  `/usr/local/bin/with-ac922-cuda525`, which kernel-checks `6.1.0-53`, prepends
  the CUDA 12.4/NVIDIA 525 compatibility libraries, and sets the pinned
  `TRITON_PTXAS_PATH`; the worker verifies these exports. The controller
  rechecks host RAM/swap and
  temperatures while it runs. A health failure or Ctrl-C stops only that
  exact unit. It does not stop or modify other services.
- Outputs go under
  `results/onecat-six-run-20261007/onecat-six-bench-<UTC timestamp>/`.
  Inspect `controller.json`, `benchmark.json`, and the unit journal. The
  worker shuts down its local vLLM engine before exit.
- This short lane does not run C4 or natural-prompt cohorts and does not
  install/build software, start an API server, or change model/runtime files.

## Stage and run after the private install is ready

Run from `~/Desktop/ssh`. The host alias below is `ac922`; change it
only if the configured SSH alias differs. These commands stage only the
benchmark script and small prompt/oracle JSON files; the model and sidecar
remain in their prepared remote locations.

```bash
REMOTE=ac922
RUN=/home/debian/ac922-vllm/experiments/upstream-six-20261007
ssh "$REMOTE" "mkdir -p '$RUN/inputs' '$RUN/results/onecat-six-run-20261007'"
scp reports/onecat-six-run-20261007/benchmark_six.py "$REMOTE:$RUN/benchmark_six.py"
scp flashnext-llama-experiment/final-results/ROUND19/real-corpus-token-ids.json \
  "$REMOTE:$RUN/inputs/real-corpus-token-ids.json"
scp onecat-flashnext-experiment/real-oracle-8k-ids.json \
  "$REMOTE:$RUN/inputs/real-oracle-8k-ids.json"
```

CPU-only fixture and no-query plan:

```bash
ssh "$REMOTE" "'$RUN/source/.venv/bin/python' '$RUN/benchmark_six.py' \
  --self-test --input-json '$RUN/inputs/real-corpus-token-ids.json' \
  --oracle-json '$RUN/inputs/real-oracle-8k-ids.json'"
ssh "$REMOTE" "'$RUN/source/.venv/bin/python' '$RUN/benchmark_six.py' --plan-only"
```

Only after `runtime-site/` is installed and the final MTP shard/index/provenance
are present, launch the guarded controller:

```bash
ssh "$REMOTE" "'$RUN/source/.venv/bin/python' '$RUN/benchmark_six.py'"
```

The controller prints the unique unit name and result paths. For its output:

```bash
ssh "$REMOTE" "journalctl --user -u onecat-six-bench-<UTC timestamp>"
```

The parent reports the MTP sidecar and its hash verification complete. The
native build/install is still in progress, so `runtime-site/` is the remaining
launch prerequisite. Do not launch until that install is complete. No GPU or
service command was executed while preparing this artifact.
