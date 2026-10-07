# Qwen4Exp PP GGUF source prefilter

Status: implementation is opt-in and CPU source-fixture verified. It has not
been exercised in a model load or a six-GPU run.

## Change

- `vllm/model_executor/model_loader/gguf_adapters/qwen4exp.py` adds a narrow
  stage-map filter for raw `blk.N.*` names where `N < num_hidden_layers`. It
  retains globals and mapped blocks at or beyond the decoder layer count. The
  `per_layer_token_embd.weight` table follows the configured PLE layer owner.
- `vllm/model_executor/model_loader/gguf_loader.py` exposes the default-off
  extra config `qwen4exp_pp_stage_prefilter`. When enabled with PP>1, it filters
  the GGUF map before the weight-type scan, PLE pinned preflight, and adapter
  weight conversion. Other GGUF loader extra-config keys remain rejected.
- Existing `PleOffloadWorker._init_offload_distributed()` removes
  `VLLM_PP_LAYER_PARTITION` before initializing its isolated PP1 group; that
  code was read and covered by the fixture, not changed here.

For the explicit `25,23` partition and 48 decoder layers, PP0 owns `[0,25)`;
PP1 owns `[25,48)`. The configured PLE id 2 is one-based, so the PLE table is
kept on PP0. The filter does not remove global embedding/output tensors or any
already-mapped `blk.N` where `N >= 48`.

The preservation rule applies at this filter boundary only. The inherited
`Qwen35Adapter.build_name_map()` separately skips block indices designated by
`num_nextn_predict_layers`; this change does not create or claim a native MTP
head mapping. Validate MTP ownership through its own loader path.

## Activation

Pass this loader extra config explicitly for a CPU/model experiment:

```text
--model-loader-extra-config '{"qwen4exp_pp_stage_prefilter": true}'
```

Without the key, or with it set to false, the map is unchanged. This is a
staged-loading candidate only; model parity and six-card acceptance remain
pending.

## CPU evidence

`pp_prefilter_fixture.py` AST-executes the pure Python functions from the live
source with CPU stubs. It verifies the 25+23 partition, retention of globals
and mapped MTP/future block entries, PP-local main decoder blocks, PLE table
ownership, opt-in default, call ordering before type-map/pinned-preflight/weight
conversion, worker partition-environment isolation, and host-budget arithmetic.
The result is in `cpu-prefilter-fixture.json`.

Exact command:

```bash
cd ~/Desktop/ssh/opensource/super-ac922-clone
PYTHONDONTWRITEBYTECODE=1 ~/Desktop/ssh/opensource/super-ac922-upstream-sync-20261007/.venv/bin/python -B reports/onecat-six-run-20261007/pp_prefilter_fixture.py > reports/onecat-six-run-20261007/cpu-prefilter-fixture.json
```

Result: `status=pass`, `cpu_only=true`, `gpu_or_service_used=false`;
`prefilter_before_type_map_pinned_preflight_and_weight_conversion=true`.
The independent AST syntax check and `git diff --check` also passed for both
changed source files. Ruff is unavailable in this virtualenv. No model fixture,
GGUF payload, GPU, CUDA build, or service was run by this lane.

`pp-prefilter.patch` contains the scoped source diff for review. Source hashes
used by the fixture are recorded in its JSON output.
