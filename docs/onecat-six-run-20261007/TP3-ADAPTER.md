# AC922 TP3 Qwen4Exp GGUF adapter wiring

Status: experimental, default-off, CPU source-fixture verified. No model
loading, six-GPU run, CUDA build, or service startup was performed in this lane.

## Activation

The loader accepts the boolean extra config `ac922_tp3`. Combine it with the PP
stage prefilter for the 25+23 staged-loading experiment:

```text
--model-loader-extra-config '{"ac922_tp3":true,"qwen4exp_pp_stage_prefilter":true}'
```

Both keys default to false. `ac922_tp3` rejects non-TP3 use, non-native GGUF
adapters, non-boolean values, and configs that do not match the canonical
Flash-Next source geometry.

## Config and weight flow

The loader builds the GGUF name map and weight-type map using the canonical
source config. It then deep-copies the target HF config, obtains the marked
TP3 view, installs that same object as both `hf_config.get_text_config()` and
`ModelConfig.hf_text_config`, and verifies the accessor identity before model
construction. The Qwen4Exp adapter keeps its canonical source config for
existing GGUF inverse transforms. When native MTP has a separate draft config,
the loader applies the same view to that config before the MTP constructor.

The view keeps logical hidden size 2560 and vocabulary size 248320. It sets
QSA KV heads to 6, GDN heads to K18/V54, MoE intermediate widths to 768, and
marks vocabulary storage padding 192. Model/MTP constructors consume the marker
for embedding and output-head padding; MTP FC projections stay full 2560×2560
and use replicated linears under the marker.

Adapter source padding happens after canonical inverse transforms:

- QSA K/V output rows are cloned using `[0,0,0,1,1,1]`, producing target KV
  heads `[0,0]`, `[0,1]`, and `[1,1]` across the three TP ranks.
- GDN Q/K/V and convolution outputs receive zero rows inside their Q, K, and V
  segments. Z, A/B, A-log, and DT-bias receive zero rows for added V heads. The
  output projection receives zero packed input blocks for the added V heads.
  PLE tensor and table formats remain unchanged.
- Regular/shared MoE gate/up weights append zero rows; down weights append zero
  input blocks. Routed expert tensors use the same 768-wide packed geometry and
  are padded lazily one tensor at a time. Q2_0 down rows retain their source 180
  bytes and append 36 zero bytes (10 to 12 blocks of 18 bytes); the adapter
  disables the source-width fallback so no Q2_0-to-Q4_1 rewrite is selected.
- MTP blocks remain outside the main-decoder stage filter. The inherited
  adapter may exclude nextn-designated blocks while building the main map; this
  wiring does not invent a separate MTP head mapping.

The config and tensor-shape changes are a structural candidate. Exact source
packed bytes and appended zero areas pass CPU checks, but runtime numerical
parity and output parity remain pending.

## CPU evidence

`tp3_adapter_fixture.py` AST-executes the live adapter and loader methods with
NumPy-backed CPU stubs. It verifies bit-exact QSA KV replication, canonical
GDN segment padding, Q2_0 output-projection and MoE down-row byte extents,
zero-filled tails, routed expert padding on access, untouched source config,
and matching target/draft text-config accessors. Its result is
`cpu-tp3-adapter-fixture.json`.

Exact command:

```bash
cd ~/Desktop/ssh/opensource/super-ac922-clone
PYTHONDONTWRITEBYTECODE=1 ~/Desktop/ssh/opensource/super-ac922-upstream-sync-20261007/.venv/bin/python -B reports/onecat-six-run-20261007/tp3_adapter_fixture.py > reports/onecat-six-run-20261007/cpu-tp3-adapter-fixture.json
```

The existing geometry fixture also passed all 11 tests, including the 640→768
packed MoE shapes and config-view opt-in markers. The adapter fixture covers
all 12 GGUF formats listed by source metadata, including IQ3_XXS gate K=2560/
N=640→N=768 and GDN output K=6144→6912 at 256-value block size, and checks
convolution's segmented Q/K/V padding. Both changed Python source
files pass AST compilation and `git diff --check`. Ruff is unavailable in this
virtualenv; its Python environment also has no `torch`, so no imported adapter
test or model load was possible. The fixture result records source hashes.

`source-integration.patch` contains the scoped `qwen4exp.py` and
`gguf_loader.py` diff. The independent PP stage-map evidence is in
`PP-PREFILTER.md` and `cpu-prefilter-fixture.json`.
