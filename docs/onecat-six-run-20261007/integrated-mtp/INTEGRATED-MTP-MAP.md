# Integrated Qwen4Exp GGUF MTP header mapping

Status: CPU header-map fix verified against the full integrated target GGUF.
No model payloads were decoded, and no GPU, service, or CUDA build was used.

## Root cause and fix

The target file
`/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf`
has 1256 header tensors: 1224 target tensors and 32 bundled tensors at
`blk.48.*`. The main Qwen4Exp config has 48 decoder layers; native MTP weights
are loaded separately from the HF safetensors source. The inherited mapper
treated `blk.48.attn_k.weight` as an unmapped layer because this GGUF does not
provide the generic `num_nextn_predict_layers` count needed for its old skip
rule.

`Qwen4ExpAdapter.build_name_map()` now skips only the exact 32 suffixes observed
for the integrated MTP block, and only within the configured MTP block range.
If `mtp_num_hidden_layers` is absent, it uses the native Qwen4Exp default of one
MTP layer; `num_nextn_predict_layers` can extend the range. It does not mutate
the config. Unknown suffixes in the MTP range, unknown tensors in target layers
0–47, and known suffixes beyond the configured MTP range still raise
`Unmapped Qwen4Exp GGUF tensor`.

## Full-header evidence

`flashnext-integrated-gguf-header-names.json` contains the complete 1256-name
header index from the integrated GGUF. It was read on AC922 with the source
`gguf_tensor_reader.GGUFReader`, which understands the file's Q2_0 type 42;
only header and tensor-directory names were emitted. The separate base two
shards contain 1224 tensors and no block 48 entries.

`verify_integrated_mtp_name_map.py` AST-executes the live Qwen4Exp and base
name-map functions over all 1256 raw names. It confirms 1224 mapped target
weights, 32 skipped `blk.48.*` weights matching the explicit suffix inventory,
and rejection of unknown `blk.0`, `blk.48`, and `blk.49` names. It also confirms
the canonical config fields remain unchanged. The same fixture checks that the
TP3 adapter leaves the generic dense `intermediate_size=5632` path unchanged;
only shared/routed MoE width 640 uses the 768 pad. The original 1224-name base
GGUF header also maps completely with no block-48 entries. Result:
`integrated-mtp-map-result.json`.

`tests/models/gguf/test_qwen4exp_adapter.py` also covers the bounded MTP skip
and rejects an unknown suffix at the first extra block.

Exact CPU command:

```bash
cd ~/Desktop/ssh/opensource/super-ac922-clone
PYTHONDONTWRITEBYTECODE=1 ~/Desktop/ssh/opensource/super-ac922-upstream-sync-20261007/.venv/bin/python -B reports/onecat-six-run-20261007/verify_integrated_mtp_name_map.py > reports/onecat-six-run-20261007/integrated-mtp-map-result.json
```

`integrated-mtp-adapter.patch` contains the adapter change and the narrow
unit-test regression. Parent handles remote deployment.
