# AC922 TP3 structural geometry candidate

Status: CPU geometry and constructor hooks are implemented behind an explicit opt-in config view. The loader owner still needs to call the view API for target and draft configs and coordinate the matching GGUF weight transforms. No GPU or runtime validation was run.

## Canonical source geometry

The fixture records the original Flash-Next HF config and GGUF metadata hashes:

- `model/config.json`: `4df6c7fd0b5811df3291c36dc0fe9104ee40739a739534ebc9d4b1d86606200c`
- `gguf-metadata.json`: `c0ea9f21b87a474da91d7d61e952417bc13cb4b826409645f69ba7f024e6bfff`

Canonical text geometry is hidden/vocab `2560/248320`, QSA `Q24/KV2/D256`, GDN `K16/V48/D128`, routed/shared expert width `640`, HC count `4`, indexer `4/1`, and MTP FC hidden width `2560`. The previous `model-3x3` inverse values are rejected by the fixture.

## TP3 hard guards and candidate dimensions

- QSA's divisibility logic rejects `KV2` at TP3 (`3 % 2 != 0`). Keep Q24 and expand packed KV rows using source map `[0,0,0,1,1,1]`. Contiguous TP shards then own source KVs `[0,0]`, `[0,1]`, `[1,1]`; the original query-to-source-KV mapping remains 12 queries per source KV.
- GDN K16 cannot shard evenly over TP3. Preserve the source V/K ratio of 3: use candidate geometry K18/V54, which gives 6 K heads and 18 V heads per rank. Combined Q/K/V projection rows grow from `10240` to `11520`. No GDN payload permutation is implemented here; the old inverse adapter must not be applied to canonical GGUF.
- FusedMoE asserts expert width divisibility by TP. Width640 fails TP3. The minimum width satisfying TP3 and source down-projection block sizes Q2_0=64 and IQ4_NL/Q8_0=32 is `768`, not `672`. Gate/up append 128 all-zero output rows; down appends 128 zero input values as complete encoded blocks. Existing packed source bytes stay byte-identical.
- Default vocabulary padding64 leaves 248320 unchanged, which is not divisible by3. Candidate padding quantum is `lcm(64,3)=192`, yielding physical vocab size248448 while logical vocab remains248320.
- MTP FC projections keep full `2560 x 2560` math. Under the opt-in marker they use `ReplicatedLinear`; otherwise the existing `ColumnParallelLinear` path is unchanged. QSA indexer4-head/1-KV projections already use replication, so their geometry needs no TP3 change.
- The existing TP4/QPN8 head gate remains untouched.

## Config view and constructor wiring

`make_ac922_tp3_config_view(text_config, tensor_parallel_size=..., extra_config=...)` is the loader hook API. Only boolean `extra_config["ac922_tp3"] == true` with TP3 creates a copy. Missing/false returns the same object. The returned view overrides QSA KV heads to6, GDN K/V heads to18/54, and `intermediate_size`, `moe_intermediate_size`, and `shared_expert_intermediate_size` to768. Logical `vocab_size=248320` and `hidden_size=2560` stay unchanged.

The view carries `_ac922_tp3_enabled`, `_ac922_tp3_original_config` (canonical source geometry plus MTP layer-count metadata), and `_ac922_tp3_vocab_padding_size=192`. `model.py` target embeddings/head and `mtp.py` draft embeddings/head consume `ac922_tp3_vocab_padding_kwargs(config)`. MTP FCs switch to replicated linears only when the marker exists. Default behavior remains unchanged until the loader installs a view.

The loader should preserve its original HF config for canonical GGUF restoration, deep-copy the HF root config, install the TP3 view into the root `text_config` (or root for text-only), and set cached `model_config.hf_text_config` to that same object. The GGUF adapter must continue using the original config for canonical transforms; the post-transform structural padder should use the view's source marker. The same view must also be applied to the separately constructed native MTP `draft_model_config`, if it is not the same object as the target config. PLE geometry remains unchanged and must not receive QSA/MoE padding.

## PP2 MTP mapping check

The HF checkpoint MTP path can be used on PP2's last rank: `Qwen4ExpMultiTokenPredictor` builds the complete stage-local draft layers on each PP rank, and `Qwen4ExpMTP.load_weights` remaps original `mtp.layers.0.*` names independently of target PP layer slicing. The draft LM head is only materialized on the last rank.

The native GGUF path still needs loader-owner confirmation. The source GGUF contains `blk.48.*`; the shared Qwen35 adapter intentionally skips nextn blocks from the target map, while this tree has no separate registered Qwen4Exp MTP GGUF adapter. The HF mapping result must not be treated as proof that GGUF MTP weights reach the draft loader.

## Files and verification

- `vllm/models/qwen4_exp/nvidia/ac922_tp3.py`: strict canonical geometry, opt-in config view/markers, vocab padding hook, and CPU packed-row/padding helpers.
- `vllm/models/qwen4_exp/nvidia/model.py`: target embedding and LM head consume the opt-in vocab padding marker.
- `vllm/models/qwen4_exp/nvidia/mtp.py`: draft embedding/head consume the marker; FC projections use ReplicatedLinear only under the marker.
- `tests/models/qwen4_exp/test_ac922_tp3_geometry_cpu.py` and `fixtures/ac922_tp3_geometry.json`: config gate/copy behavior, source marker retention, QSA row ownership/query mapping, 640-to-768 packed payload layout, and Q2_0/IQ4_NL/Q8_0 block alignment.

Verification commands:

- `.venv/bin/python tests/models/qwen4_exp/test_ac922_tp3_geometry_cpu.py` — 11 CPU tests passed.
- `.venv/bin/python -m py_compile vllm/models/qwen4_exp/nvidia/ac922_tp3.py vllm/models/qwen4_exp/nvidia/model.py vllm/models/qwen4_exp/nvidia/mtp.py tests/models/qwen4_exp/test_ac922_tp3_geometry_cpu.py` — passed.
- Ruff is unavailable in this project virtualenv. The fixture checks byte preservation and zero byte blocks; GGUF dequantizer-level proof, full loader integration, and model parity remain pending.
