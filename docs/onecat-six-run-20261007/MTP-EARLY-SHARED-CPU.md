# AC922 TP3 MTP early-share CPU gate

Status: PASS for the V2 sharing context and the MTP constructor branch. This
fixture does not instantiate a model or allocate CUDA tensors.

## What it checks

[`mtp_early_shared_context_cpu.py`](<~/Desktop/ssh/reports/onecat-six-run-20261007/fixtures/mtp_early_shared_context_cpu.py>) calls the actual
`vllm.v1.worker.gpu.spec_decode.eagle.utils.load_eagle_model` with tiny CPU
embedding/head modules and a fake `get_model` constructor. The PP group is
stubbed to world size 2 to exercise the AC922 branch without distributed or GPU
initialization. The fake constructor reads the real `ac922_tp3` ContextVar.

It verifies:

- Opt-in construction sees the exact target embedding and head objects.
- The returned draft keeps those same object identities on PP2.
- The ContextVar resets after both successful construction and a thrown
  constructor error.
- With the opt-in marker absent, the fake constructor sees no shared context,
  the draft embedding remains separate, and existing generic head sharing still
  occurs.
- AST checks on the installed `mtp.py` confirm that marked/context-present
  branches assign the shared embedding/head before the fallback
  `VocabParallelEmbedding` or `ParallelLMHead` constructor calls.

## Result

Remote command used the private runtime site and the source venv Python with
`CUDA_VISIBLE_DEVICES=''`, `PYTHONDONTWRITEBYTECODE=1`, and `python -B -`.
Result:

```text
PASS: actual V2 load_eagle_model CPU cases (PP2 pointer identity, success/error ContextVar reset, non-opt-in behavior), actual MTP constructor AST selects shared modules before VocabParallelEmbedding/ParallelLMHead allocation; CUDA initialized=False
```

The runtime printed expected “No CUDA runtime” and missing `vllm._C` import
warnings; this gate uses neither CUDA operations nor compiled model kernels.
It proves branch wiring and object identity, not the real model load, GPU peak
memory reduction, or inference parity.
