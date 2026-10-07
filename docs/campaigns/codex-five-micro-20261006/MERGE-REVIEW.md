# Concurrent optimization merge review — 2026-10-06

## Ownership and GPU handoff

This lane writes only `experiment/codex-five-micro-20261006` and its local mirror. The other agent's live `sources/llama-v100-v3`, normal units and `experiment/zcode-fattn-stream-20261006` were read-only here. All leader-owned transient GPU runs have stopped. No new full-model queue is scheduled.

A stable six-file source snapshot is retained under `integration/external-snapshot-20261006-233203`. It is evidence of that timestamp, not permission to overwrite later edits. The live source differs from the frozen winner in two engine files and one test file: `src/llama-context.cpp`, `src/models/qwen4exp.cpp`, `tests/test-backend-ops.cpp`; compared current tracked and untracked source files did not reveal additional engine differences.

## What can be combined

| Item | Status / action |
|---|---|
| NextN flag-change reserve + fused four-head indexer | Already validated in frozen V1. `integration/merge-packet/tested-upstream-backports.patch` restores these two source differences against the captured external snapshot. Dry-run passed; **not applied to live source**. |
| SM70 indexer tile | CUDA source already matches frozen winner at snapshot. Check actual rebuilt object/library provenance before assuming a live binary contains the tile. |
| UUID topology and P2P / scalar GPU-chain settings | Reusable configuration, not a tree merge. Use the tested UUID order and inspect actual activation markers. The other inspected profiles use numeric `0,4,2,1,3,5` under `PCI_BUS_ID`, which mixes the measured islands, and set P2P AR off. Their measurements cannot be directly ranked against the proper-island/P2P winner. |
| Same-UID graph validation | Opt-in CUDA patch has a passing real graph-pointer fixture. Useful for future double-frame graph ownership; it does not implement that pipeline. |
| PLE batch dedup | Private CPU DSO and integrated 4-thread fixture pass. Keep candidate separate until the full model performance/parity gate is settled. |
| Native Q8K/Q4V vector | Two CUDA TUs compiled, combined with the UID object, and actual CUDA fixture reports `route=vec-q8q4 vector_pair_compiled=1`; previous F16 conversion removed. Sparse QSA still rejects Q4V. No whole-model performance or quality claim. |
| Multi-sequence hidden carry | Actual activation observed, but 2/4-stream microtests are slower and parity fails. **Do not enable or merge as a production default.** Full token-chain and sparse active-sequence mapping remain unimplemented. |
| Island DWDP | Real expert transfer + computation bitwise exact for IQ2_XXS/Q2_0. Loader placement / all seven formats / full model not implemented. |

## Other agent's MWE evidence

Read `manifests/zcode-mwe-{off,on}-al-r{1,2}`. All stored on/off 512-token comparisons report matched parity. Averaged recorded 64K prefill is about +0.39%, decode about -1.26%; 8K has effectively flat prefill and about -1.2% decode. These are the recorded profile differences, not established MWE-kernel gains.

At inspection, `LLAMA_MWE_ATTN` was absent from the current C/C++/CUDA source; the launch wrapper does not map it; current `libllama.so` / `libggml-cuda.so` strings and inspected on logs contain no MWE activation. The original run's exact mapped library hashes are needed to establish whether it ever activated. Treat the switch as unconfirmed, not a new accepted technology.

## Correctness gate discovered here

Same-input real-text multistream outputs are not strictly repeatable even in frozen V1: 64-token repeats first differ at 60/26 (two flows) and 12/1/36/no difference (four flows). The new carry candidate also fails parity and shows no improvement; it remains off. These data do not isolate the divergence to any new patch. Investigate multistream scheduling/state/numerics before a production promotion.

No modifications were made to the other agent's source, scripts, services, or logs. The patch packet and opt-in candidates are reviewable artifacts only.
