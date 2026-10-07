# super-ac922 upstream source synchronization

## Completed source operation

- Original clean fork: `opensource/super-ac922-clone`, branch `ac922-ppc64le`, HEAD `747db5fb482f9c4388aa3fa59399b899dce9c520`.
- Upstream snapshot: `1CatAI/1Cat-vLLM:main` at `bc01f2899db721a99788f3bc2abc041e4b9a7fd0` (includes #1000/#1001). Latest published tag queried: v1.5.1.
- Isolated detached worktree: `~/Desktop/ssh/opensource/super-ac922-upstream-sync-20261007`.
- `git merge --no-commit --no-ff <upstream>` succeeded without conflicts. The merge is staged and **uncommitted**; HEAD still identifies the original fork, MERGE_HEAD identifies upstream. Original branch and remote runtime trees remain untouched.
- Merged source tree ID: `2c950c452865509b47177694a344f9c43132d2a0`.
- Backup: `fork-before-sync.mbox`; review/recovery patch: `upstream-merge.patch`.
- Roll back only the isolated source operation with `git -C ~/Desktop/ssh/opensource/super-ac922-upstream-sync-20261007 merge --abort`.

GitHub compare reports 1,408 added commits from the shared base `db292f9a4`; main first-parent history has 199 added entries. The original local repository is shallow, making its full-DAG 19,857 count unsuitable as a PR count. The upstream diff covers 1,021 files / ~191,723 insertions. PR-number difference is not a backlog count or a schedule estimate.

## Verification performed

- No unresolved merge conflicts.
- 683 changed Python files parse successfully, including Python 3.11 grammar.
- All 99 tracked `ac922/` files retain exact original bytes.
- Executed the actual upstream pure ring-admission function against all 15 four-GPU subsets of the documented 3+3 adjacency: no supported four-rank ring. Positive synthetic four-rank ring admitted; three-rank shape rejected.
- Executed actual pinned-PLE admission with current host figures: TP3 rejected; TP4 rejected on fair host capacity. Current `/proc/meminfo` is 31.5 GiB CPU memory, not aggregate GPU HBM; default host reserve is one quarter of CPU memory.
- Executed actual setup requirement parser under a mocked CUDA12.4/SM70 build policy: retains Torch2.10.0 and does not invoke the x86/ARM CUDA12.8-wheel URL helper. This is a packaging-policy check, not compilation.
- Full `git diff --cached --check` reports existing whitespace in newly synchronized upstream vendored sources/patches. Those lines were not rewritten for cosmetic cleanup.

**Not performed:** dependency resolution, POWER9 native extension/wheel rebuild, imports of a newly compiled extension, GPU fixtures, model parity/performance, deployment, publishing. Source synchronization is complete; runtime revival is not yet validated. No hardware or normal service changes were made.

## PR status and activation

| PR | Current main | AC922 relevance |
|---|---|---|
| #927/#928/#929/#936 | merged | GDN/MTP layout/copy improvements; qualified shapes may change under TP3 padding. |
| #931/#933 | merged | Canonical GGUF experts and measured down-vector dispatch. Prefer these over duplicate old loader/kernel overrides. |
| #930/#934 | closed, unmerged | Negative screens; do not resurrect as active defaults. |
| #932 | open draft | Candidate benchmarks; production dispatch unchanged. |
| #935 | merged | Particular TP4/FP16/draft7 DFlash2 path, measured on 27B NVFP4. Not Flash-Next evidence. |
| #957 | merged | TP4 full mesh, eight rows, FP32 residual; ~0.065ms measured single-request benefit. No TP3 fast activation. |
| #992 | merged | TP4 rank XOR/forwarding geometry and qualified direct NVLink ring. Three-rank admission is intentionally false. |
| #998 | merged | Rank-local QPN Q/K/V at M8/K5120/N3584; operator evidence, not established model uplift. |
| #962/#967/#988/#996/#1000 | merged | New GGUF/GDN layout, PLE, n-gram preparation, pinned row and expert paths; check their source/table/topology qualifications. |

See `communication.md`, `speculation.md`, `ple-layout.md` and their JSON evidence for exact linked source locations.

## What our own work should contribute

### P0 — make the existing model correct on TP3 x PP2

Use the upstream canonical GGUF/PLE model and seven-format expert implementation. Rebase only structural AC922 adapters: TP3 head/channel/width padding, PP-local filtering before CPU copies, staged island loading, physical host-memory/cgroup limits, and last-stage native MTP head ownership. Avoid copying the old model/loader overlay wholesale: its best stored 3+3 output first diverged from the oracle at token 5. Avoid stacking the old inverse norm/A/V transformations on a loader which now handles its own layout restoration.

Keep actual decode CUDA Graphs. Determine the target verify/draft shapes before minimizing capture sizes: MTP4 needs verification shapes such as M5/M20, not merely a nominal [1,2] decode set. Flash-Next initially uses its matching native MTP; the measured 27B DFlash2 acceptance is not transferable.

### P1 — TP3 island-local communication

This is the clearest missing topology. The generic `CustomAllreduce._SUPPORTED_WORLD_SIZES` is `[2,4,6,8]`, so TP3 falls back to NCCL. Port the proven three-peer P2P reduction idea into a communicator-owned three-rank route with persistent buffers, graph lifetime, generations and skew tests. Do not drop the #992 world-size guard: XOR rank neighbors and four-way shard geometry do not become correct at size three. Cross-island traffic belongs to the PP bridge, not a six-rank NVLink ring.

### P1 — bounded encoded PLE row access

Current host memory cannot accommodate the default pinned table plus reserve; TP3/PP2 also does not qualify for the current pinned local-TP4 route. Reuse mmap encoded rows and bounded batching/cache (the latest CPU fixture caps cache+scratch at ~60MiB), preserving n-gram/head offsets and encoded row geometry. Port through the vLLM CPU-offload/PLE interfaces; the GGML GET_ROWS callback itself is not an interchangeable vLLM operator.

### P2 — match new projection kernels to TP3 geometry

Use #998/#993/#999 capability/packing/ledger structure, then qualify the actual TP3 Q/K/V, GDN and expert dimensions. Retain fallback for unmeasured rows and formats. Our SM70 indexer tile and compact scratch ideas are useful only where the new attention/indexer path still has the same work; a llama.cpp CUDA source file is not a direct vLLM backend replacement.

### Keep out of defaults

The failed multistream carry prototype, unqualified GPU-PLE 132K placement and the +2.2% MonoMoE candidate are not wins to transplant. vLLM already has per-request GPU token/hidden state and step graph replay; inspect what remains missing before duplicating that controller. The llama.cpp same-UID pointer fix does not identify a matching PyTorch CUDAGraph bug.

## Build/runtime boundaries

The server has CMake3.31.6, GCC12.2 and CUDA12.4. Saved sites contain both Torch2.10.0 and a Torch2.12.0a1 MRV2 runtime. Select and record one native ABI; do not mix wheel headers, libtorch, Triton and `_C` from these sites. Start from the available Torch2.10/CUDA12.4 build site rather than installing x86 CUDA12.8 wheels.

The preserved historical AC922 build script pins `db292f9a4` and assumes an old RAM image; it is a historical reproduction recipe, not a new-head builder. A revival builder must take explicit source/tree identity and isolated output paths, preserve the existing POWER9 wheel mirrors, and avoid mutating either live runtime. Build availability does not prove kernel/model activation.

The source merge took minutes. Native rebuild, correct 3+3 integration and acceptance gates determine the remaining effort; 1–2 weeks is plausible as a planning budget, not a measured completion forecast.

## Confirmed six-card target

The requested model stays Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS with the existing Q8 MTP head and all 512 experts/PLE, running six 16GB V100s as TP3×PP2. The original GGUF remains read-only. This is the Flash-Next line, not the successful 27B dense service.

The actual checkpoint has only `ple_layer_ids=[2]`. The current upstream first-stage PLE guard passes this model under PP2 25+23; the report's earlier generalization that all PLE requires PP1 was corrected. The separate pinned decode feature still requires local TP4/PP1 and more host capacity.
