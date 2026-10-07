# Prefill stage pipeline experiment

## Cleanup/refactor plan

- Keep the execution path opt-in with `GGML_META_PREFILL_STAGE_PIPELINE=1`; scope it to one-sequence, no-output, non-MTP target prefill using layer split and native scheduler copies.
- Reuse the scheduler's existing per-copy INPUT tensors. On graph reuse, rotate copy slots by remapping only split-node INPUT sources; do not rebuild graph contexts or move already-buffered KV/recurrent state.
- Keep the current last-output and graph-rebuild drain. Skip the reuse full-scheduler sync only after all graph INPUT leaves are host-backed and per-copy pinned snapshots/events have been allocated.
- Extend pinned snapshots from META to CUDA. Wait on the snapshot's H2D event only when its pinned host slot is reused. Preserve the zero-input split fence until allocator lifetime separation is demonstrated.

## State and fixture gate

State: narrow source patch is staged; CPU-only syntax compilation is pending; no GPU, inference, or service work from this lane. Upstream experimental source is a copied tree without `.git`; original target files are backed up under `backups/stage-pipeline/`. `CURRENT-STATE.md` is owned by the parent experiment and will not be edited here.

Before use, run the parent's existing stock-layer synthetic 8K / 512-token parity fixture with `GGML_META_PREFILL_STAGE_PIPELINE=0` and `=1`, then the fixed 64K prefill/decode comparison. Check the `prefill stage pipeline: frame=... copy=...` trace and input-frame values, four-slot rotation, same-sequence KV/GDN ordering, final-output drain, shape-rebuild drain, and absence of host-buffer reuse before H2D completion. Confirm pinned snapshots stay under the 512 MiB cap; a failed reservation must disable the path and release partial buffers. Reject or disable the path on any mismatch, changed 512-token output, or incomplete final synchronization.

No GPU test or service start is allowed from this implementation lane. Cross-SYS copies may still take the existing synchronized fallback when CUDA P2P async copy is unavailable.

## Final status

Rejected and rolled back: tiny GPU fixture passed, but full-model real8K repeat diverged at token37. Same layer configuration without the new stage path was repeatable. Active source/bin restored to bundle-v1; failed code/runtime archived under artifacts/rejected-stage-pipeline. See REPORT.md.
