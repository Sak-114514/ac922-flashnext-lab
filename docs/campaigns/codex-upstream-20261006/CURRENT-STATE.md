# Codex upstream and execution optimization

2026-10-06: source and runtime snapshot complete; GPU tests not started. Scope includes prefill and decode; no production source writes.
2026-10-06 U1: PR30020 graph-shape flags now trigger sched reserve only on a value change; isolated stage snapshot created; build/runtime verification pending.
2026-10-06 bundle: PR29825 CUDA four-head support and PR29901 tile applied; model scoring uses opt-in fused op with original block/cell bias and unit head weights; compilation pending.
2026-10-06: seven-arm report received: stock layer TG 98.25/85.29 vs TP 92.93/83.07; own baseline synthetic PP/TG 648.86/93.67 at8K,738.80/83.02 at64K; real8K 173.73/53.57,real64K369.84/72.64. Baseline released. Candidate bundle will combine upstream memory+tile+NextN and test stock layer.
2026-10-06 U1/U2 fixtures: actual CPU NextN setter/reserve, shared-row/stale,partial seq_cp PASS; PR29994 standard kpool not used in this fork, no forced C++ patch. Argmax unused-logit drop already exists at top-level decode; no redundant optimization.
2026-10-06 U3 GPU fixture: 9/9 LIGHTNING_INDEXER cases PASS (4 heads,kv257,batch7/8/9,F32/F16/Q8_0),runtime vector and tile confirmed. CUDA .cu rebuilt alone from frozen objects; baseline runtime kept immutable.
2026-10-06 topology: old PCI_BUS_ID + visible0,4,2,1,3,5 gives mixed-SYS Meta groups; CUDA runtime peer matrix confirmed first3 not all peers. UUID order corrected to two NVLink3 groups (all intragroup peers true). Combined island+P2PAR+indexer+NextN test launched; no result claim yet.
2026-10-06 bundle bootstrap SIGSEGV: dynamic loader mixed experimental libggml-base with /usr/local old CPU/CUDA backends because new bin lacked backend libraries. Fixed with local libggml*.so symlinks to frozen/updated experiment prefix; no production or kernel source rollback.
2026-10-06 bundle-islands-fixed: 8K PP964.69/TG113.73 vs648.86/93.67, synthetic512 IDs exact; real8K PP210.99/TG62.22 but tokens differ. Not production-accepted; investigate baseline nondeterminism and numerical paths.
2026-10-06 baseline real8K repeat: unchanged frozen binary/config also differs from first real8K output; one-run realtext512 equality is not a stable reference. Keep strict synthetic512+operator fixture gates; report realtext acceptance/speed with this limitation.
2026-10-06 bundle-layer-u1024: PP863.51/TG101.04 at8K,PP880.88/TG87.28 at64K; synthetic512 parity both true. Real8K193.77/61.99, tokens differ. No production deployment.

- 2026-10-06 19:04 capacity combo: ctx262144 + UB1024 + q8K/q4V + fused indexer + true-island P2P failed during first 8K prefill (device3 workspace 4824.31MiB allocation OOM), no timing/parity result; owned unit stopped, all GPU memory released. Retry only UB512 with same bundle.

2026-10-06 bundle-cap262-u512: ctx262144,q8K/q4V,synthetic8K PP881.43/TG113.92 and64K PP987.49/TG99.20,512-token parity both true; minfree1783MiB before final real8K and peakGPU46C. Actual input tested64K, not262K.

2026-10-06 bundle-islands-u1024:132K q8K/q8V,8K913.12/114.21,64K1006.00/99.27; synthetic512 parity both true. Real8K181.40/78.28 but old TOPK_STABLE off, no stable real oracle.
2026-10-06 bundle-r19-u1024: added already-repaired GPU_CHAIN/GPU_EMBED/MTP_FIFO/STABLE_TOPK/compact arena/packed cache together;8K920.56/125.06,64K1003.40/108.81,synthetic512 exact. Frozen real8K cold183.83/71.29 and warm949.21/98.48,512 outputs identical,367/573 draft accepted both. Old baseline without stable topk not a valid exact-realtext oracle. GPU/host peak and minfree saved in telemetry; owned unit released.

2026-10-06 stage FIFO: default-off layer-prefill pipeline built; parent repaired atomic src/view data remapping, preserved legacy FIFO. Tiny cross-SYS two-target-stage Qwen4Exp CUDA Graph fixture passed8 full-logit rows,4-copy wraparound/reset127tail/output automatic drain, maxAbs=1.1640000004446358e-10. Not yet a speed result.

2026-10-06 stage combo:8K912.40/102.70,64K1176.37/88.46,synthetic512 parity; tiny8x128logits PASS1.164e-10. Real8K repeat diverges at37. Same layer/R19/fused/UB512 with stage off repeats512/512 and350/644acceptance. Failed full-model correctness gate; stage source/runtime archived under artifacts/rejected-stage-pipeline and canonical3files+bin restored to measured bundle-v1, all rollback hashes match. No production/hardware changes.
