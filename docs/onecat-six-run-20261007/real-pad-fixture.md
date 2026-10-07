# Real GGUF TP3 row-copy and padding fixture

Status: CPU-only dequantization checks passed. No model, loader, service, GPU, CUDA build, or original GGUF write was performed.

## Input and bounded read

The fixture read `/home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf` in read-only mode. Its size is 78,615,622,080 bytes. It streams the GGUF header and uses `os.pread` for tensor slices; it does not mmap or decode the complete file.

It ran as one process using the AC922 onecat venv Python 3.11.2 and NumPy 1.24.2, with `CUDA_VISIBLE_DEVICES` empty and CPU thread limits set to one. The independent decode oracle calls `ggml_get_type_traits(type)->to_float` from the frozen `libggml-base.so.0.25.1` (SHA-256 `53b90e05de2af16c5a57d67b543041031213bc2e7444a6d5fbdf5e778e9b7c67`). The GGUF Python package supplies descriptor and block metadata; the roothelper NumPy pad functions are not used as the oracle.

The measured bound is 40,502,208 bytes of original GGUF header plus tensor payload, plus 76,032,000 bytes of padded raw representations constructed in memory, for 116,534,208 bytes total. That stays below the 128 MiB cap. The original tensor payload sample is 29,475,840 bytes.

## Formats present and CPU oracle coverage

The target's 144 routed expert tensors in blocks 0-47 use seven types: IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_NL, and Q2_0. The 24 QSA K/V matrices use IQ4_XS, Q4_K, Q5_K, Q6_K, and IQ3_S. The appended MTP block 48 has Q8_0 expert tensors and is inventoried separately because it has a separate loader contract.

The frozen CPU trait matched the GGUF block size and byte size and decoded a zero block to all-zero values for the full 12-type union. It therefore covers Q2_0's 64-element / 18-byte block, including its two added zero blocks in the down-projection K pad.

## Checks performed

- **QSA K/V row-copy:** all 24 real `attn_k.weight` and `attn_v.weight` tensors have GGML shape `[2560, 512]`. Each source has two 256-row KV heads. The fixture builds target rows with map `[0,0,0,1,1,1]`, yielding 1,536 rows. For every matrix, the copied raw rows match the source map exactly and the frozen GGML dequantized rows match exactly.
- **Gate/up output row pad:** one real expert from representative layers covers all five actual gate/up types. The fixture copies 640 raw rows and appends 128 all-zero rows to reach 768. It checks the original raw prefix byte-for-byte, padded raw bytes are zero, dequantized prefix is exact, and dequantized padded rows are zero.
- **Down input K pad:** one real expert for each actual down type (Q2_0 and IQ4_NL) expands K from 640 to 768. The fixture checks every row's original bytes, appends only complete encoded blocks, and verifies the dequantized first 640 values are exact and the 128 new values are zero.

The Q2_0 representative is `blk.1.ffn_down_exps.weight`, expert 0, with source row size 180 bytes and target row size 216 bytes. Its source prefix is 460,800 bytes, and each of 2,560 output rows gets 36 zero bytes (two Q2_0 blocks) for the pad.

The result JSON contains each tensor name, type, shape, row/block geometry, source/target SHA-256, and prefix/padding assertions: [real-pad-fixture.json](real-pad-fixture.json). The script itself is [real-pad-fixture.py](real-pad-fixture.py).

## Reproduction

From the local workspace, pipe the allowed fixture into the AC922 venv and save stdout only in this report directory:

```sh
ssh ac922 'CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=/home/debian/ac922-llama-next/experiment/codex-five-micro-20261006/05-dwdp/fixtures:/home/debian/ac922-llama-next/experiment/codex-five-micro-20261006/05-dwdp/source/gguf-py /home/debian/ac922-vllm/experiments/upstream-six-20261007/source/.venv/bin/python -B -' < ~/Desktop/ssh/reports/onecat-six-run-20261007/real-pad-fixture.py > ~/Desktop/ssh/reports/onecat-six-run-20261007/real-pad-fixture.json
```
