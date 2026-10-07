# RowParallelLinear Q2_0 TP3 CPU fixture

Status: **passed**. This executes the AST-extracted body of the current
`RowParallelLinear.weight_loader` with real Torch `UninitializedParameter`
instances, using three rows sampled from the real GGUF Q2_0 down-expert tensor.
It exercises the existing `quant_size(42) -> (64, 18)` helper and the raw-byte
TP split. It does not initialize the full vLLM model or service.

## Input and assertions

The read-only GGUF sample is `blk.1.ffn_down_exps.weight`, expert 0, with
source shape `[640, 2560, 512]`. Three real rows contribute 540 bytes at 180
bytes per row. The fixture appends 36 zero bytes per row, forming a packed
`K=768` tensor with 216 bytes per row. With TP=3 and local `K=256`, the actual
loader materializes a `[3, 72]` uint8 parameter for each rank.

For rank 0, 1, and 2, the loader output exactly matches byte ranges `[0,72)`,
`[72,144)`, and `[144,216)` from every global row. Concatenating the three rank
outputs reconstructs the padded global tensor exactly; the real 180-byte
source prefix is unchanged and the 36-byte padded tail stays zero for each
row. Header plus selected payload read was 11,026,908 bytes, below the 128 MiB
fixture cap. CUDA was hidden and `torch.cuda.is_initialized()` remained false.

The tested source snapshot is recorded in
[rowparallel-q2-0-weight-loader-cpu.json](rowparallel-q2-0-weight-loader-cpu.json):
`linear.py` SHA-256 `c157db7851fec3575d4f45718b60f2c05b5d2b02a09c138714039de4fcf4b345`.

## Reproduction

Run from the local workspace; stdout is saved as the result JSON. The Torch
Python package is supplied by the existing AC922 runtime path, with no build:

```sh
ssh ac922 'CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LD_LIBRARY_PATH=/home/debian/ac922-vllm/runtime-525-20260928/extra-libs PYTHONPATH=/home/debian/ac922-vllm/runtime-tp3-mtp-20260929/site-packages:/home/debian/ac922-vllm/experiments/upstream-six-20261007/source:/home/debian/ac922-llama-next/experiment/codex-five-micro-20261006/05-dwdp/fixtures /home/debian/ac922-vllm/experiments/upstream-six-20261007/source/.venv/bin/python -B -' < ~/Desktop/ssh/reports/onecat-six-run-20261007/fixtures/rowparallel_q2_0_weight_loader_cpu.py > ~/Desktop/ssh/reports/onecat-six-run-20261007/rowparallel-q2-0-weight-loader-cpu.json
```

The fixture source is
[rowparallel_q2_0_weight_loader_cpu.py](fixtures/rowparallel_q2_0_weight_loader_cpu.py).
This is loader-body and raw-byte correctness evidence only; it does not cover
full model startup or GPU execution.
