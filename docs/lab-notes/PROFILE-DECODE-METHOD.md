# Bounded Nsight Systems prefill diagnostic

`profile_decode.py` is a read-only controller for one already-running AC922
profile. It does not start `llama-server`, submit HTTP requests, stop services,
or change benchmark data.

The helper snapshots the current log size, ignores all old lines, waits for a
new unique `case_start` marker, runs `nsys start --session`, then stops at a
new `case_complete` marker. The default markers profile the complete 1 x 8K
request. For a true prompt-end marker supplied by the parent launcher, pass a
more specific `--stop-pattern`; otherwise use the whole-request trace and
filter the exported timeline by the prompt interval. The helper defaults to
the `prefill` label and records that the trace is diagnostic.

## AC922 launch prefix

Use the current MTP4 optimized model and existing wrapper environment. Replace
the output directory and preserve the parent-owned arguments after the binary:

```sh
SESSION=flashnext-prefill-8k-$(date +%s)
PROFILE_DIR=/home/debian/ac922-llama-next/profiles/$SESSION
mkdir -p "$PROFILE_DIR"
cd "$PROFILE_DIR"

/usr/local/cuda/bin/nsys launch \
  --session-new="$SESSION" \
  --trace=cuda,nvtx,osrt --cuda-graph-trace=node \
  --cuda-memory-usage=false -- \
  /usr/bin/numactl --cpunodebind=0 --membind=0 \
  /usr/local/bin/with-ac922-cuda525 \
  /home/debian/ac922-llama-next/sources/llama-v100-v3/build-ac922/bin/llama-server \
  --model /home/debian/models/qwen3.8-flash-next-gsq-rco/integrated/IQ3_XXS-MTP-Q8.gguf \
  --host 127.0.0.1 --port 8002 --split-mode tensor \
  --tensor-split 1,1,1,1,1,1 --n-gpu-layers 999 --fit off \
  --ctx-size 132096 --parallel 1 --kv-unified \
  --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on \
  --cache-type-k-draft q8_0 --cache-type-v-draft q8_0 \
  --batch-size 2048 --ubatch-size 512 \
  --spec-type draft-mtp --spec-draft-n-max 4
```

The parent launcher owns the remaining environment flags, including GPU order,
graph settings, NCCL settings, MTP folding, GDN x4, and the optional fair
prefill setting. Do not add MTP6 for this diagnostic.

Start the helper after the model is healthy and before the optimized 8K POST:

```sh
python3 profile_decode.py \
  --log /home/debian/ac922-llama-next/logs/profile-8k.log \
  --session "$SESSION" --output-prefix "$PROFILE_DIR/analysis" \
  --start-pattern 'case_start.*c1-p8192\.json' \
  --stop-pattern 'case_complete.*c1-p8192\.json' \
  --phase prefill --timeout 1800
```

After stop, the helper exports SQLite and writes:

- `*.kernels.csv`: `cuda_gpu_kern_sum:base` for MoE, NCCL, GDN, and attention names;
- `*.gpu_memory.csv`: GPU memory activity;
- `*.cuda_api.csv`: CUDA API summaries including allocation, graph, and sync calls;
- `*.kernel-categories.txt`: name-based category slices for quick inspection.

Nsight CUDA graph node tracing can add substantial synchronization and tracing
overhead. Use the report to attribute prefill cost and distinguish CUDA API
setup from GPU kernels. Do not use profiled TPS as the benchmark headline.

On this AC922 Nsight Systems 2023.4.4 build, `nsys launch` does not accept
`--sample` or `--cpuctxsw`; those controls belong on the later `nsys start`
command. Passing them to `launch` makes the command exit before the model
starts. The helper supplies them at start time. The raw report is
`reportN.nsys-rep` in the profiler working directory; the helper selects the
newest report created after its log watch began, and `--report-file` can pin an
exact path when several profiles share a directory.
