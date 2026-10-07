#!/usr/bin/env bash
set -euo pipefail

root="${1:-/home/debian/ac922-llama-next/sources/llama-v100-v3}"
cmake -S "$root" -B "$root/build-ac922" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DGGML_CUDA=ON -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_CUDA_NCCL=ON \
  -DNCCL_INCLUDE_DIR=/usr/local/nccl/include \
  -DNCCL_LIBRARY=/usr/local/nccl/lib/libnccl.so.2
cmake --build "$root/build-ac922" --parallel 6 \
  --target llama-server llama-cli llama-bench
