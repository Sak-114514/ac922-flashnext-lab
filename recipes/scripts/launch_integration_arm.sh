#!/usr/bin/env bash
# Integration arm launcher: ALL landed single-path switches ON (user: "直接能快多少加多少").
# $1 = arm name (for log naming). Adds GGML_SM70_MOE_REWORK=1 on top of the tree baseline
# (GGML_SM70_BF16_GEMV defaults ON). Extends launch_moerework_arm.sh unchanged otherwise.
set -euo pipefail
ROOT=/home/debian/ac922-llama-next
SCRIPTS=$ROOT/scripts
ARM=$1
systemd-run --user --unit=ac922-flashnext-test --collect \
  --property=WorkingDirectory=$ROOT \
  --property=TimeoutStopSec=90 --property=KillSignal=SIGINT --property=RuntimeMaxSec=14400 \
  --property=StandardOutput=append:$ROOT/logs/integration-$ARM.log \
  --property=StandardError=append:$ROOT/logs/integration-$ARM.log \
  --setenv=AC922_TEST_PLE_SHARD=0 --setenv=AC922_TEST_MTP=1 --setenv=AC922_TEST_MTP_DEPTH=4 \
  --setenv=AC922_TEST_MTP_PAD=6 --setenv=AC922_TEST_MTP_UBATCH=32 --setenv=AC922_TEST_CUDA_GRAPHS=1 \
  --setenv=AC922_TEST_BATCH=2048 --setenv=AC922_TEST_UBATCH=512 --setenv=AC922_TEST_CTX=132096 \
  --setenv=AC922_TEST_SLOTS=1 --setenv=AC922_TEST_GPU_ORDER=0,4,2,1,3,5 --setenv=AC922_TEST_HC_FUSED=1 \
  --setenv=AC922_TEST_HTTP_THREADS=8 --setenv=AC922_TEST_NCCL_AUTO=1 --setenv=AC922_TEST_NUMA0=1 \
  --setenv=LLAMA_MTP_FOLD=1 --setenv=LLAMA_GPU_ARGMAX=1 --setenv=GGML_CUDA_VOLTA_GDN_X4=1 \
  --setenv=GGML_CUDA_VOLTA_Q8_HEAD_TC=0 --setenv=LLAMA_QSA_SPARSE_KERNEL=1 --setenv=LLAMA_QSA_GATHER_MAX_TOKENS=32 \
  --setenv=LLAMA_ISLAND_SPLIT=3,3 \
  --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID --setenv=GGML_CUDA_MOE_GROUPED=1 --setenv=LLAMA_DRAFT_DETP=0 \
  --setenv=GGML_CUDA_FATTN_ONECAT=0 --setenv=LLAMA_PLE_ONECAT=0 --setenv=LLAMA_MONOMOE_SM70=0 \
  --setenv=AC922_TEST_P2P_AR=0 --setenv=GGML_META_ASYNC_COPY=1 --setenv=GGML_META_FIFO_PIPELINE=1 \
  --setenv=LLAMA_PREFILL_KV_PAD=8192 \
  --setenv=GGML_SM70_BF16_GEMV=1 --setenv=GGML_SM70_MOE_REWORK=1 \
  /bin/bash $SCRIPTS/run_test_server.sh
