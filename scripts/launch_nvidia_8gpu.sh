#!/usr/bin/env bash
set -euo pipefail

export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  -m lunaris_dense_irl2.training.train \
  "$@"
