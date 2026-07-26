#!/usr/bin/env bash
set -euo pipefail

# Recommended by AMD to improve compute/RCCL stream overlap on MI300X.
export TORCH_NCCL_HIGH_PRIORITY="${TORCH_NCCL_HIGH_PRIORITY:-1}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"

# These diagnostics are useful for the first cluster run and inexpensive.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-OFF}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  -m lunaris_dense_irl2.training.train \
  "$@"
