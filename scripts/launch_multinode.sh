#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
#   NNODES, NODE_RANK, MASTER_ADDR, MASTER_PORT, GPUS_PER_NODE
: "${NNODES:?Set NNODES}"
: "${NODE_RANK:?Set NODE_RANK}"
: "${MASTER_ADDR:?Set MASTER_ADDR}"
: "${MASTER_PORT:?Set MASTER_PORT}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE}"

torchrun \
  --nnodes="${NNODES}" \
  --node-rank="${NODE_RANK}" \
  --nproc-per-node="${GPUS_PER_NODE}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  -m lunaris_dense_irl2.training.train \
  "$@"
