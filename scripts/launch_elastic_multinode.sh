#!/usr/bin/env bash
set -euo pipefail

# Required environment variables:
#   MIN_NODES, MAX_NODES, GPUS_PER_NODE, RDZV_ENDPOINT, RUN_ID
#
# All nodes must see the same checkpoint directory and token shards. Elastic
# membership changes restart the workers from the latest complete checkpoint.
: "${MIN_NODES:?Set MIN_NODES}"
: "${MAX_NODES:?Set MAX_NODES}"
: "${GPUS_PER_NODE:?Set GPUS_PER_NODE}"
: "${RDZV_ENDPOINT:?Set RDZV_ENDPOINT as host:port}"
: "${RUN_ID:?Set a unique RUN_ID}"

MAX_RESTARTS_VALUE="${MAX_RESTARTS:-3}"
MONITOR_INTERVAL_VALUE="${MONITOR_INTERVAL:-5}"

torchrun \
  --nnodes="${MIN_NODES}:${MAX_NODES}" \
  --nproc-per-node="${GPUS_PER_NODE}" \
  --rdzv-id="${RUN_ID}" \
  --rdzv-backend="c10d" \
  --rdzv-endpoint="${RDZV_ENDPOINT}" \
  --max-restarts="${MAX_RESTARTS_VALUE}" \
  --monitor-interval="${MONITOR_INTERVAL_VALUE}" \
  -m lunaris_dense_irl2.training.train \
  "$@"
