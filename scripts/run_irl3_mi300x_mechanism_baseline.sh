#!/usr/bin/env bash
set -euo pipefail

artifact_root="qualification_artifacts/irl3_mechanism/baseline_1000"
checkpoint="qualification_artifacts/irl3/portable_irl3_followup_extended"
validation="data/qualification/validation"

mkdir -p "${artifact_root}"
echo "[mechanism-baseline] started $(date --iso-8601=seconds)"
rocm-smi --showuse --showmeminfo vram --showmemuse --json \
  > "${artifact_root}/rocm_before.json"

timeout --foreground 1800s python3 scripts/benchmark_recurrent_depth.py \
  --architecture irl3 \
  --checkpoint "${checkpoint}" \
  --data "${validation}" \
  --depths 2,4,6,8,12,16,24,32 \
  --sequence-length 4096 \
  --batch-size 1 \
  --batches 200 \
  --repeats 1 \
  --precision bf16 \
  --halt-mode fixed \
  --state-reference-depth 2 \
  --state-comparison-batches 8 \
  --output "${artifact_root}/depth_state_sweep.json"

rocm-smi --showuse --showmeminfo vram --showmemuse --json \
  > "${artifact_root}/rocm_after.json"
echo "[mechanism-baseline] completed $(date --iso-8601=seconds)"
