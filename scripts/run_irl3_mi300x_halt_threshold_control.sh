#!/usr/bin/env bash
set -euo pipefail

artifact_root="qualification_artifacts/irl3_mechanism/deep_compute/evaluation/halt_threshold_control"
checkpoint="qualification_artifacts/irl3_mechanism/deep_compute/portable"
validation="data/qualification/validation"
session_start="$(date +%s)"

mkdir -p "${artifact_root}"
git rev-parse HEAD > "${artifact_root}/source_revision.txt"
git status --short > "${artifact_root}/source_status.txt"
cp "$0" "${artifact_root}/run_command.sh"

for threshold in 0.95 0.9999 0.99999 0.999999 0.99999999; do
  label="${threshold//./_}"
  timeout --foreground 600s python3 scripts/benchmark_recurrent_depth.py \
    --architecture irl3 \
    --checkpoint "${checkpoint}" \
    --data "${validation}" \
    --depths 8 \
    --sequence-length 4096 \
    --batch-size 1 \
    --batches 64 \
    --repeats 1 \
    --precision bf16 \
    --halt-mode adaptive \
    --halt-threshold "${threshold}" \
    --output "${artifact_root}/threshold_${label}.json"
done

python3 scripts/summarize_irl3_mechanism_study.py \
  --baseline-fixed \
    qualification_artifacts/irl3_mechanism/baseline_1000/depth_state_sweep.json \
  --baseline-control-fixed \
    qualification_artifacts/irl3/followup_irl3_fixed_depth.json \
  --baseline-writeback-off \
    qualification_artifacts/irl3/followup_irl3_writeback_off.json \
  --baseline-act qualification_artifacts/irl3/followup_irl3_act_depth.json \
  --baseline-adaptive \
    qualification_artifacts/irl3/followup_irl3_adaptive_depth.json \
  --deep-fixed \
    qualification_artifacts/irl3_mechanism/deep_compute/evaluation/fixed_depth_state_sweep.json \
  --deep-metrics \
    qualification_artifacts/irl3_mechanism/deep_compute/metrics.jsonl \
  --deep-writeback-off \
    qualification_artifacts/irl3_mechanism/deep_compute/evaluation/writeback_off_depth6.json \
  --deep-act \
    qualification_artifacts/irl3_mechanism/deep_compute/evaluation/act_depth_sweep.json \
  --deep-adaptive \
    qualification_artifacts/irl3_mechanism/deep_compute/evaluation/adaptive_depth_sweep.json \
  --halt-threshold-directory "${artifact_root}" \
  --starting-step 1000 \
  --json-output qualification_artifacts/irl3_mechanism/comparison.json \
  --markdown-output qualification_artifacts/irl3_mechanism/REPORT.md

echo "$(( $(date +%s) - session_start ))" \
  > "${artifact_root}/elapsed_seconds.txt"
