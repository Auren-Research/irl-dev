#!/usr/bin/env bash
set -euo pipefail

artifact_root="qualification_artifacts/irl3_mechanism/deep_compute/evaluation"
checkpoint="qualification_artifacts/irl3_mechanism/deep_compute/portable"
validation="data/qualification/validation"
session_start="$(date +%s)"

mkdir -p "${artifact_root}"
git rev-parse HEAD > "${artifact_root}/source_revision.txt"
git status --short > "${artifact_root}/source_status.txt"
cp "$0" "${artifact_root}/run_command.sh"
find "${checkpoint}" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${artifact_root}/checkpoint_sha256.txt"
echo "[deep-compute-evaluation] started $(date --iso-8601=seconds)"
rocm-smi --showuse --showmeminfo vram --showmemuse --json \
  > "${artifact_root}/rocm_before.json"

# High-statistics fixed-budget sweep, paired against T=2. This uses the same
# validation range and state-comparison sample as the immutable step-1000 run.
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
  --output "${artifact_root}/fixed_depth_state_sweep.json"

# Same-checkpoint recurrent write-back control at the trained compute budget.
timeout --foreground 600s python3 scripts/benchmark_recurrent_depth.py \
  --architecture irl3 \
  --checkpoint "${checkpoint}" \
  --data "${validation}" \
  --depths 6 \
  --sequence-length 4096 \
  --batch-size 1 \
  --batches 200 \
  --repeats 1 \
  --precision bf16 \
  --halt-mode fixed \
  --disable-writeback \
  --output "${artifact_root}/writeback_off_depth6.json"

# The halt head received no direct objective during this continuation. These
# runs diagnose whether useful later computation, if any, transfers to the
# existing ACT-soft and adaptive-hard policies without claiming retraining.
for halt_mode in act adaptive; do
  timeout --foreground 900s python3 scripts/benchmark_recurrent_depth.py \
    --architecture irl3 \
    --checkpoint "${checkpoint}" \
    --data "${validation}" \
    --depths 4,8,12,32 \
    --sequence-length 4096 \
    --batch-size 1 \
    --batches 64 \
    --repeats 1 \
    --precision bf16 \
    --halt-mode "${halt_mode}" \
    --output "${artifact_root}/${halt_mode}_depth_sweep.json"
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
  --deep-fixed "${artifact_root}/fixed_depth_state_sweep.json" \
  --deep-metrics \
    qualification_artifacts/irl3_mechanism/deep_compute/metrics.jsonl \
  --deep-writeback-off "${artifact_root}/writeback_off_depth6.json" \
  --deep-act "${artifact_root}/act_depth_sweep.json" \
  --deep-adaptive "${artifact_root}/adaptive_depth_sweep.json" \
  --starting-step 1000 \
  --json-output qualification_artifacts/irl3_mechanism/comparison.json \
  --markdown-output qualification_artifacts/irl3_mechanism/REPORT.md

rocm-smi --showuse --showmeminfo vram --showmemuse --json \
  > "${artifact_root}/rocm_after.json"
echo "$(( $(date +%s) - session_start ))" \
  > "${artifact_root}/elapsed_seconds.txt"
echo "[deep-compute-evaluation] completed $(date --iso-8601=seconds)"
