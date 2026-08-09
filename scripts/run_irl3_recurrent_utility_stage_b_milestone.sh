#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <100m|200m|350m|500m> <resume-checkpoint>" >&2
  exit 2
fi

label="$1"
resume_checkpoint="$(realpath "$2")"
artifact_root="${IRL3_UTILITY_ARTIFACT_ROOT:-qualification_artifacts/irl3_recurrent_utility}"
data_root="${IRL3_UTILITY_DATA_ROOT:-data/irl3_recurrent_utility}"
artifact_root="$(realpath -m "${artifact_root}")"
data_root="$(realpath -m "${data_root}")"

case "${label}" in
  100m) target_step=6104 ;;
  200m) target_step=12208 ;;
  350m) target_step=21363 ;;
  500m) target_step=30518 ;;
  *) echo "unknown milestone: ${label}" >&2; exit 2 ;;
esac

test -f "${resume_checkpoint}/_SUCCESS"
config="${artifact_root}/stage_b/config.yaml"
test -f "${config}"
root="${artifact_root}/stage_b/milestones/${label}"
checkpoint_root="${artifact_root}/stage_b/checkpoints"
printf -v step_name 'step_%08d' "${target_step}"
final_checkpoint="${checkpoint_root}/${step_name}"
portable="${root}/portable"
mkdir -p "${root}"

started="$(date +%s)"
if [[ ! -f "${final_checkpoint}/_SUCCESS" ]]; then
  echo "[stage-b] training to ${label}/${target_step} $(date --iso-8601=seconds)"
  timeout --foreground 18000s python3 -m lunaris_dense_irl3.training.train \
    --config "${config}" \
    --max-steps "${target_step}" \
    --resume "${resume_checkpoint}" \
    2>&1 | tee "${root}/training.log"
fi
echo "$(( $(date +%s) - started ))" > "${root}/training_elapsed_seconds.txt"

if [[ ! -f "${portable}/model.pt" ]]; then
  python3 -m lunaris_dense_irl3.training.export_checkpoint \
    --checkpoint "${final_checkpoint}" \
    --output "${portable}"
fi

depths="2,4,6,8,12,16"
if [[ "${label}" == "500m" ]]; then
  depths="${depths},24,32"
fi

for split in mechanism_dev confirmation_holdout; do
  output="${root}/${split}_depth_sweep.json"
  if [[ -f "${output}" ]]; then
    continue
  fi
  echo "[stage-b] evaluating ${label} ${split} $(date --iso-8601=seconds)"
  timeout --foreground 1800s python3 scripts/benchmark_recurrent_depth.py \
    --architecture irl3 \
    --checkpoint "${portable}" \
    --data "${data_root}/${split}" \
    --depths "${depths}" \
    --sequence-length 4096 \
    --batch-size 1 \
    --batches 200 \
    --repeats 1 \
    --precision bf16 \
    --halt-mode fixed \
    --state-reference-depth 2 \
    --state-comparison-batches 8 \
    --output "${output}"
done

python3 scripts/summarize_irl3_recurrent_utility.py \
  --artifact-root "${artifact_root}"
echo "[stage-b] ${label} complete $(date --iso-8601=seconds)"
