#!/usr/bin/env bash
set -euo pipefail

artifact_root="qualification_artifacts/irl3_mechanism/deep_compute"
starting_checkpoint="$(pwd)/qualification_artifacts/irl3/checkpoints/followup_extended/step_00001000"
profile="training_profiles/irl3_135m_1x_mi300x_deep_compute_continuation.yaml"
session_start="$(date +%s)"

mkdir -p "${artifact_root}"
test -f "${starting_checkpoint}/_SUCCESS"
cp "${profile}" "${artifact_root}/training_profile.yaml"
git rev-parse HEAD > "${artifact_root}/source_revision.txt"
git status --short > "${artifact_root}/source_status.txt"
git diff --binary > "${artifact_root}/code_at_experiment_start.diff"
find "${starting_checkpoint}" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "${artifact_root}/starting_checkpoint_sha256.txt"
printf '%s\n' "${starting_checkpoint}" \
  > "${artifact_root}/starting_checkpoint_path.txt"

echo "[deep-compute] started $(date --iso-8601=seconds)"
rocm-smi --showuse --showmeminfo vram --showmemuse --json \
  > "${artifact_root}/rocm_before.json"

python3 -m lunaris_dense_irl3.training.preflight \
  --config "${profile}" \
  --world-size 1 \
  --require-data \
  --output "${artifact_root}/preflight.json"

timeout --foreground 5400s python3 -m lunaris_dense_irl3.training.train \
  --config "${profile}" \
  --resume "${starting_checkpoint}"

final_checkpoint="${artifact_root}/checkpoints/step_00003000"
test -f "${final_checkpoint}/_SUCCESS"
python3 -m lunaris_dense_irl3.training.export_checkpoint \
  --checkpoint "${final_checkpoint}" \
  --output "${artifact_root}/portable"

rocm-smi --showuse --showmeminfo vram --showmemuse --json \
  > "${artifact_root}/rocm_after.json"
echo "$(( $(date +%s) - session_start ))" \
  > "${artifact_root}/elapsed_seconds.txt"
echo "[deep-compute] completed $(date --iso-8601=seconds)"
