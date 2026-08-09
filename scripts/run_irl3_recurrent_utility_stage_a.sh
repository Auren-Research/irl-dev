#!/usr/bin/env bash
set -euo pipefail

artifact_root="${IRL3_UTILITY_ARTIFACT_ROOT:-qualification_artifacts/irl3_recurrent_utility}"
starting_checkpoint="${IRL3_UTILITY_STARTING_CHECKPOINT:-qualification_artifacts/irl3/checkpoints/followup_extended/step_00001000}"
positive_checkpoint="${IRL3_UTILITY_POSITIVE_CHECKPOINT:-qualification_artifacts/irl3_mechanism/deep_compute/checkpoints/step_00003000}"
data_root="${IRL3_UTILITY_DATA_ROOT:-data/irl3_recurrent_utility}"
source_shard="${IRL3_UTILITY_SOURCE_SHARD:-data/source/ultrafineweb/shard_0000.npy}"
tokenizer="${IRL3_UTILITY_TOKENIZER:-data/source/ultrafineweb/lunaris-tokenizer.json}"

repo_root="$(pwd)"
starting_checkpoint="$(realpath "${starting_checkpoint}")"
positive_checkpoint="$(realpath "${positive_checkpoint}")"
artifact_root="$(realpath -m "${artifact_root}")"
data_root="$(realpath -m "${data_root}")"

test -f "${starting_checkpoint}/_SUCCESS"
test -f "${positive_checkpoint}/_SUCCESS"
mkdir -p "${artifact_root}"

python3 -m scripts.prepare_irl3_recurrent_utility_data \
  --source "${source_shard}" \
  --tokenizer "${tokenizer}" \
  --output-root "${data_root}"

python3 scripts/generate_irl3_recurrent_utility_configs.py \
  --repo-root "${repo_root}" \
  --artifact-root "${artifact_root}" \
  --starting-checkpoint "${starting_checkpoint}" \
  --positive-checkpoint "${positive_checkpoint}" \
  --stage a

git rev-parse HEAD > "${artifact_root}/source_revision.txt"
git status --short > "${artifact_root}/source_status.txt"
rocm-smi --showuse --showmeminfo vram --showmemuse --json \
  > "${artifact_root}/rocm_before_stage_a.json"

train_condition() {
  local condition="$1"
  local root="${artifact_root}/stage_a/conditions/${condition}"
  local config="${root}/config.yaml"
  local final_checkpoint="${root}/checkpoints/step_00003000"
  local portable="${root}/portable"
  local started
  started="$(date +%s)"

  if [[ ! -f "${final_checkpoint}/_SUCCESS" ]]; then
    echo "[stage-a] training ${condition} $(date --iso-8601=seconds)"
    timeout --foreground 3600s python3 -m lunaris_dense_irl3.training.train \
      --config "${config}" \
      --resume "${starting_checkpoint}" \
      2>&1 | tee "${root}/training.log"
  fi
  echo "$(( $(date +%s) - started ))" > "${root}/training_elapsed_seconds.txt"

  if [[ ! -f "${portable}/model.pt" ]]; then
    python3 -m lunaris_dense_irl3.training.export_checkpoint \
      --checkpoint "${final_checkpoint}" \
      --output "${portable}"
  fi
}

for condition in control fixed_depth_only full_credit_only zero_ponder_only; do
  train_condition "${condition}"
done

positive_root="${artifact_root}/stage_a/conditions/positive_control"
if [[ ! -f "${positive_root}/portable/model.pt" ]]; then
  python3 -m lunaris_dense_irl3.training.export_checkpoint \
    --checkpoint "${positive_checkpoint}" \
    --output "${positive_root}/portable"
fi
printf '%s\n' "reused:${positive_checkpoint}" \
  > "${positive_root}/training_elapsed_seconds.txt"

evaluate_condition() {
  local condition="$1"
  local root="${artifact_root}/stage_a/conditions/${condition}"
  local portable="${root}/portable"
  local split

  for split in mechanism_dev confirmation_holdout; do
    local output="${root}/${split}_depth_sweep.json"
    if [[ -f "${output}" ]]; then
      continue
    fi
    echo "[stage-a] evaluating ${condition} ${split} $(date --iso-8601=seconds)"
    timeout --foreground 1200s python3 scripts/benchmark_recurrent_depth.py \
      --architecture irl3 \
      --checkpoint "${portable}" \
      --data "${data_root}/${split}" \
      --depths 2,4,6,8,12,16 \
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
}

for condition in control fixed_depth_only full_credit_only zero_ponder_only positive_control; do
  evaluate_condition "${condition}"
done

python3 scripts/summarize_irl3_recurrent_utility.py \
  --artifact-root "${artifact_root}"

rocm-smi --showuse --showmeminfo vram --showmemuse --json \
  > "${artifact_root}/rocm_after_stage_a.json"
echo "[stage-a] complete $(date --iso-8601=seconds)"
