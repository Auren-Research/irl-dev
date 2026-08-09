#!/usr/bin/env bash
set -euo pipefail

# Bounded, resumable single-MI300X IRL-3 architecture qualification.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
artifact_root="${project_root}/qualification_artifacts/irl3"
source_root="${project_root}/data/source/ultrafineweb"
data_root="${project_root}/data/qualification"
smoke_config="training_profiles/irl3_mi300x_smoke.yaml"
main_config="training_profiles/irl3_135m_1x_mi300x_qualification.yaml"
baseline_config="training_profiles/irl2_135m_1x_mi300x_matched.yaml"
session_budget_seconds="${QUALIFICATION_MAX_SECONDS:-13800}"
session_start="$(date +%s)"

mkdir -p "${artifact_root}"
exec > >(tee -a "${artifact_root}/qualification.log") 2>&1
cd "${project_root}"

latest_checkpoint() {
  local root="$1"
  find "${root}" -mindepth 2 -maxdepth 2 -name _SUCCESS -printf '%h\n' \
    | sort | tail -n 1
}

echo "[irl3-qualification] started $(date --iso-8601=seconds)"
echo "[irl3-qualification] wall-clock budget ${session_budget_seconds}s"

export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${artifact_root}/inductor_cache}"

bash scripts/bootstrap_primus_env.sh

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname --showmeminfo vram --showuse --showtemp --showpower \
    | tee "${artifact_root}/rocm_smi.txt"
fi
"${python_bin}" scripts/inspect_accelerator.py \
  --output "${artifact_root}/accelerator_report.json"

echo "[irl3-qualification] repository tests"
timeout --foreground 900s "${python_bin}" -m pytest -q \
  | tee "${artifact_root}/pytest.txt"

bash scripts/download_ultrafineweb_shard.sh "${source_root}"
"${python_bin}" scripts/prepare_ultrafineweb.py \
  --source "${source_root}/shard_0000.npy" \
  --tokenizer "${source_root}/lunaris-tokenizer.json" \
  --output-root "${data_root}" \
  --train-tokens 100000000 \
  --validation-tokens 1000000 \
  --vocab-size 65536
cp "${data_root}/manifest.json" "${artifact_root}/data_manifest.json"

"${python_bin}" -m lunaris_dense_irl3.training.preflight \
  --config "${main_config}" \
  --world-size 1 \
  --require-data \
  --output "${artifact_root}/irl3_preflight.json"
"${python_bin}" -m lunaris_dense_irl2.training.preflight \
  --config "${baseline_config}" \
  --world-size 1 \
  --require-data \
  --output "${artifact_root}/irl2_preflight.json"

echo "[irl3-qualification] four-step BF16 full-path smoke"
timeout --foreground 720s "${python_bin}" -m lunaris_dense_irl3.training.train \
  --config "${smoke_config}"

echo "[irl3-qualification] 135M IRL-3 architecture run"
timeout --foreground 6000s "${python_bin}" -m lunaris_dense_irl3.training.train \
  --config "${main_config}"

echo "[irl3-qualification] checkpoint reload and five-step extension"
timeout --foreground 600s "${python_bin}" -m lunaris_dense_irl3.training.train \
  --config "${main_config}" \
  --max-steps 205 \
  --resume auto

irl3_checkpoint="$(latest_checkpoint "${artifact_root}/checkpoints/main")"
if [[ -z "${irl3_checkpoint}" ]]; then
  echo "No complete IRL-3 checkpoint found" >&2
  exit 1
fi
"${python_bin}" -m lunaris_dense_irl3.training.export_checkpoint \
  --checkpoint "${irl3_checkpoint}" \
  --output "${artifact_root}/portable_irl3"

echo "[irl3-qualification] guaranteed parameter/sequence-matched IRL-2 baseline"
timeout --foreground 1200s "${python_bin}" -m lunaris_dense_irl2.training.train \
  --config "${baseline_config}"
irl2_checkpoint="$(latest_checkpoint "${artifact_root}/checkpoints/irl2_baseline")"
if [[ -z "${irl2_checkpoint}" ]]; then
  echo "No complete IRL-2 checkpoint found" >&2
  exit 1
fi
"${python_bin}" -m lunaris_dense_irl2.training.export_checkpoint \
  --checkpoint "${irl2_checkpoint}" \
  --output "${artifact_root}/portable_irl2"
timeout --foreground 480s "${python_bin}" scripts/benchmark_recurrent_depth.py \
  --architecture irl2 \
  --checkpoint "${artifact_root}/portable_irl2" \
  --data "${data_root}/validation" \
  --depths 1,2,4,6 \
  --sequence-length 4096 \
  --batch-size 1 \
  --batches 1 \
  --repeats 1 \
  --precision bf16 \
  --output "${artifact_root}/irl2_depth.json"

echo "[irl3-qualification] fixed-depth test-time compute sweep"
timeout --foreground 1200s "${python_bin}" scripts/benchmark_recurrent_depth.py \
  --architecture irl3 \
  --checkpoint "${artifact_root}/portable_irl3" \
  --data "${data_root}/validation" \
  --depths 2,4,6,8,12 \
  --sequence-length 4096 \
  --batch-size 1 \
  --batches 2 \
  --repeats 2 \
  --precision bf16 \
  --halt-mode fixed \
  --output "${artifact_root}/irl3_fixed_depth.json"

echo "[irl3-qualification] adaptive-halting budget sweep"
timeout --foreground 480s "${python_bin}" scripts/benchmark_recurrent_depth.py \
  --architecture irl3 \
  --checkpoint "${artifact_root}/portable_irl3" \
  --data "${data_root}/validation" \
  --depths 4,8,12 \
  --sequence-length 4096 \
  --batch-size 1 \
  --batches 2 \
  --repeats 1 \
  --precision bf16 \
  --halt-mode adaptive \
  --output "${artifact_root}/irl3_adaptive_depth.json"

echo "[irl3-qualification] ACT-soft budget sweep"
timeout --foreground 480s "${python_bin}" scripts/benchmark_recurrent_depth.py \
  --architecture irl3 \
  --checkpoint "${artifact_root}/portable_irl3" \
  --data "${data_root}/validation" \
  --depths 4,8,12 \
  --sequence-length 4096 \
  --batch-size 1 \
  --batches 2 \
  --repeats 1 \
  --precision bf16 \
  --halt-mode act \
  --output "${artifact_root}/irl3_act_depth.json"

echo "[irl3-qualification] write-back OFF control at the trained reference depth"
timeout --foreground 300s "${python_bin}" scripts/benchmark_recurrent_depth.py \
  --architecture irl3 \
  --checkpoint "${artifact_root}/portable_irl3" \
  --data "${data_root}/validation" \
  --depths 6 \
  --sequence-length 4096 \
  --batch-size 1 \
  --batches 2 \
  --repeats 2 \
  --precision bf16 \
  --halt-mode fixed \
  --disable-writeback \
  --output "${artifact_root}/irl3_writeback_off.json"

summary_args=(
  --irl3-metrics "${artifact_root}/main_metrics.jsonl"
  --irl3-fixed "${artifact_root}/irl3_fixed_depth.json"
  --irl3-act "${artifact_root}/irl3_act_depth.json"
  --irl3-adaptive "${artifact_root}/irl3_adaptive_depth.json"
  --irl3-writeback-off "${artifact_root}/irl3_writeback_off.json"
  --irl2-metrics "${artifact_root}/irl2_baseline_metrics.jsonl"
  --irl2-depth "${artifact_root}/irl2_depth.json"
  --accelerator "${artifact_root}/accelerator_report.json"
  --json-output "${artifact_root}/summary.json"
  --markdown-output "${artifact_root}/SUMMARY.md"
)
"${python_bin}" scripts/summarize_irl3_qualification.py "${summary_args[@]}"

echo "[irl3-qualification] completed $(date --iso-8601=seconds)"
echo "[irl3-qualification] elapsed $(( $(date +%s) - session_start ))s"
