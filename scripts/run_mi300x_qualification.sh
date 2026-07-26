#!/usr/bin/env bash
set -euo pipefail

# End-to-end single-MI300X qualification with resumable checkpoints.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
artifact_root="${project_root}/qualification_artifacts"
source_root="${project_root}/data/source/ultrafineweb"
data_root="${project_root}/data/qualification"
main_config="training_profiles/135m_1x_mi300x_qualification.yaml"
smoke_config="training_profiles/mi300x_smoke.yaml"

mkdir -p "${artifact_root}"
exec > >(tee -a "${artifact_root}/qualification.log") 2>&1

cd "${project_root}"
echo "[qualification] started at $(date --iso-8601=seconds)"

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

"${python_bin}" -m pytest -q \
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

"${python_bin}" - <<'PY'
from pathlib import Path

from lunaris_dense_irl2.training.config import TrainConfig

artifacts = Path("qualification_artifacts")
for name, source in (
    ("smoke_config.json", "training_profiles/mi300x_smoke.yaml"),
    (
        "main_config.json",
        "training_profiles/135m_1x_mi300x_qualification.yaml",
    ),
):
    TrainConfig.from_yaml(source).save_json(artifacts / name)
PY

echo "[qualification] running four-step GPU safety test"
"${python_bin}" -m lunaris_dense_irl2.training.train \
  --config "${smoke_config}"

echo "[qualification] running the 500-step architecture experiment"
"${python_bin}" -m lunaris_dense_irl2.training.train \
  --config "${main_config}"

echo "[qualification] reloading the checkpoint and extending to step 510"
"${python_bin}" -m lunaris_dense_irl2.training.train \
  --config "${main_config}" \
  --max-steps 510 \
  --resume auto

"${python_bin}" scripts/summarize_qualification.py \
  --metrics "${artifact_root}/main_metrics.jsonl" \
  --json-output "${artifact_root}/summary.json" \
  --markdown-output "${artifact_root}/SUMMARY.md"

echo "[qualification] completed at $(date --iso-8601=seconds)"

