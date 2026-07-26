#!/usr/bin/env bash
set -euo pipefail

# Run this script inside the AMD Primus PyTorch/ROCm environment.
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"

cd "${project_root}"

"${python_bin}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot see the MI300X accelerator")
if torch.version.hip is None:
    raise RuntimeError("The active PyTorch build is not a ROCm build")
print(f"[bootstrap] torch={torch.__version__} rocm={torch.version.hip}")
print(f"[bootstrap] device={torch.cuda.get_device_name(0)}")
PY

# Install only lightweight project dependencies and never replace Primus PyTorch.
if ! "${python_bin}" -c "import numpy, pytest, yaml" >/dev/null 2>&1; then
  "${python_bin}" -m pip install "numpy>=1.26" "pytest>=8.0" "PyYAML>=6.0"
fi
"${python_bin}" -m pip install --no-deps --editable .

