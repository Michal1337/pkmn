#!/bin/bash
# One-time CUDA-torch venv for GPU experiments (separate from .venv so it can't
# disturb running CPU jobs). Run on the login node.
set -e
cd ~/src/pkmn

python3 -m venv .venv-gpu
. .venv-gpu/bin/activate
python -m pip install --upgrade pip wheel setuptools

# CUDA torch (cu124 wheels)
pip install torch --index-url https://download.pytorch.org/whl/cu124

pip install -e .
pip install --no-deps "git+https://github.com/Kaggle/kaggle-environments.git"
pip install jsonschema requests
pip install "numpy>=1.26,<2.0"

echo "=== import check ==="
python - <<'PY'
import torch, numpy, rl
from kaggle_environments import make
make("cabt")
print("GPU_VENV_OK torch", torch.__version__, "numpy", numpy.__version__, "cuda_build", torch.version.cuda)
PY
echo "GPU_SETUP_DONE"
