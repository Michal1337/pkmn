#!/bin/bash
# One-time environment setup on Eden (run on the login node; install only).
# Order matters: torch (CPU wheel) -> editable install -> kaggle-environments
# (--no-deps) -> repin numpy<2.0 LAST (torch/others drag in numpy 2.x, which
# crashes on the login node's pre-x86-64-v2 CPU).
set -e
cd ~/src/pkmn

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# CPU-only torch (no GPUs on sr-1; the net is tiny so CPU is fine)
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -e .
pip install --no-deps "git+https://github.com/Kaggle/kaggle-environments.git"
# kaggle-environments core runtime deps that --no-deps skips (the rest of its
# deps are heavy ML extras for other envs that cabt does not need).
pip install jsonschema requests
pip install "numpy>=1.26,<2.0"

echo "=== import check ==="
python - <<'PY'
import torch, numpy, rl
from kaggle_environments import make
e = make("cabt")
print("SETUP_OK torch", torch.__version__, "numpy", numpy.__version__, "cabt env loaded")
PY
echo "SETUP_DONE"
