#!/bin/bash
# PPO training on the Eden cluster. The cabt engine is CPU-bound and runs one
# battle per process, so this is a many-CPU / tiny-GPU job: 16 subprocess env
# workers + one small MIG GPU slice.
#
# Submit from $HOME so the log lands there:
#   sbatch --output=$HOME/pkmn_ppo_%j.log scripts/train_eden.sh
# Override hyperparams by appending them (passed through to rl.train), e.g.:
#   sbatch --output=$HOME/pkmn_ppo_%j.log scripts/train_eden.sh --total-timesteps 40000000
#
# First-time setup on the cluster (see HPC.md):
#   cd ~/src/pkmn
#   export PATH="$HOME/.local/bin:$PATH"
#   uv venv --python 3.12 .venv && . .venv/bin/activate
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
#   uv pip install -e .
#   uv pip install --no-deps "git+https://github.com/Kaggle/kaggle-environments.git"
#   uv pip install "numpy>=1.26,<2.0"     # repin LAST (torch drags in numpy 2.x)

#SBATCH -A re-com
#SBATCH -p hopper-2
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH --job-name=pkmn-ppo

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

cd ~/src/pkmn
. .venv/bin/activate

OUT="${OUT:-$HOME/pkmn_runs/ppo_${SLURM_JOB_ID:-local}}"
echo "writing checkpoints to $OUT"

python -m rl.train \
  --num-envs 16 \
  --total-timesteps "${STEPS:-20000000}" \
  --selfplay-start 1000000 \
  --snapshot-every 200000 \
  --device cuda \
  --out "$OUT" \
  "$@"
