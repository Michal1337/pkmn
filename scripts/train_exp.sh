#!/bin/bash
# Generic experiment launcher. Resources come from sbatch FLAGS (partition,
# cpus, mem, time, gres, nodelist); the rl.train config comes from script ARGS.
# Example:
#   sbatch -p hopper --cpus-per-task=48 --mem=96G -t 04:00:00 \
#     --output=$HOME/pkmn_exp_%j.log scripts/train_exp.sh \
#     --num-envs 48 --decks all --selfplay-start 500000 --trunk-h 512 --device cpu
#SBATCH -A re-com
#SBATCH --job-name=pkmn-exp

set -euo pipefail
# many subprocess workers each open lots of fds (torch + kaggle_environments);
# raise the open-file limit so high --num-envs doesn't hit Errno 24.
ulimit -n 16384 2>/dev/null || ulimit -n 8192 2>/dev/null || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd ~/src/pkmn
. "${VENV:-.venv}/bin/activate"   # VENV=.venv-gpu for CUDA runs

OUT="${OUT:-$HOME/pkmn_runs/exp_${SLURM_JOB_ID:-local}}"
echo "out=$OUT  args=$*"
python -m rl.train --out "$OUT" "$@"
