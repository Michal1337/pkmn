#!/bin/bash
# AlphaZero (MCTS self-play -> distill) launcher. CPU-bound (self-play in workers).
#   sbatch -p hopper --cpus-per-task=32 --mem=96G -t 04:00:00 \
#     --output=$HOME/pkmn_az_%j.log scripts/az.sh --workers 32 --init-from <champion.pt> ...
#SBATCH -A re-com
#SBATCH --job-name=pkmn-az
set -euo pipefail
ulimit -n 16384 2>/dev/null || ulimit -n 8192 2>/dev/null || true
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd ~/src/pkmn
. "${VENV:-.venv}/bin/activate"
OUT="${OUT:-$HOME/pkmn_runs/az_${SLURM_JOB_ID:-local}}"
echo "out=$OUT  args=$*"
python -m rl.az --out "$OUT" "$@"
