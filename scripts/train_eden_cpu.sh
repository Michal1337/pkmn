#!/bin/bash
# CPU-only PPO on sr-1 (no free GPUs). The cabt engine is CPU-bound anyway and
# the net is tiny (~385k params), so CPU training is fine; throughput scales
# with --num-envs subprocess workers.
#
# Submit (pinned to sr-1):
#   sbatch --output=$HOME/pkmn_ppo_cpu_%j.log scripts/train_eden_cpu.sh
# Override: STEPS=... OUT=... or append rl.train flags.

#SBATCH -A re-com
#SBATCH -p long
#SBATCH -w sr-1
#SBATCH --cpus-per-task=24
#SBATCH --mem=96G
#SBATCH -t 48:00:00
#SBATCH --job-name=pkmn-ppo-cpu

set -euo pipefail
# one thread per process: 24 env workers + cheap CPU PPO update; avoids the
# torch/OpenSpiel thread explosion that 24 workers would otherwise cause.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd ~/src/pkmn
. .venv/bin/activate

OUT="${OUT:-$HOME/pkmn_runs/ppo_cpu_${SLURM_JOB_ID:-local}}"
echo "writing checkpoints to $OUT"

python -m rl.train \
  --num-envs 24 \
  --total-timesteps "${STEPS:-50000000}" \
  --selfplay-start 1000000 \
  --snapshot-every 200000 \
  --save-every 500000 \
  --device cpu \
  --out "$OUT" \
  "$@"
