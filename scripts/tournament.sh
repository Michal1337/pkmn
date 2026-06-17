#!/bin/bash
# Run an Elo round-robin as a batch job. Resources via sbatch flags; contestant
# ckpts + options via script args (passed through to rl.tournament).
#   sbatch -p short --cpus-per-task=16 --mem=64G -t 01:00:00 \
#     --output=$HOME/pkmn_ladder_%j.log scripts/tournament.sh \
#     --ckpts a.pt b.pt ... --anchor-first --anchor-random --games 16
#SBATCH -A re-com
#SBATCH --job-name=pkmn-tourney
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
cd ~/src/pkmn
. "${VENV:-.venv}/bin/activate"
python -m rl.tournament "$@"
