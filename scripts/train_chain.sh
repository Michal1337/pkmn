#!/bin/bash
# Chained PPO self-play -> AlphaZero distill, as two dependent SLURM jobs.
# Run on the Eden LOGIN node -- it only SUBMITS (does not train):
#     bash scripts/train_chain.sh
#
# Rationale: the MLP plateaus ~2M PPO steps, so the PPO phase mainly re-confirms a
# solid MULTI-DECK champion (and adapts to MAX_OPTIONS=128); the AZ phase is the lever
# (search->distill compounding) and gets the bulk of the overnight budget.
#
# Knobs (env vars):
#   INIT_FROM  warm-start checkpoint for PPO ("" = train fresh)   [default: current champion]
#   DECKS      deck pool: all|official|sample|<name>              [default: all]
#   PPO_HOURS / AZ_HOURS   wall-clock per phase                   [default: 04 / 10]
#   GPU_PART / CPU_PART    partitions                             [default: hopper / hopper]
set -euo pipefail
cd ~/src/pkmn

TS=$(date +%Y%m%d_%H%M%S)
RUN=$HOME/pkmn_runs
PPO_OUT=$RUN/ppo_chain_$TS
AZ_OUT=$RUN/az_chain_$TS

DECKS=${DECKS:-all+gen}           # 55-deck pool (official+sample+50 generated)
INIT_FROM=${INIT_FROM:-$RUN/ppo_cpu_1713153/latest.pt}   # current champion; set "" for fresh
PPO_HOURS=${PPO_HOURS:-04}
AZ_HOURS=${AZ_HOURS:-10}
GPU_PART=${GPU_PART:-hopper}
CPU_PART=${CPU_PART:-hopper}
NENVS=${NENVS:-64}                # scale env workers to the 128-core node (was 32)

PPO_INIT=""; [ -n "$INIT_FROM" ] && PPO_INIT="--init-from $INIT_FROM"
echo "[chain] PPO_OUT=$PPO_OUT"
echo "[chain] AZ_OUT =$AZ_OUT   decks=$DECKS   warm=${INIT_FROM:-<fresh>}"

# ---- 1) PPO self-play: 1 learner GPU + NENVS workers on dedicated cores. The opponent
#         runs LOCALLY per worker (jit) -- a central inference server bottlenecked at
#         scale (~2-3x slower at 64 workers), so no second GPU. ----
JID=$(sbatch --parsable \
  -A re-com -p "$GPU_PART" --gres=gpu:1 --cpus-per-task=$((NENVS + 12)) --mem=160G -t "${PPO_HOURS}:00:00" \
  --job-name=pkmn-ppo-chain --output="$HOME/pkmn_ppo_chain_%j.log" \
  --export=ALL,VENV=.venv-gpu,OUT="$PPO_OUT" \
  scripts/train_exp.sh --arch mlp --decks "$DECKS" --num-envs "$NENVS" \
    --total-timesteps 40000000 --selfplay-start 500000 \
    --snapshot-every 200000 --save-every 250000 --device cuda $PPO_INIT)
echo "[chain] submitted PPO: job $JID"

# ---- 2) AZ distill (CPU), starts after PPO TERMINATES (afterany: survives a PPO timeout,
#         since train.py saves latest.pt periodically), warm-started from the PPO output ----
AID=$(sbatch --parsable --dependency=afterany:"$JID" \
  -A re-com -p "$CPU_PART" --gres=gpu:1 --cpus-per-task=64 --mem=160G -t "${AZ_HOURS}:00:00" \
  --job-name=pkmn-az-chain --output="$HOME/pkmn_az_chain_%j.log" \
  --export=ALL,VENV=.venv-gpu,OUT="$AZ_OUT" \
  scripts/az.sh --init-from "$PPO_OUT/latest.pt" --decks "$DECKS" --workers 48 \
    --games-per-iter 384 --iters 300 --n-sims 40 --n-det 2 --device cuda)
echo "[chain] submitted AZ:  job $AID  (afterany:$JID)"
echo "[chain] monitor: squeue -u \$USER ;  tail -f $HOME/pkmn_ppo_chain_${JID}.log"
