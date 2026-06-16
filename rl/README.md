# RL training (cabt PPO + self-play)

CleanRL-style masked PPO with a per-option policy and a random→self-play
curriculum. See `STATE_ACTION_SPACE.md` for the observation/action design.

## Modules

| file | role |
|---|---|
| `card_features.py` | `EN_Card_Data.csv` → static per-card feature table (`[1268, 59]`) + embedding vocab |
| `encoding.py` | `obs` dict → fixed-shape numpy arrays + per-option features + action mask |
| `env.py` | `CabtEnv`: single-agent Gym-style wrapper (direct engine driving, multi-select buffering, pluggable opponent, ±1/0 reward) |
| `policy.py` | `ActorCritic`: shared card embedding, board encoder, per-option scorer + submit head + value head (~385k params) |
| `vec_env.py` | `SubprocVecEnv`: one battle per process, auto-reset, broadcasts opponent snapshots for self-play |
| `train.py` | PPO loop, GAE, masking, self-play scheduler, checkpoints |

## Key constraints (from probing the engine)

- **One battle per process** (global native pointer) → parallelism = subprocess
  workers. Throughput scales with `--num-envs`; the net is tiny so the GPU is
  nearly idle (it's a CPU-bound job).
- **Dynamic action space** → the actor scores each option from its own features
  and masks illegal/padded slots; multi-select is buffered into single picks.

## Local run

```bash
pip install -r requirements.txt        # torch, then repin numpy<2.0
python -m rl.train --num-envs 8 --total-timesteps 2000000 --device cuda
# quick check (tiny):
python -m rl.train --num-envs 4 --num-steps 16 --total-timesteps 4096 --selfplay-start 1024
```

Logs print `winrate` (fraction of finished episodes with reward > 0; ~0.5 vs an
equal opponent), `ep_ret`, PPO losses, `sps`, and the self-play flag. Checkpoints
(`ckpt_<step>.pt`, `latest.pt`) hold `net`, `net_config`, and `args`.

## Eden cluster

CPU-bound + one-battle-per-process ⇒ many CPUs + a small MIG GPU (see `HPC.md`).

```bash
# setup (once)
cd ~/src/pkmn && export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e .
uv pip install --no-deps "git+https://github.com/Kaggle/kaggle-environments.git"
uv pip install "numpy>=1.26,<2.0"      # repin LAST

# submit (hopper-2 H200 MIG, 16 workers, 24h)
sbatch --output=$HOME/pkmn_ppo_%j.log scripts/train_eden.sh --total-timesteps 40000000
```

The script defaults to `gpu:h200_3g.71gb:1`, `--cpus-per-task=16`, `-A re-com`,
and 16 env workers. Override steps/output via `STEPS=...`, `OUT=...`, or by
appending `rl.train` flags.

## Tuning knobs

- `--num-envs` — rollout throughput (match `--cpus-per-task`).
- `--selfplay-start` / `--snapshot-every` / `--pool-size` — curriculum.
- `--shaping prize_diff` — dense reward (net prize cards taken) for early signal.
- `--num-steps`, `--lr`, `--ent-coef`, `--update-epochs` — standard PPO.

## Not yet built (next steps)

- `evaluate` a checkpoint vs the random/`first` baselines over N games.
- Export a trained net into a Kaggle submission (`agent/main.py` that loads the
  weights + encoder + card CSV and runs greedy inference).
- Optional: bind the engine's native `SearchBegin/Step` for lookahead/MCTS, and
  learn the deck (currently fixed to `agent/deck.csv`).
