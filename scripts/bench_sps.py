"""Benchmark env-collection throughput vs num_envs at a fixed CPU allocation.

Measures the REAL bottleneck (SubprocVecEnv stepping the cabt engine + per-decision
encoding, random opponent = current warmup regime). No PPO update / GPU -> isolates
collection. Use to pick the best 'envs per cpu' (oversubscription) ratio before long runs.

    python scripts/bench_sps.py          # sweeps mlp + transformer2
"""
import sys
import time
import numpy as np
from rl.vec_env import SubprocVecEnv
import rl.decks_generated as dg


def bench(num_envs, arch, steps=150, warmup=20):
    decks = list(dg.GENERATED.values())
    env_kwargs = dict(agent_decks=decks, opponent_decks=decks, randomize_side=True)
    nc = {"arch": arch}
    if arch == "transformer2":
        nc.update(emb_dim=48, d_model=128, nhead=4, nlayers=3, ff=256)
    ve = SubprocVecEnv(num_envs, env_kwargs, nc, base_seed=0)
    try:
        ve.set_opponent(None)                       # random opponent = warmup regime
        obs, _ = ve.reset()
        for _ in range(warmup):
            obs, _, _, _ = ve.step([int(np.argmax(r)) for r in obs["action_mask"]])
        t0 = time.time()
        for _ in range(steps):
            obs, _, _, _ = ve.step([int(np.argmax(r)) for r in obs["action_mask"]])
        dt = time.time() - t0
    finally:
        ve.close()
    return num_envs * steps / dt


if __name__ == "__main__":
    sweeps = {"mlp": [32, 64, 96, 128], "transformer2": [32, 64, 96]}
    print("env-collection throughput (random opp); cpus =", __import__("os").cpu_count(), flush=True)
    for arch, nes in sweeps.items():
        for ne in nes:
            try:
                sps = bench(ne, arch)
                print(f"  arch={arch:13s} num_envs={ne:3d}  sps={sps:7.0f}  ({sps/ne:5.2f}/env)", flush=True)
            except Exception as e:
                print(f"  arch={arch:13s} num_envs={ne:3d}  FAILED: {e}", flush=True)
