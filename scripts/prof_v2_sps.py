"""Diagnose v2 (token-transformer) self-play SPS: where the time goes and what helps.

(1) single CPU forward latency (1 thread) + encode cost + int8-dynamic-quant speedup
(2) self-play SPS (real net opponent) vs warmup SPS (random opponent) across num_envs
    -> the gap is the per-worker transformer-opponent tax; the num_envs sweep exposes
       CPU contention from oversubscription (which helps mlp but may hurt tf).

    PYTHONPATH=. python scripts/prof_v2_sps.py
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import torch

torch.set_grad_enabled(False)

from rl.card_features import get_card_table
from rl.encoding import TokenEncoder
from rl.policy import build_token_net
from rl.vec_env import SubprocVecEnv
import rl.decks_generated as dg

NC = {"arch": "transformer2", "emb_dim": 48, "d_model": 128, "nhead": 4, "nlayers": 3, "ff": 256}


def single_forward_bench():
    torch.set_num_threads(1)
    enc = TokenEncoder(get_card_table())
    net = build_token_net(enc.cards, NC); net.eval()
    from sdk_cg.game import battle_start, battle_select, battle_finish
    decks = list(dg.GENERATED.values())
    try: battle_finish()
    except Exception: pass
    obs = battle_start(decks[0], decks[1])[0]
    while obs.get("select") is None and obs["current"]["result"] < 0:
        obs = battle_select([int(c) for c in decks[0]])
    try: battle_finish()
    except Exception: pass

    int_keys = enc.int_keys
    def to_t(o):
        return {k: torch.as_tensor(np.asarray(v)[None], dtype=(torch.long if k in int_keys else torch.float32))
                for k, v in o.items()}
    o = enc.encode(obs, set(), self_deck=decks[0]); t = to_t(o)

    def time_ms(fn, n=200):
        for _ in range(5): fn()
        t0 = time.time()
        for _ in range(n): fn()
        return (time.time() - t0) / n * 1000

    fwd = time_ms(lambda: net.logits_value(t))
    encode = time_ms(lambda: enc.encode(obs, set(), self_deck=decks[0]))
    print(f"[single decision, 1 thread, b=1]")
    print(f"  encode      = {encode:.2f} ms")
    print(f"  fwd fp32    = {fwd:.2f} ms")
    try:
        qnet = torch.quantization.quantize_dynamic(net, {torch.nn.Linear}, dtype=torch.qint8)
        qfwd = time_ms(lambda: qnet.logits_value(t))
        print(f"  fwd int8    = {qfwd:.2f} ms ({fwd/qfwd:.2f}x)")
    except Exception as e:
        print(f"  fwd int8    = FAILED ({type(e).__name__}: {str(e)[:60]})")
    # dynamic-seq-truncation: drop padded tokens (mathematically identical under pad mask)
    print(f"  -> per-decision opponent cost ~= encode+fwd = {encode+fwd:.2f} ms fp32", flush=True)
    return {k: v.cpu() for k, v in net.state_dict().items()}


def selfplay_sps(num_envs, opp_weights, steps=100, warmup=15, mode="local", server_device="cpu"):
    decks = list(dg.GENERATED.values())
    env_kwargs = dict(agent_decks=decks, opponent_decks=decks, randomize_side=True)
    ve = SubprocVecEnv(num_envs, env_kwargs, NC, base_seed=0,
                       opponent_mode=mode, server_device=server_device)
    try:
        ve.set_opponent(opp_weights)
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
    from rl.policy import _TRUNC_B1
    envs = [int(x) for x in sys.argv[1:]] or [8, 16, 24, 32]
    print(f"cpu_count: {os.cpu_count()}   b1_truncation={'ON' if _TRUNC_B1 else 'OFF (baseline)'}", flush=True)
    sd = single_forward_bench()
    srv_dev = os.environ.get("SRV_DEV", "cpu")
    print(f"\n[self-play SPS] warmup(random) | local(per-worker CPU) | server({srv_dev} batched)", flush=True)
    for ne in envs:
        rnd = selfplay_sps(ne, None, mode="local")
        loc = selfplay_sps(ne, sd, mode="local")
        srv = selfplay_sps(ne, sd, mode="server", server_device=srv_dev)
        print(f"  num_envs={ne:3d}  warmup={rnd:7.0f}   local={loc:7.0f} ({loc/ne:4.1f}/env)   "
              f"server={srv:7.0f} ({srv/ne:4.1f}/env)   server/local={srv/max(loc,1):.2f}x", flush=True)
