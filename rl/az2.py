"""AlphaZero for the v2 token-transformer (encoding/policy2) — the v2 counterpart of az.py.

MCTS self-play via search_agent2.mcts_policy: BOTH players move by MCTS; we record
(encoded state, MCTS visit distribution pi, outcome z) at each searchable decision and train
the net so its policy matches pi (CE) and its value matches z (MSE). This trains the value
head ON THE SEARCH-PLAY DISTRIBUTION — the proper fix that the one-shot frozen-trunk value
calibration could not give (it regressed weak temp-1.0 play and LOST 0.39 head-to-head).

Each player keeps its OWN GameTracker + AbilityTracker fed only its own decision obs, and the
recorded training state is encoded the same way the live agent encodes -> train == test.
Self-play is CPU-heavy (MCTS + engine), so it runs in subprocess workers (one SDK sim each).
Warm-start from a PPO v2 checkpoint via --init-from.

    python -m rl.az2 --workers 24 --games-per-iter 192 --iters 60 --n-sims 64 \
        --init-from $HOME/pkmn_runs/v2_baseline/ckpt_5505024.pt --out $HOME/pkmn_runs/az2
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _selfplay_game(net, enc, deck_pool, n_sims, n_det, rng):
    """One MCTS self-play game, per-player decks sampled from the pool. Each side keeps its
    own trackers (fed only its own decision obs). Returns [(enc_obs, pi, player)]."""
    from sdk_cg.game import battle_start, battle_select, battle_finish
    from .encoding import GameTracker, AbilityTracker
    from .search_agent2 import mcts_policy
    try:
        battle_finish()
    except Exception:
        pass
    decks = {0: rng.choice(deck_pool), 1: rng.choice(deck_pool)}
    tr = {0: GameTracker(), 1: GameTracker()}
    ab = {0: AbilityTracker(), 1: AbilityTracker()}
    obs = battle_start(decks[0], decks[1])[0]
    samples, steps = [], 0
    while obs["current"]["result"] < 0 and steps < 4000:
        sel = obs["select"]; p = obs["current"]["yourIndex"]
        if sel is None:
            obs = battle_select([int(c) for c in decks[p]]); continue
        tr[p].update(obs); ab[p].note_turn((obs.get("current") or {}).get("turn"))
        pick, pi = mcts_policy(obs, net, enc, decks[p], tr[p], ab[p].slots,
                               n_sims=n_sims, n_det=n_det, rng=rng)
        if pi is not None:
            samples.append([enc.encode(obs, set(), self_deck=decks[p], tracker=tr[p],
                                       ability_slots=ab[p].slots), pi, p])
        ab[p].record(sel, pick)
        obs = battle_select(pick); steps += 1
    result = obs["current"]["result"]
    return [(arrs, pi, np.float32(0.0 if result == 2 else (1.0 if result == p else -1.0)))
            for arrs, pi, p in samples]


def _worker(remote, parent_remote, net_config, deck_pool, n_sims, n_det, seed):
    parent_remote.close()
    import logging; logging.disable(logging.CRITICAL)
    import random
    import torch as T
    T.set_num_threads(1)
    from .card_features import get_card_table
    from .encoding import TokenEncoder
    from .policy2 import build_token_net
    enc = TokenEncoder(get_card_table())
    net = build_token_net(enc.cards, net_config); net.eval()      # no jit: transformer doesn't trace cleanly
    rng = random.Random(seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "weights":
                net.load_state_dict({k: T.as_tensor(v) for k, v in data.items()}); net.eval()
                remote.send(True)
            elif cmd == "play":
                samples = []
                for _ in range(data):
                    samples.extend(_selfplay_game(net, enc, deck_pool, n_sims, n_det, rng))
                remote.send(samples)
            elif cmd == "close":
                remote.send(True); break
    except (KeyboardInterrupt, EOFError):
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init-from", type=str, default=None, help="warm-start net (PPO v2 checkpoint)")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--games-per-iter", type=int, default=192)
    p.add_argument("--iters", type=int, default=60)
    p.add_argument("--n-sims", type=int, default=64)
    p.add_argument("--n-det", type=int, default=2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--decks", type=str, default="all+gen", help="all|gen|all+gen|official|sample|<name>")
    # transformer dims (used only if not warm-starting; else taken from the ckpt's net_config)
    p.add_argument("--emb-dim", type=int, default=48)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--static", action="store_true",
                   help="static per-card features (MUST match the --init-from net, e.g. the abomS specialist)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=str, default=os.path.join(os.environ.get("HOME", "."), "pkmn_runs", "az2"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"[cfg] {vars(args)}", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    from .card_features import get_card_table
    from .decks import DECKS
    try:
        from .decks_generated import GENERATED
    except Exception:
        GENERATED = {}
    from .encoding import TokenEncoder
    from .env import load_deck
    from .policy2 import build_token_net
    from .policy import load_compatible
    ct = get_card_table(); enc = TokenEncoder(ct)

    if args.decks == "all":
        deck_pool = list(DECKS.values()) + [load_deck()]
    elif args.decks == "gen":
        deck_pool = list(GENERATED.values()) or [load_deck()]
    elif args.decks == "all+gen":
        deck_pool = list(DECKS.values()) + [load_deck()] + list(GENERATED.values())
    elif args.decks == "official":
        deck_pool = list(DECKS.values())
    elif args.decks == "sample":
        deck_pool = [load_deck()]
    elif args.decks in DECKS:
        deck_pool = [DECKS[args.decks]]
    elif args.decks in GENERATED:
        deck_pool = [GENERATED[args.decks]]
    else:
        raise SystemExit(f"unknown --decks '{args.decks}'")
    print(f"[decks] pool={args.decks} size={len(deck_pool)}", flush=True)

    net_config = {"arch": "transformer2", "emb_dim": args.emb_dim, "d_model": args.d_model,
                  "nhead": args.nhead, "nlayers": args.nlayers, "ff": args.ff, "static": args.static}
    ck = None
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        net_config = ck.get("net_config", net_config)
    net = build_token_net(ct, net_config).to(device)
    if ck is not None:
        skipped = load_compatible(net, ck["net"])
        print(f"[init] warm-started from {args.init_from}"
              + (f" (reinit {len(skipped)})" if skipped else ""), flush=True)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    print(f"[net] config={net_config} params={sum(p.numel() for p in net.parameters()):,}", flush=True)

    ctx = mp.get_context("spawn")
    remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(args.workers)])
    procs = []
    for i, (wr, r) in enumerate(zip(work_remotes, remotes)):
        p = ctx.Process(target=_worker, args=(wr, r, net_config, deck_pool, args.n_sims, args.n_det,
                                              args.seed * 100 + i), daemon=True)
        p.start(); procs.append(p)
    for wr in work_remotes:
        wr.close()

    start = time.time()
    for it in range(1, args.iters + 1):
        sd = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}   # numpy: no fd blowup
        for r in remotes:
            r.send(("weights", sd))
        for r in remotes:
            r.recv()
        per = [args.games_per_iter // args.workers] * args.workers
        for r, g in zip(remotes, per):
            r.send(("play", g))
        samples = []
        for r in remotes:
            samples.extend(r.recv())
        if not samples:
            print(f"[it {it}] no samples; skipping", flush=True); continue

        obs_keys = list(samples[0][0].keys())
        B = len(samples)
        obs_b = {k: torch.as_tensor(np.stack([s[0][k] for s in samples]),
                                    dtype=(torch.long if k in enc.int_keys else torch.float32),
                                    device=device) for k in obs_keys}
        pi_b = torch.as_tensor(np.stack([s[1] for s in samples]), dtype=torch.float32, device=device)
        z_b = torch.as_tensor(np.array([s[2] for s in samples]), dtype=torch.float32, device=device)

        net.train()
        idx = np.arange(B); pl = vl = 0.0; nb = 0
        for _ in range(args.epochs):
            np.random.shuffle(idx)
            for s0 in range(0, B, args.batch_size):
                mb = idx[s0:s0 + args.batch_size]
                o = {k: obs_b[k][mb] for k in obs_keys}
                logits, value = net.logits_value(o)
                logp = torch.log_softmax(logits, dim=-1)
                policy_loss = -(pi_b[mb] * logp).sum(-1).mean()
                value_loss = ((value - z_b[mb]) ** 2).mean()
                loss = policy_loss + args.vf_coef * value_loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                pl += float(policy_loss); vl += float(value_loss); nb += 1
        net.eval()
        print(f"[it {it}/{args.iters}] samples={B} games={args.games_per_iter} "
              f"policy_loss={pl/nb:.3f} value_loss={vl/nb:.3f} elapsed={int(time.time()-start)}s", flush=True)
        torch.save({"net": net.state_dict(), "net_config": net_config, "iter": it, "args": vars(args)},
                   os.path.join(args.out, "latest.pt"))
        if it % 5 == 0 or it == args.iters:
            torch.save({"net": net.state_dict(), "net_config": net_config, "iter": it},
                       os.path.join(args.out, f"az2_it{it}.pt"))
            print(f"[ckpt] az2_it{it}.pt", flush=True)

    for r in remotes:
        try: r.send(("close", None)); r.recv()
        except Exception: pass
    print("done.", flush=True)


if __name__ == "__main__":
    main()
