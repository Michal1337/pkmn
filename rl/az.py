"""AlphaZero-style training: MCTS self-play -> distill into the net.

Justified by the result that PUCT MCTS beats the plain net (~0.75): we generate
self-play games where BOTH players move by MCTS, record (state, MCTS visit
distribution pi, outcome z) at each searchable decision, and train the net so its
policy matches pi (cross-entropy) and its value matches z (MSE). A stronger net
makes MCTS stronger -> better data -> repeat.

Self-play is CPU-heavy (MCTS + engine search), so it runs in subprocess workers
(one SDK simulator per process). Warm-start from the PPO champion via --init-from.

    python -m rl.az --workers 24 --games-per-iter 240 --iters 50 \
        --init-from $HOME/pkmn_runs/.../latest.pt --n-sims 40 --out $HOME/pkmn_runs/az
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


def _selfplay_game(net, enc, deck_pool, device, n_sims, n_det, rng):
    """One MCTS self-play game with per-player decks sampled from the pool (cross-deck,
    so the net + inference learn diverse matchups, not just the mirror). Returns list of
    (obs_arrays, pi, player)."""
    from sdk_cg.game import battle_start, battle_select, battle_finish
    from .search_agent import mcts_policy
    try:
        battle_finish()
    except Exception:
        pass
    decks = {0: rng.choice(deck_pool), 1: rng.choice(deck_pool)}
    obs = battle_start(decks[0], decks[1])[0]
    samples, steps = [], 0
    while obs["current"]["result"] < 0 and steps < 4000:
        sel = obs["select"]
        player = obs["current"]["yourIndex"]
        if sel is None:
            obs = battle_select([int(c) for c in decks[player]]); continue
        pick, pi = mcts_policy(obs, net, enc, decks[player], device, n_sims=n_sims, n_det=n_det, rng=rng)
        if pi is not None:
            samples.append([enc.encode(obs), pi, player])
        obs = battle_select(pick); steps += 1
    result = obs["current"]["result"]
    out = []
    for arrs, pi, player in samples:
        z = 0.0 if result == 2 else (1.0 if result == player else -1.0)
        out.append((arrs, pi, np.float32(z)))
    return out


def _worker(remote, parent_remote, net_config, deck_pool, n_sims, n_det, seed):
    parent_remote.close()
    import logging; logging.disable(logging.CRITICAL)
    import random
    import torch as T
    T.set_num_threads(1)
    from .card_features import get_card_table
    from .encoding import Encoder
    from .policy import build_net
    enc = Encoder(get_card_table())
    net = build_net(enc.cf, enc.cards.vocab_size, net_config)
    net.eval()
    rng = random.Random(seed)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "weights":
                net.load_state_dict(data); net.eval()
                remote.send(True)
            elif cmd == "play":
                n_games = data
                samples = []
                for _ in range(n_games):
                    samples.extend(_selfplay_game(net, enc, deck_pool, "cpu", n_sims, n_det, rng))
                remote.send(samples)
            elif cmd == "close":
                remote.send(True); break
    except (KeyboardInterrupt, EOFError):
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init-from", type=str, default=None, help="warm-start net (PPO champion)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--games-per-iter", type=int, default=160)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--n-sims", type=int, default=40)
    p.add_argument("--n-det", type=int, default=2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--deck", type=str, default=None, help="single deck file (overrides --decks)")
    p.add_argument("--decks", type=str, default="all", help="deck pool: all|official|sample|<name>")
    p.add_argument("--emb-dim", type=int, default=32)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=str, default=os.path.join(os.environ.get("HOME", "."), "pkmn_runs", "az"))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"[cfg] {vars(args)}", flush=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    from .card_features import get_card_table
    from .decks import DECKS
    from .encoding import Encoder
    from .env import load_deck
    from .policy import build_net, load_compatible, obs_to_tensors
    ct = get_card_table(); enc = Encoder(ct)
    if args.deck:
        deck_pool = [[int(l) for l in open(args.deck) if l.strip()]]
    elif args.decks == "all":
        deck_pool = list(DECKS.values()) + [load_deck()]
    elif args.decks == "official":
        deck_pool = list(DECKS.values())
    elif args.decks == "sample":
        deck_pool = [load_deck()]
    elif args.decks in DECKS:
        deck_pool = [DECKS[args.decks]]
    else:
        raise SystemExit(f"unknown --decks '{args.decks}' (all|official|sample|{'|'.join(DECKS)})")
    print(f"[decks] pool={'file:'+args.deck if args.deck else args.decks} size={len(deck_pool)}", flush=True)

    # warm start (champion is MLP default arch); net_config carried for workers
    net_config = {"arch": "mlp", "emb_dim": args.emb_dim}
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        net_config = ck.get("net_config", net_config)
    net = build_net(enc.cf, ct.vocab_size, net_config).to(device)
    if args.init_from:
        skipped = load_compatible(net, ck["net"])
        msg = f" (reinit {len(skipped)} params: {skipped})" if skipped else ""
        print(f"[init] warm-started from {args.init_from}{msg}", flush=True)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    print(f"[net] params={sum(p.numel() for p in net.parameters()):,}", flush=True)

    ctx = mp.get_context("spawn")
    remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(args.workers)])
    procs = []
    for i, (wr, r) in enumerate(zip(work_remotes, remotes)):
        p = ctx.Process(target=_worker, args=(wr, r, net_config, deck_pool, args.n_sims, args.n_det, args.seed * 100 + i), daemon=True)
        p.start(); procs.append(p)
    for wr in work_remotes:
        wr.close()

    shapes = enc.shapes
    start = time.time()
    for it in range(1, args.iters + 1):
        sd = {k: v.cpu() for k, v in net.state_dict().items()}
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

        # assemble training tensors
        obs_keys = list(shapes)
        B = len(samples)
        obs_b = {k: torch.as_tensor(np.stack([s[0][k] for s in samples]),
                                    dtype=(torch.long if k in enc.int_keys else torch.float32),
                                    device=device) for k in obs_keys}
        pi_b = torch.as_tensor(np.stack([s[1] for s in samples]), dtype=torch.float32, device=device)
        z_b = torch.as_tensor(np.array([s[2] for s in samples]), dtype=torch.float32, device=device)

        net.train()
        idx = np.arange(B)
        pl = vl = 0.0; nb = 0
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
        sps = int(B / (time.time() - start)) if it == 1 else None
        print(f"[it {it}/{args.iters}] samples={B} games={args.games_per_iter} "
              f"policy_loss={pl/nb:.3f} value_loss={vl/nb:.3f} elapsed={int(time.time()-start)}s", flush=True)
        torch.save({"net": net.state_dict(), "net_config": net_config, "iter": it, "args": vars(args)},
                   os.path.join(args.out, "latest.pt"))
        if it % 5 == 0 or it == args.iters:
            torch.save({"net": net.state_dict(), "net_config": net_config, "iter": it},
                       os.path.join(args.out, f"az_it{it}.pt"))
            print(f"[ckpt] az_it{it}.pt", flush=True)

    for r in remotes:
        try: r.send(("close", None)); r.recv()
        except Exception: pass
    print("done.", flush=True)


if __name__ == "__main__":
    main()
