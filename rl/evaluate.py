"""Greedy evaluation of a trained checkpoint vs a baseline opponent.

Unlike the training-time win rate (which samples actions with an entropy bonus
over a noisy sliding window), this runs the policy GREEDILY (argmax over legal
options) for N full games and reports the true win/draw/loss split.

    python -m rl.evaluate --ckpt ~/pkmn_runs/.../latest.pt --games 200 --opponent random
    python -m rl.evaluate --ckpt latest.pt --games 100 --opponent first
    python -m rl.evaluate --ckpt latest.pt --games 100 --opponent self
"""

from __future__ import annotations

import argparse
import logging
import math

logging.disable(logging.CRITICAL)

import numpy as np
import torch

from .card_features import get_card_table
from .decks import DECKS
from .encoding import Encoder, SUBMIT_ACTION
from .env import CabtEnv, load_deck, random_opponent
from .policy import ActorCritic, greedy_action


def named_deck(name: str) -> list[int]:
    """'sample' -> engine sample; otherwise a deck name from rl.decks."""
    return load_deck() if name in ("sample", None) else DECKS[name]


def deck_pool(name: str) -> list[list[int]]:
    if name in ("pool", "all"):
        return list(DECKS.values()) + [load_deck()]
    if name == "official":
        return list(DECKS.values())
    return [named_deck(name)]


def _play(net, device, opp_fn, agent_decks, opp_decks, games, seed):
    """Greedy agent over `games`; returns (wins, draws, losses)."""
    env = CabtEnv(agent_decks=agent_decks, opponent_decks=opp_decks,
                  opponent_fn=opp_fn, seed=seed)
    w = d = l = 0
    for _ in range(games):
        obs, _ = env.reset()
        done = False; r = 0.0
        while not done:
            obs, r, term, trunc, _ = env.step(greedy_action(net, obs, device))
            done = term or trunc
        w += r > 0; l += r < 0; d += r == 0
    env.close()
    return w, d, l


def first_opponent(raw_obs, rng):
    """Deterministic baseline: take the first maxCount legal options."""
    sel = raw_obs["select"]
    n, k = len(sel["option"]), sel["maxCount"]
    return list(range(min(k, n)))


def make_self_opponent(net, enc, device):
    """Greedy mirror opponent driven by the same net (with multi-select buffering)."""
    @torch.no_grad()
    def opp(raw_obs, rng):
        sel = raw_obs["select"]
        picked: list[int] = []
        while True:
            o = enc.encode(raw_obs, set(picked))
            a = greedy_action(net, o, device)
            if a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                break
        return sorted(set(picked))
    return opp


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, default=None, help="checkpoint .pt (omit = random-init net, for smoke test)")
    p.add_argument("--games", type=int, default=200)
    p.add_argument("--opponent", choices=["random", "first", "self"], default="random")
    p.add_argument("--agent-deck", type=str, default="sample", help="deck our agent pilots (name|sample)")
    p.add_argument("--opp-deck", type=str, default="sample", help="opponent deck(s): name|sample|pool|official")
    p.add_argument("--sweep", action="store_true",
                   help="sweep our agent's deck over the pool vs --opp-deck; report best for submission")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device(args.device)
    ct = get_card_table()
    enc = Encoder(ct)

    net_config = {}
    net = ActorCritic(enc.cf, ct.vocab_size, **net_config).to(device)
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location=device)
        net_config = ck.get("net_config", {})
        net = ActorCritic(enc.cf, ct.vocab_size, **net_config).to(device)
        net.load_state_dict(ck["net"])
        print(f"loaded {args.ckpt} (trained to step {ck.get('global_step')})")
    net.eval()

    if args.opponent == "random":
        opp_fn = random_opponent
    elif args.opponent == "first":
        opp_fn = first_opponent
    else:
        opp_fn = make_self_opponent(net, enc, device)

    opp_decks = deck_pool(args.opp_deck)
    n = args.games

    if args.sweep:
        # which of our decks does THIS policy pilot best? -> submission deck.
        candidates = list(DECKS) + ["sample"]
        print(f"sweeping {len(candidates)} agent decks x {n} games vs opp='{args.opp_deck}' ({args.opponent})\n")
        rows = []
        for name in candidates:
            w, d, l = _play(net, device, opp_fn, [named_deck(name)], opp_decks, n, args.seed)
            wr = w / n
            rows.append((name, wr, w, d, l))
            print(f"  {name:16} win_rate={wr:.3f}  (W{w} D{d} L{l})")
        rows.sort(key=lambda r: -r[1])
        print(f"\nBEST DECK TO SUBMIT: {rows[0][0]}  (win_rate {rows[0][1]:.3f})")
        return

    w, d, l = _play(net, device, opp_fn, [named_deck(args.agent_deck)], opp_decks, n, args.seed)
    wins, draws, losses = w, d, l
    wr = wins / n
    nonloss = (wins + draws) / n
    se = math.sqrt(wr * (1 - wr) / n)  # binomial std error on win rate
    print(f"=== greedy {args.ckpt or 'random-init'} (deck={args.agent_deck}) vs {args.opponent} (deck={args.opp_deck}) | {n} games ===")
    print(f"  wins={wins} draws={draws} losses={losses}")
    print(f"  win_rate={wr:.3f} +/- {1.96 * se:.3f} (95% CI)   non_loss={nonloss:.3f}")


if __name__ == "__main__":
    main()
