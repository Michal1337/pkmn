"""Round-robin tournament + Bradley-Terry/Elo ratings to track self-play progress.

Self-play win rate hovers at ~0.5 (moving opponent) and win rate vs random
saturates, so neither tracks absolute strength. This plays every contestant
(checkpoints + fixed anchors) against every other, then fits Elo from the win
matrix. Rising Elo over training step = real improvement; the full matrix also
exposes non-transitive strategy cycling (A>B>C>A).

    python -m rl.tournament \
        --ckpts run/ckpt_500000.pt run/ckpt_1000000.pt run/ckpt_2000000.pt \
        --anchor-random --anchor-first --games 60 --temp 0.7

Anchors keep the Elo scale comparable across runs (random is pegged to 0).
"""

from __future__ import annotations

import argparse
import itertools
import logging

logging.disable(logging.CRITICAL)

import numpy as np
import torch

from .card_features import get_card_table
from .encoding import Encoder, SUBMIT_ACTION
from .env import load_deck
from .policy import ActorCritic, obs_to_tensors


# ---- policies: (raw_obs, rng) -> full engine selection list[int] ----------
def random_policy(raw_obs, rng):
    sel = raw_obs["select"]
    n, k = len(sel["option"]), sel["maxCount"]
    return rng.sample(range(n), min(k, n)) if n else []


def first_policy(raw_obs, rng):
    sel = raw_obs["select"]
    return list(range(min(sel["maxCount"], len(sel["option"]))))


def net_policy(net, enc, device, temp=0.0):
    @torch.no_grad()
    def pol(raw_obs, rng):
        sel = raw_obs["select"]
        picked: list[int] = []
        while True:
            o = obs_to_tensors(enc.encode(raw_obs, set(picked)), device)
            o = {k: v[None] for k, v in o.items()}
            logits = net.logits_value(o)[0][0]
            if temp <= 0:
                a = int(logits.argmax())
            else:
                p = torch.softmax(logits / temp, -1)
                a = int(torch.multinomial(p, 1))
            if a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                break
        return sorted(set(picked))
    return pol


def play_game(game, deck, pol0, pol1, rng):
    """Return result: 0 = player0 win, 1 = player1 win, 2 = draw."""
    try:
        game.battle_finish()
    except Exception:
        pass
    obs, sd = game.battle_start(deck, deck)
    if obs is None:
        return 2
    steps = 0
    while obs["current"]["result"] < 0 and steps < 4000:
        idx = obs["current"]["yourIndex"]
        picks = (pol0 if idx == 0 else pol1)(obs, rng)
        try:
            obs = game.battle_select(picks)
        except Exception:
            return 1 - idx  # illegal move -> that player loses
        steps += 1
    return obs["current"]["result"]


def play_match(game, deck, pa, pb, games, rng):
    """pa vs pb over `games`, alternating sides. Returns (wins_a, draws, wins_b)."""
    wa = wb = dr = 0
    for g in range(games):
        if g % 2 == 0:
            r = play_game(game, deck, pa, pb, rng)
            wa += r == 0; wb += r == 1
        else:
            r = play_game(game, deck, pb, pa, rng)
            wb += r == 0; wa += r == 1
        dr += r == 2
    return wa, dr, wb


# ---- Bradley-Terry MLE -> Elo ---------------------------------------------
def fit_elo(win, played, anchor_idx=None, iters=2000, smooth=2.0):
    """win[i,j] = wins of i over j (draws split). Returns Elo ratings.

    `smooth` adds a few phantom split games to each played pairing (Laplace
    prior) so 0-win / all-win players get a finite, bounded rating instead of
    diverging to +/-inf.
    """
    win, played = win.copy(), played.copy()
    if smooth > 0:
        mask = played > 0
        win[mask] += smooth / 2.0
        played[mask] += smooth
    P = win.shape[0]
    W = win.sum(axis=1)                       # total wins per player
    gamma = np.ones(P)
    for _ in range(iters):
        new = np.empty(P)
        for i in range(P):
            denom = sum(played[i, j] / (gamma[i] + gamma[j])
                        for j in range(P) if j != i and played[i, j] > 0)
            new[i] = W[i] / denom if denom > 0 else gamma[i]
        new = np.clip(new, 1e-9, None)
        new /= np.exp(np.mean(np.log(new)))   # normalise (geometric mean 1)
        if np.max(np.abs(np.log(new) - np.log(gamma))) < 1e-9:
            gamma = new; break
        gamma = new
    elo = 400.0 * np.log10(gamma)
    if anchor_idx is not None:
        elo -= elo[anchor_idx]                # peg anchor to 0
    return elo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpts", nargs="*", default=[], help="checkpoint .pt paths")
    p.add_argument("--anchor-random", action="store_true")
    p.add_argument("--anchor-first", action="store_true")
    p.add_argument("--games", type=int, default=60, help="games per pairing")
    p.add_argument("--temp", type=float, default=0.7, help="sampling temp for net policies (0=greedy)")
    p.add_argument("--deck", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import random as _random
    rng = _random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    ct = get_card_table()
    enc = Encoder(ct)
    deck = load_deck(args.deck) if args.deck else load_deck()

    from kaggle_environments.envs.cabt.cg import game

    names, pols = [], []
    for path in args.ckpts:
        ck = torch.load(path, map_location=device)
        net = ActorCritic(enc.cf, ct.vocab_size, **ck.get("net_config", {})).to(device)
        net.load_state_dict(ck["net"]); net.eval()
        step = ck.get("global_step", "?")
        names.append(f"step{step}")
        pols.append(net_policy(net, enc, device, temp=args.temp))
    anchor_idx = None
    if args.anchor_first:
        names.append("first"); pols.append(first_policy)
    if args.anchor_random:
        names.append("random"); pols.append(random_policy)
        anchor_idx = len(names) - 1

    P = len(pols)
    if P < 2:
        raise SystemExit("need at least 2 contestants (add --anchor-random / more ckpts)")

    win = np.zeros((P, P)); played = np.zeros((P, P))
    print(f"contestants: {names}\nplaying {args.games} games/pair, temp={args.temp} ...")
    for i, j in itertools.combinations(range(P), 2):
        wa, dr, wb = play_match(game, deck, pols[i], pols[j], args.games, rng)
        win[i, j] += wa + 0.5 * dr; win[j, i] += wb + 0.5 * dr
        played[i, j] += args.games; played[j, i] += args.games
        print(f"  {names[i]:>14} vs {names[j]:<14}  {wa}-{dr}-{wb}")
    try:
        game.battle_finish()
    except Exception:
        pass

    elo = fit_elo(win, played, anchor_idx=anchor_idx)
    order = np.argsort(-elo)
    print("\n=== Elo ratings (random pegged to 0) ===")
    for r in order:
        wins = win[r].sum(); tot = played[r].sum()
        print(f"  {names[r]:>14}  Elo {elo[r]:+7.1f}   overall {wins/tot:.3f} ({int(tot)} games)")


if __name__ == "__main__":
    main()
