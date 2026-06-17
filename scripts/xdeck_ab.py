"""Cross-deck A/B: does observation-conditioned inference beat the mirror assumption?

Agent plays the sample deck with MCTS (champion net + search). Opponent is the SAME
champion net (trained on all decks, so competent) playing a DIFFERENT deck. We run the
MCTS twice -- once with the real inference determinization, once with the old mirror
determinization -- so the only variable is the opponent model the search assumes.
"""
import logging; logging.disable(logging.CRITICAL)
import random, sys, time
import torch
from rl.card_features import get_card_table
from rl.encoding import Encoder
from rl.policy import build_net
from rl import search_agent as SA
from rl.decks import DECKS
from rl.env import load_deck
from sdk_cg.game import battle_start, battle_select, battle_finish

ct = get_card_table(); enc = Encoder(ct)
ck = torch.load("ckpt_baseline.pt", map_location="cpu")
net = build_net(enc.cf, ct.vocab_size, ck.get("net_config", {"emb_dim": 32}))
net.load_state_dict(ck["net"]); net.eval()
SAMPLE = load_deck()
N_SIMS, N_DET = 24, 2
GAMES = 12                      # per (deck, mode); sides alternated

_INFER = SA._determinize        # the real observation-conditioned determinizer

def _mirror(obs, deck, rng, enc):
    """Old behavior: sample every hidden card from the AGENT's own deck."""
    s = obs["current"]; me = s["yourIndex"]
    mp, op = s["players"][me], s["players"][1 - me]
    pick = lambda n: [rng.choice(deck) for _ in range(n)]
    fa = bool(op.get("active") and op["active"][0] is None)
    return dict(your_deck=pick(mp["deckCount"]), your_prize=pick(len(mp.get("prize") or [])),
                opponent_deck=pick(op["deckCount"]), opponent_prize=pick(len(op.get("prize") or [])),
                opponent_hand=pick(op.get("handCount", 0)),
                opponent_active=([rng.choice(deck)] if fa else []))

def play(agent_idx, opp_deck, rng):
    d0, d1 = (SAMPLE, opp_deck) if agent_idx == 0 else (opp_deck, SAMPLE)
    try: battle_finish()
    except Exception: pass
    obs = battle_start(d0, d1)[0]
    steps = 0
    while obs["current"]["result"] < 0 and steps < 4000:
        sel = obs.get("select")
        if sel is None:
            deck = SAMPLE if obs["current"]["yourIndex"] == agent_idx else opp_deck
            obs = battle_select([int(c) for c in deck]); continue
        if obs["current"]["yourIndex"] == agent_idx:
            pick = SA.mcts_select(obs, net, enc, SAMPLE, "cpu", n_sims=N_SIMS, n_det=N_DET, rng=rng)
        else:
            pick = SA._net_greedy_select(obs, net, enc, "cpu")
        obs = battle_select(pick); steps += 1
    return obs["current"]["result"]

def run_mode(name, det_fn):
    SA._determinize = det_fn
    rng = random.Random(123)
    print(f"\n=== mode={name} ===", flush=True)
    for dname in ("dragapult", "iono", "mega_lucario"):
        opp = DECKS[dname]; w = l = d = 0
        for g in range(GAMES):
            ai = g % 2                       # alternate sides
            r = play(ai, opp, rng)
            if r == 2: d += 1
            elif r == ai: w += 1
            else: l += 1
        print(f"  vs {dname:14s}: W{w} L{l} D{d}  winrate {w/max(1,w+l+d):.3f}", flush=True)

t0 = time.time()
run_mode("infer", _INFER)
run_mode("mirror", _mirror)
print(f"\ntotal {int(time.time()-t0)}s", flush=True)
