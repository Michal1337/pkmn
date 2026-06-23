"""v2 head-to-head over the SDK forward model: two v2 agents (each mcts|greedy), each with
its OWN GameTracker+AbilityTracker (via make_ai), mirror deck, sides alternated. Reports
side-A win-rate. Use to measure the v2 MCTS-lift (mcts vs greedy on the SAME net) and to
compare nets (e.g. AZ-net+MCTS vs PPO-net+MCTS).

  PYTHONPATH=. python scripts/h2h_v2_mcts.py --a-ckpt A.pt --a-mode mcts \
      --b-ckpt A.pt --b-mode greedy --games 24 --n-sims 64
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play import make_ai                                          # noqa: E402

import torch                                                      # noqa: E402
torch.set_grad_enabled(False)
from rl.decks import DECKS                                        # noqa: E402
from rl.env import load_deck                                      # noqa: E402
from sdk_cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a-ckpt", required=True); p.add_argument("--a-mode", default="mcts", choices=["mcts", "greedy"])
    p.add_argument("--b-ckpt", required=True); p.add_argument("--b-mode", default="greedy", choices=["mcts", "greedy"])
    p.add_argument("--games", type=int, default=24)
    p.add_argument("--n-sims", type=int, default=64)
    p.add_argument("--n-det", type=int, default=2)
    p.add_argument("--deck", default=None, help="deck name (default: engine sample deck); mirror")
    p.add_argument("--label", default="")
    a = p.parse_args()

    deck = DECKS.get(a.deck) if a.deck else load_deck()
    a_ai, a_arch = make_ai(a.a_ckpt, a.a_mode, a.n_sims, a.n_det, {"deck": deck})
    b_ai, b_arch = make_ai(a.b_ckpt, a.b_mode, a.n_sims, a.n_det, {"deck": deck})
    print(f"A={a_arch}:{a.a_mode}({a.a_ckpt})  vs  B={b_arch}:{a.b_mode}({a.b_ckpt})  "
          f"sims={a.n_sims}x{a.n_det}  games={a.games}", flush=True)

    w = l = d = 0; t0 = time.time()
    for g in range(a.games):
        a_pl = g % 2                                              # alternate which player is side A
        try: battle_finish()
        except Exception: pass
        obs = battle_start(deck, deck)[0]; steps = 0
        while obs["current"]["result"] < 0 and steps < 4000:
            ai = a_ai if obs["current"]["yourIndex"] == a_pl else b_ai
            obs = battle_select(ai(obs)); steps += 1     # ai() handles deck-step + trackers internally
        r = obs["current"]["result"]
        if r == 2: d += 1
        elif r == a_pl: w += 1
        else: l += 1
        print(f"  g{g}: A=P{a_pl} r{r} -> A {w}-{d}-{l}  ({(time.time()-t0)/(g+1):.1f}s/g)", flush=True)
    n = max(1, w + l + d)
    print(f"\n{a.label or 'A vs B'}: A {w}-{d}-{l}  win-rate={w/n:.3f}  "
          f"(excl. draws {w/max(w+l,1):.3f})  over {a.games}", flush=True)


if __name__ == "__main__":
    main()
