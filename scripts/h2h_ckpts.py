"""Head-to-head between two checkpoints piloting the SAME deck (default mega_abomasnow),
alternating seats. Isolates which NET plays the deck better — e.g. an abomasnow-specialist
vs an all-decks generalist, both on abomasnow.

  PYTHONPATH=. python scripts/h2h_ckpts.py --a _play/abom_latest.pt --b _play/gen_2M.pt \
      --deck mega_abomasnow --games 40
"""
from __future__ import annotations
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # play.py
from play import make_ai  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True, help="ckpt A (e.g. abomasnow specialist)")
    p.add_argument("--b", required=True, help="ckpt B (e.g. generalist)")
    p.add_argument("--deck", default="mega_abomasnow")
    p.add_argument("--a-deck", default=None, help="deck for A (default: --deck)")
    p.add_argument("--b-deck", default=None, help="deck for B (default: --deck)")
    p.add_argument("--games", type=int, default=40)
    p.add_argument("--mode", choices=["greedy", "mcts"], default="greedy")
    p.add_argument("--a-mode", choices=["greedy", "mcts"], default=None)
    p.add_argument("--b-mode", choices=["greedy", "mcts"], default=None)
    p.add_argument("--n-sims", type=int, default=40)
    p.add_argument("--n-det", type=int, default=2)
    a = p.parse_args()

    from kaggle_environments import make
    from rl.decks import DECKS
    deckA = DECKS[a.a_deck or a.deck]
    deckB = DECKS[a.b_deck or a.deck]
    a_mode, b_mode = (a.a_mode or a.mode), (a.b_mode or a.mode)
    aiA, arA = make_ai(a.a, a_mode, a.n_sims, a.n_det, {"deck": deckA})
    aiB, arB = make_ai(a.b, b_mode, a.n_sims, a.n_det, {"deck": deckB})
    print(f"A={a.a} ({arA},{a_mode}) deck={a.a_deck or a.deck}  vs  B={a.b} ({arB},{b_mode}) deck={a.b_deck or a.deck}  "
          f"games={a.games}", flush=True)

    env = make("cabt", debug=False)
    w = d = l = 0
    for g in range(a.games):
        a_p0 = (g % 2 == 0)
        agents = [aiA, aiB] if a_p0 else [aiB, aiA]
        env.reset(); env.run(agents)
        r0, r1 = env.state[0]["reward"], env.state[1]["reward"]
        ar, br = (r0, r1) if a_p0 else (r1, r0)
        if ar == br:
            d += 1
        elif ar > br:
            w += 1
        else:
            l += 1
        if (g + 1) % 10 == 0:
            print(f"  after {g+1}: A {w}-{d}-{l} (wr={w/max(w+l+d,1):.3f})", flush=True)
    tot = w + d + l
    print(f"\nA(={os.path.basename(a.a)}) vs B(={os.path.basename(a.b)}): {w}-{d}-{l}  "
          f"A win-rate={w/max(tot,1):.3f}  (excl draws {w/max(w+l,1):.3f})  over {tot} games", flush=True)


if __name__ == "__main__":
    main()
