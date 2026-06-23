"""Round-robin greedy h2h among labeled (ckpt, deck) agents. Each agent pilots its own deck,
alternating seats. Prints the full win matrix + a total-wins ranking. Use to find a specialist's
mirror-strength peak across checkpoints AND cross-deck matchups in one run.

  PYTHONPATH=. python scripts/rr_h2h.py --games 24 \
      --agents abom4M:~/pkmn_runs/abomS/ckpt_4014080.pt:mega_abomasnow \
               abom6M:~/pkmn_runs/abomS/ckpt_6012928.pt:mega_abomasnow ...

CAVEAT: mirror/self-lineage Elo is a WEAK leaderboard proxy (LB can peak earlier then regress on
self-play overfit). Cross-deck and external-anchor results are more meaningful than within-lineage.
"""
from __future__ import annotations
import argparse
import itertools
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))  # play.make_ai
from play import make_ai  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--agents", nargs="+", required=True, help="label:ckpt:deck triples")
    p.add_argument("--games", type=int, default=24, help="games per pairing (split over both seats)")
    p.add_argument("--mode", choices=["greedy", "mcts"], default="greedy")
    p.add_argument("--n-sims", type=int, default=40)
    p.add_argument("--n-det", type=int, default=2)
    a = p.parse_args()

    from kaggle_environments import make
    from rl.decks import DECKS
    ALL_DECKS = dict(DECKS)                                # named + generated + meta archetypes
    try:
        from rl.decks_generated import GENERATED; ALL_DECKS.update(GENERATED)
    except Exception:
        pass
    try:
        from rl.decks_meta import META; ALL_DECKS.update(META)
    except Exception:
        pass

    specs = []
    for s in a.agents:
        label, ck, deck = s.split(":")
        ck = os.path.expanduser(ck)
        ai, arch = make_ai(ck, a.mode, a.n_sims, a.n_det, {"deck": ALL_DECKS[deck]})
        specs.append((label, ai, deck))
        print(f"loaded {label} ({arch},{a.mode}) deck={deck}", flush=True)

    env = make("cabt", debug=False)
    labels = [s[0] for s in specs]
    wins = {l: 0 for l in labels}
    losses = {l: 0 for l in labels}
    played = {l: 0 for l in labels}

    for (la, aiA, da), (lb, aiB, db) in itertools.combinations(specs, 2):
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
        wins[la] += w; losses[la] += l; played[la] += w + d + l
        wins[lb] += l; losses[lb] += w; played[lb] += w + d + l
        print(f"{la} vs {lb}: {w}-{d}-{l}  ({la} wr={w / max(w + l, 1):.3f})", flush=True)

    print("\n=== RANKING (total wins across all pairings) ===", flush=True)
    for lab in sorted(labels, key=lambda x: -wins[x]):
        print(f"  {lab:10s} wins={wins[lab]:3d} losses={losses[lab]:3d} "
              f"wr={wins[lab] / max(wins[lab] + losses[lab], 1):.3f}  ({played[lab]} games)", flush=True)


if __name__ == "__main__":
    main()
