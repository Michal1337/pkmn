"""Rank decks by win-rate IN OUR AGENT'S HANDS (greedy) to (a) choose which deck to
SHIP and (b) expose the weak tail of the training pool. Both seats use the SAME net,
so the result isolates DECK strength rather than net skill. Decks alternate seats to
cancel first-player bias.

  PYTHONPATH=. python scripts/deck_gauntlet.py --net v1 --bundle submission_mcts \
      --candidates meta --panel meta --games 12 --add-ship
  PYTHONPATH=. python scripts/deck_gauntlet.py --net v2 --ckpt rl/runs/.../latest.pt \
      --candidates all --panel meta --games 4
"""
from __future__ import annotations
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # for h2h_v2


def load_csv_deck(path):
    with open(path) as f:
        return [int(x) for x in f if x.strip()]


def make_agents(args):
    """Return (agentA, holderA, agentB, holderB, step) — same net, two instances."""
    holderA, holderB = {"deck": None}, {"deck": None}
    if args.net == "v1":
        from h2h_v2 import _old_agent
        a, step = _old_agent(args.bundle, holderA)
        b, _ = _old_agent(args.bundle, holderB)
    else:
        from h2h_v2 import _v2_agent
        a, step = _v2_agent(args.ckpt, holderA)
        b, _ = _v2_agent(args.ckpt, holderB)
    return a, holderA, b, holderB, step


def get_decks(which):
    from rl.decks import DECKS
    if which == "good":                       # ALL decks the curated generalist trained on (15)
        from rl.train import resolve_deck_pool
        named = dict(DECKS)
        for mod, attr in [("rl.decks_meta", "META"), ("rl.decks_kaggle", "KAGGLE_DECKS")]:
            try:
                named.update(getattr(__import__(mod, fromlist=[attr]), attr))
            except Exception:
                pass
        k = lambda d: tuple(sorted(int(x) for x in d))
        gk = {k(d) for d in resolve_deck_pool("good")}
        return {n: d for n, d in named.items() if k(d) in gk}
    out = dict(DECKS)
    if which == "all":
        from rl.decks_generated import GENERATED
        out.update(GENERATED)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net", choices=["v1", "v2"], default="v1")
    p.add_argument("--bundle", default="submission_mcts")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--candidates", default="meta", choices=["meta", "all", "good"])
    p.add_argument("--panel", default="meta", choices=["meta", "all", "good"])
    p.add_argument("--games", type=int, default=12)
    p.add_argument("--max-cand", type=int, default=0)
    p.add_argument("--add-ship", action="store_true", help="include agent/deck.csv as a candidate")
    p.add_argument("--chunk", default=None, help="'i/n' — process strided candidate slice i of n")
    p.add_argument("--out", default=None, help="write results as JSON to this path")
    args = p.parse_args()

    from kaggle_environments import make
    cand = get_decks(args.candidates)
    panel = get_decks(args.panel)
    if args.add_ship:
        cand = {"SHIPPED_sample": load_csv_deck(os.path.join(ROOT, "agent", "deck.csv")), **cand}
    cand_items = list(cand.items())[: args.max_cand] if args.max_cand else list(cand.items())
    if args.chunk:
        i, n = (int(x) for x in args.chunk.split("/"))
        cand_items = cand_items[i::n]
    panel_items = list(panel.items())

    aA, hA, aB, hB, step = make_agents(args)
    env = make("cabt", debug=False)
    print(f"deck gauntlet: net={args.net} step={step} candidates={len(cand_items)} "
          f"panel={len(panel_items)} games/matchup={args.games} (greedy)", flush=True)

    results = {}
    for cname, cdeck in cand_items:
        w = d = l = 0
        for pname, pdeck in panel_items:
            if pname == cname:
                continue
            hA["deck"], hB["deck"] = cdeck, pdeck
            for g in range(args.games):
                cand_p0 = (g % 2 == 0)
                agents = [aA, aB] if cand_p0 else [aB, aA]
                env.reset(); env.run(agents)
                r0, r1 = env.state[0]["reward"], env.state[1]["reward"]
                cr, opp = (r0, r1) if cand_p0 else (r1, r0)
                if cr == opp:
                    d += 1
                elif cr > opp:
                    w += 1
                else:
                    l += 1
        tot = w + d + l
        wr = w / max(tot, 1)
        results[cname] = (w, d, l, wr)
        print(f"  {cname:22s}: {w:3d}-{d:3d}-{l:3d}  wr={wr:.3f}  ({tot} games)", flush=True)

    print("\n=== ranked by win-rate vs panel ===", flush=True)
    for name, (w, d, l, wr) in sorted(results.items(), key=lambda x: -x[1][3]):
        print(f"  {wr:.3f}  {name}  ({w}-{d}-{l})", flush=True)

    if args.out:
        import json
        with open(args.out, "w") as f:
            json.dump({k: v[:3] for k, v in results.items()}, f)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
