"""Mine downloaded Kaggle episode JSONs for the decks real agents play.
Episode deck ids == our EN_Card_Data.csv ids (same cabt engine), so decks are directly
usable + guaranteed legal. Clusters decks by archetype (their ex/Mega-ex Pokemon), ranks
by frequency + win-rate, prints a representative 60-card list per archetype, and flags Crustle.

  PYTHONPATH=. python scripts/mine_episodes.py _kaggle_scout/ep
"""
from __future__ import annotations
import csv
import glob
import json
import os
import re
import sys
import collections

EP_DIR = sys.argv[1] if len(sys.argv) > 1 else "_kaggle_scout/ep"


def card_db():
    name, is_key = {}, {}
    for r in csv.DictReader(open("EN_Card_Data.csv", encoding="utf-8")):
        cid = r.get("Card ID") or r.get("﻿Card ID")
        try:
            cid = int(re.search(r"\d+", cid).group())
        except Exception:
            continue
        nm = r["Card Name"]
        name[cid] = nm
        is_key[cid] = nm.endswith(" ex") or nm.startswith("Mega ")   # archetype marker
    return name, is_key


def is_deck(a):
    return isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a)


def extract(ep):
    d = json.load(open(ep, encoding="utf-8"))
    rew = d.get("rewards") or [0, 0]
    decks = {}
    for step in d["steps"]:
        for pi, ag in enumerate(step):
            a = ag.get("action")
            if is_deck(a) and pi not in decks:
                decks[pi] = a
        if len(decks) == 2:
            break
    out = []
    for pi, dk in decks.items():
        won = rew[pi] > rew[1 - pi]
        out.append((tuple(dk), won))
    return out


def archetype(deck, name, is_key):
    keys = sorted({name.get(c, str(c)) for c in set(deck) if is_key.get(c)})
    return " + ".join(keys) if keys else "(no-ex)"


def main():
    name, is_key = card_db()
    eps = glob.glob(os.path.join(EP_DIR, "*.json"))
    groups = collections.defaultdict(lambda: {"n": 0, "w": 0, "lists": collections.Counter()})
    n_decks = 0
    for ep in eps:
        try:
            for deck, won in extract(ep):
                arch = archetype(deck, name, is_key)
                g = groups[arch]
                g["n"] += 1
                g["w"] += int(won)
                g["lists"][deck] += 1
                n_decks += 1
        except Exception:
            continue
    print(f"episodes={len(eps)}  decks={n_decks}  archetypes={len(groups)}\n")
    ranked = sorted(groups.items(), key=lambda kv: -kv[1]["n"])
    print("=== archetypes by frequency (n, winrate) ===")
    for arch, g in ranked[:25]:
        print(f"  n={g['n']:3d}  wr={g['w']/g['n']:.2f}  variants={len(g['lists'])}  {arch[:70]}")
    # Crustle: count decks containing Crustle/Dwebble across ALL groups
    crustle = [(lst, c) for g in groups.values() for lst, c in g["lists"].items() if 345 in lst or 344 in lst]
    n_crustle = sum(c for _, c in crustle)
    print(f"\n=== Crustle decks: {n_crustle} ({len(crustle)} variants) ===")

    # dump representative (most-common exact) list per archetype n>=8, plus Crustle
    from sdk_cg.game import battle_start, battle_finish

    def ok(deck):
        try:
            battle_start(list(deck), list(deck))
            try: battle_finish()
            except Exception: pass
            return True
        except Exception:
            return False

    dump = {}
    for arch, g in ranked:
        if g["n"] < 8 or arch == "(no-ex)":
            continue
        key = "kaggle_" + re.sub(r"[^a-z0-9]+", "_", arch.lower()).strip("_")
        dump[key] = g["lists"].most_common(1)[0][0]
    if crustle:
        dump["kaggle_crustle"] = max(crustle, key=lambda x: x[1])[0]   # most common Crustle list

    valid = {k: list(v) for k, v in dump.items() if ok(v)}
    with open("rl/decks_kaggle.py", "w", encoding="utf-8") as f:
        f.write('"""Decks mined from real Kaggle competition episodes (2026-06-19), by win/frequency.\n')
        f.write('Native cabt card ids (directly usable, guaranteed legal)."""\n\nKAGGLE_DECKS = {\n')
        for k, v in valid.items():
            f.write(f"    {k!r}: {v},\n")
        f.write("}\n")
    print(f"\nwrote {len(valid)} validated decks -> rl/decks_kaggle.py: {list(valid)}")
    return groups, name


if __name__ == "__main__":
    main()
