"""Map researched competition decklists (by card NAME) onto EN_Card_Data.csv card ids:
report coverage/missing, measure the card-coverage GAIN vs our existing pool, and write the
fully-mapped 60-card decks to rl/decks_meta.py for use in training.
"""
from __future__ import annotations
import csv
import re
import collections

CSV = "EN_Card_Data.csv"


def norm(s):
    s = (s or "").lower().strip()
    s = s.replace("’", "'").replace("‘", "'").replace("é", "e")
    s = re.sub(r"\s+", " ", s)
    return s


def load_names():
    name2ids = collections.defaultdict(list)
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        cid = r.get("Card ID") or r.get("﻿Card ID")
        try:
            cid = int(re.search(r"\d+", cid).group())
        except Exception:
            continue
        name2ids[norm(r["Card Name"])].append(cid)
    return name2ids


# --- verified competition decklists (name -> count), strongest placements -----
DECKS = {
    "meta_gardevoir_wada_1st": {  # 1st Regional Belo Horizonte
        "Ralts": 3, "Kirlia": 2, "Gardevoir ex": 2, "Mega Gardevoir ex": 1, "Munkidori": 3,
        "Drifloon": 1, "Scream Tail": 1, "Cleffa": 1, "Fezandipiti ex": 1, "Lillie's Clefairy ex": 1,
        "Iron Bundle": 1, "Iono": 4, "Lillie's Determination": 3, "Professor's Research": 3,
        "Ultra Ball": 4, "Earthen Vessel": 3, "Rare Candy": 3, "Night Stretcher": 2, "Nest Ball": 2,
        "Super Rod": 1, "Counter Catcher": 1, "Secret Box": 1, "Bravery Charm": 2, "Mystery Garden": 2,
        "Artazon": 1, "Psychic Energy": 8, "Darkness Energy": 3,
    },
    "meta_lucario_tasaki_14th": {  # 14th Champions League Fukuoka
        "Riolu": 4, "Mega Lucario ex": 3, "Solrock": 3, "Lunatone": 2, "Munkidori": 2, "Meowth ex": 1,
        "Lillie's Determination": 4, "Boss's Orders": 3, "Judge": 2, "Crispin": 1, "Wally's Compassion": 1,
        "Carmine": 1, "Ultra Ball": 4, "Poke Pad": 4, "Fighting Gong": 4, "Premium Power Pro": 4,
        "Switch": 1, "Scoop Up Cyclone": 1, "Air Balloon": 1, "Gravity Mountain": 1,
        "Fighting Energy": 11, "Darkness Energy": 2,
    },
    "meta_lucario_takahashi_27th": {  # 27th Champions League Fukuoka
        "Riolu": 4, "Mega Lucario ex": 3, "Solrock": 3, "Lunatone": 2, "Munkidori": 2,
        "Lillie's Determination": 4, "Boss's Orders": 4, "Poke Pad": 4, "Fighting Gong": 4,
        "Premium Power Pro": 4, "Ultra Ball": 3, "Crispin": 2, "Night Stretcher": 2, "Switch": 2,
        "Judge": 1, "Unfair Stamp": 1, "Gravity Mountain": 1, "Team Rocket's Watchtower": 1,
        "Fighting Energy": 10, "Darkness Energy": 3,
    },
    "meta_gardevoir_jellicent_sato_20th": {  # 20th Regional Las Vegas
        "Ralts": 3, "Kirlia": 2, "Gardevoir ex": 2, "Mega Gardevoir ex": 1, "Munkidori": 3,
        "Frillish": 2, "Jellicent ex": 1, "Lillie's Clefairy ex": 1, "Mew ex": 1, "Scream Tail": 1,
        "Lillie's Determination": 4, "Iono": 4, "Arven": 2, "Ultra Ball": 4, "Earthen Vessel": 3,
        "Night Stretcher": 2, "Rare Candy": 2, "Buddy-Buddy Poffin": 1, "Nest Ball": 1, "Super Rod": 1,
        "Secret Box": 1, "Counter Catcher": 1, "Technical Machine: Evolution": 1,
        "Technical Machine: Devolution": 1, "Air Balloon": 1, "Bravery Charm": 1, "Mystery Garden": 2,
        "Artazon": 1, "Psychic Energy": 7, "Darkness Energy": 3,
    },
}

ALIASES = {
    "psychic energy": ["basic {p} energy", "basic psychic energy", "{p} energy"],
    "fighting energy": ["basic {f} energy", "basic fighting energy", "{f} energy"],
    "darkness energy": ["basic {d} energy", "basic darkness energy", "{d} energy", "dark energy"],
    "poke pad": ["pokepad", "poke pad"],
    "technical machine: evolution": ["technical machine evolution"],
    "technical machine: devolution": ["technical machine devolution"],
}


def resolve(name, name2ids):
    n = norm(name)
    if n in name2ids:
        return name2ids[n][0]
    for alt in ALIASES.get(n, []):
        if norm(alt) in name2ids:
            return name2ids[norm(alt)][0]
    hits = [v[0] for k, v in name2ids.items() if n == k or (len(n) > 5 and n in k)]
    return hits[0] if len(set(hits)) == 1 else None


def main():
    name2ids = load_names()
    built = {}
    for deck, cards in DECKS.items():
        ids, missing, total = [], [], 0
        for name, cnt in cards.items():
            total += cnt
            r = resolve(name, name2ids)
            (ids.extend([r] * cnt) if r is not None else missing.append(f"{name} x{cnt}"))
        ok = (not missing and total == 60)
        print(f"=== {deck}: total={total} mapped={len(ids)} missing={len(missing)} {'OK' if ok else ''}")
        if missing:
            print("   MISSING:", "; ".join(missing))
        if ok:
            built[deck] = ids

    # coverage gain vs existing pool
    import sys; sys.path.insert(0, ".")
    from rl.decks import DECKS as POOL
    try:
        from rl.decks_generated import GENERATED
    except Exception:
        GENERATED = {}
    existing = set()
    for d in list(POOL.values()) + list(GENERATED.values()):
        existing.update(d)
    meta_ids = set()
    for ids in built.values():
        meta_ids.update(ids)
    new = meta_ids - existing
    print(f"\nCOVERAGE: existing pool covers {len(existing)} unique card ids over "
          f"{len(POOL)+len(GENERATED)} decks; the {len(built)} meta decks use {len(meta_ids)} unique ids, "
          f"of which {len(new)} are NEW (not in any existing deck).")

    # write decks_meta.py
    if built:
        with open("rl/decks_meta.py", "w", encoding="utf-8") as f:
            f.write('"""Competition meta decks (Mega Evolution Standard) mapped to our card ids.\n')
            f.write('Auto-generated by scripts/map_decks.py from verified Limitless decklists.\n"""\n\n')
            f.write("META = {\n")
            for name, ids in built.items():
                f.write(f"    {name!r}: {ids},\n")
            f.write("}\n")
        print(f"\nwrote rl/decks_meta.py with {len(built)} decks")


if __name__ == "__main__":
    main()
