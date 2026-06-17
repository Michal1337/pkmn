"""Generate diverse, legal 60-card decks from the in-pool cards (EN_Card_Data.csv).

Adapts the Kaggle deck engine (nursrijan/pokemon-tcg-eda-deck-engine): trace each
attacker's evolution line, add a VARIED Trainer/tool/stadium/ACE package, fill with
TYPE-MATCHED basic energy (the notebook's generic 'Basic {C} Energy' isn't a real
card), validate tournament legality, and resolve every card to its Card ID so the decks
plug straight into our training pool.

Selection spans many archetypes: it balances regular `Pokémon ex` and `Mega Pokémon ex`
attackers across every energy type, and the large support pools (filtered to in-pool)
mean different decks draw different cards -> broader embedding coverage.

  python scripts/build_decks.py --n 30 --out rl/decks_generated.py
"""
from __future__ import annotations
import argparse, csv, os, random, re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "EN_Card_Data.csv")
STAGE = "Stage (Pokémon)/Type (Energy and Trainer)"

ENERGY = {"{G}": "Basic {G} Energy", "{R}": "Basic {R} Energy", "{W}": "Basic {W} Energy",
          "{L}": "Basic {L} Energy", "{P}": "Basic {P} Energy", "{F}": "Basic {F} Energy",
          "{D}": "Basic {D} Energy", "{M}": "Basic {M} Energy"}

# candidate support cards (anything not in the pool is silently dropped at runtime).
ITEMS = ["Ultra Ball", "Nest Ball", "Buddy-Buddy Poffin", "Night Stretcher", "Super Rod",
         "Earthen Vessel", "Switch", "Counter Catcher", "Pokégear 3.0", "Pokémon Catcher",
         "Trekking Shoes", "Great Ball", "Energy Retrieval", "Pal Pad", "Lost Vacuum",
         "Hisuian Heavy Ball", "Energy Search", "Switch Cart", "Scoop Up Cyclone",
         "Electric Generator", "Capturing Aroma", "Precious Trolley"]
SUPPORTERS = ["Professor's Research", "Iono", "Carmine", "Boss's Orders", "Judge", "Arven",
              "Cynthia's Power Weight", "Marnie", "Roxanne", "Klara", "Penny", "Larry",
              "Bird Keeper", "Colress's Experiment", "Lillie's Determination", "Crispin"]
TOOLS = ["Air Balloon", "Bravery Charm", "Rescue Board", "Defiance Band", "Vitality Band",
         "Leftovers", "Technical Machine: Evolution", "Technical Machine: Devolution"]
STADIUMS = ["Surfing Beach", "Artazon", "Jamming Tower", "Town Store", "Gravity Mountain",
            "Levincia", "Area Zero Underdepths", "Magma Basin"]
ACE_SPECS = ["Master Ball", "Prime Catcher", "Unfair Stamp", "Hero's Cape", "Maximum Belt",
             "Secret Box", "Awakening Drum", "Sparkling Crystal", "Scramble Switch"]


def load():
    rows_by_name = defaultdict(list); meta = {}; name2id = {}
    with open(CSV, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            r = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            nm, cid = r["Card Name"], int(r["Card ID"])
            rows_by_name[nm].append(r)
            name2id[nm] = min(cid, name2id.get(nm, cid))
            meta.setdefault(nm, r)
    return rows_by_name, meta, name2id


def trace_evolution(name, meta):
    path, cur = [name], name
    while True:
        r = meta.get(cur)
        if not r:
            break
        prev = r.get("Previous stage")
        if not prev or prev in ("n/a", "None", cur):
            break
        path.append(prev); cur = prev
    return path


def attack_energy_types(name, rows_by_name, meta):
    types = Counter()
    for r in rows_by_name.get(name, []):
        for sym in re.findall(r"\{[A-Z]\}", r.get("Cost") or ""):
            if sym in ENERGY:
                types[sym] += 1
    if types:
        return [t for t, _ in types.most_common()]
    t = meta.get(name, {}).get("Type") or ""
    return [t] if t in ENERGY else ["{W}"]


def _is_ace(c, meta):
    return "ACE SPEC" in (meta.get(c, {}).get("Rule") or "")


def build_deck(attacker, meta, rows_by_name, name2id, rng):
    if attacker not in name2id:
        return None
    evo = list(reversed(trace_evolution(attacker, meta)))
    if any(c not in name2id for c in evo):
        return None
    deck = Counter()
    for c, q in zip(evo, {1: [4], 2: [4, 3], 3: [4, 3, 3]}.get(len(evo), [4] + [3] * (len(evo) - 1))):
        deck[c] += q
    stage2 = any("Stage 2" in (meta[c].get(STAGE) or "") for c in evo)

    def pool(names, exclude_ace=True):
        return [x for x in names if x in name2id and not (exclude_ace and _is_ace(x, meta))]
    items, sups, tools, stads, aces = (pool(ITEMS), pool(SUPPORTERS), pool(TOOLS),
                                       pool(STADIUMS), [x for x in ACE_SPECS if x in name2id])
    for p in (items, sups, tools, stads, aces):
        rng.shuffle(p)

    if stage2 and "Rare Candy" in name2id:
        deck["Rare Candy"] += 4
    if "Ultra Ball" in name2id:
        deck["Ultra Ball"] += 4
    for it in items[:rng.randint(6, 8)]:
        deck[it] += rng.choice([1, 2, 2, 3])
    for sp in sups[:rng.randint(4, 5)]:
        deck[sp] += rng.choice([2, 3, 4])
    for tl in tools[:rng.randint(1, 2)]:
        deck[tl] += rng.choice([1, 2])
    if stads:
        deck[stads[0]] += rng.choice([2, 3])
    if aces:
        deck[aces[0]] += 1                   # exactly one ACE SPEC

    for c in list(deck):                     # cap non-energy at 4
        deck[c] = min(deck[c], 4)
    while sum(deck.values()) > 47:           # leave room for >=13 energy
        cands = [c for c in deck if deck[c] > 1 and c not in evo and not _is_ace(c, meta)]
        if not cands:
            break
        deck[rng.choice(cands)] -= 1

    etypes = attack_energy_types(attacker, rows_by_name, meta)
    need = 60 - sum(deck.values())
    if need < 0:
        return None
    for i in range(need):
        deck[ENERGY[etypes[i % len(etypes)]]] += 1
    return deck if sum(deck.values()) == 60 else None


def validate(deck, meta, name2id):
    if sum(deck.values()) != 60:
        return f"not 60 ({sum(deck.values())})"
    for c, q in deck.items():
        if c not in name2id:
            return f"unknown {c!r}"
        if not (c.startswith("Basic ") and c.endswith("Energy")) and q > 4:
            return f"{c} x{q}>4"
    if sum(1 for c in deck if _is_ace(c, meta)) > 1:
        return "multiple ACE SPEC"
    for c in deck:
        prev = meta.get(c, {}).get("Previous stage")
        if prev and prev not in ("n/a", "None", c) and prev not in deck:
            return f"{c} missing pre-evo {prev}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(ROOT, "rl", "decks_generated.py"))
    a = ap.parse_args()
    rows_by_name, meta, name2id = load()
    rng = random.Random(a.seed)

    # candidates split by rule tier so we don't get 20 Mega beatsticks
    mega, regular = [], []
    for nm, r in meta.items():
        rule = r.get("Rule") or ""; stage = r.get(STAGE) or ""
        if "Pokémon" not in stage or "ex" not in rule:
            continue
        try: hp = int(r.get("HP") or 0)
        except ValueError: hp = 0
        (mega if "Mega" in rule else regular).append((hp, nm, r.get("Type") or "?"))
    mega.sort(reverse=True); regular.sort(reverse=True)
    print(f"candidate attackers: {len(mega)} Mega-ex, {len(regular)} regular-ex")

    decks, by_type, seen_basic = {}, Counter(), Counter()
    # interleave the two tiers so the set is balanced
    interleaved = [x for pair in zip(mega, regular + [None] * len(mega)) for x in pair if x]
    interleaved += regular[len(mega):]
    cap = max(4, a.n // 5)                    # per energy-type cap (soft balance)
    for hp, nm, typ in interleaved:
        if len(decks) >= a.n:
            break
        if by_type[typ] >= cap:
            continue
        basic = trace_evolution(nm, meta)[-1]
        if seen_basic[basic] >= 2:           # allow regular-ex + Mega-ex of a line
            continue
        deck = build_deck(nm, meta, rows_by_name, name2id, rng)
        if deck is None or validate(deck, meta, name2id):
            continue
        key = re.sub(r"[^a-z0-9]+", "_", nm.lower()).strip("_")
        decks[key] = (nm, deck); by_type[typ] += 1; seen_basic[basic] += 1

    lines = ['"""Auto-generated decks (scripts/build_decks.py) for embedding coverage."""',
             "", "GENERATED = {"]
    uniq = set()
    for key, (nm, deck) in decks.items():
        ids = []
        for c, q in deck.items():
            ids += [name2id[c]] * q
            uniq.add(name2id[c])
        lines.append(f"    {key!r}: {sorted(ids)},  # {nm}: {len(deck)} kinds")
    lines.append("}")
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"generated {len(decks)} decks -> {a.out}   ({by_type.most_common()})")
    print(f"unique card ids across decks: {len(uniq)} / {len(name2id)} pool cards")


if __name__ == "__main__":
    main()
