"""Play the cabt Pokemon TCG against a trained checkpoint, in the terminal.

Renders the board + your legal options each of your decisions; you type the option
number(s). The opponent is a checkpoint (v1 mlp or v2 transformer; greedy or MCTS,
auto-detected from the ckpt). `--demo` plays AI-vs-AI and just renders (no input).

  PYTHONPATH=. python scripts/play.py --ckpt _tourney/v2_latest/latest.pt --mode mcts --n-sims 80
  PYTHONPATH=. python scripts/play.py --ckpt _tourney/v1_playfix/latest.pt --mode greedy --demo
"""
from __future__ import annotations
import argparse
import random
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)

from rl.card_features import get_card_table
from rl.decks import DECKS
from rl.env import load_deck
from sdk_cg.game import battle_start, battle_select, battle_finish

ENERGY = "CGRWLPFDM A"
# Must match sdk_cg.api.OptionType exactly (the old map mislabeled YES/NO/DISCARD/RETREAT/etc.)
TYPE = {0: "NUMBER", 1: "YES", 2: "NO", 3: "CARD", 4: "TOOL", 5: "ENERGY_CARD",
        6: "ENERGY", 7: "PLAY", 8: "ATTACH", 9: "EVOLVE", 10: "ABILITY",
        11: "DISCARD", 12: "RETREAT", 13: "ATTACK", 14: "END"}
ZONE = {1: "deck", 2: "hand", 3: "discard", 4: "active", 5: "bench", 7: "stadium", 12: "looking"}

try:
    from rl.attack_data import ATTACKS as _ATTACKS
except Exception:
    _ATTACKS = {}


def _cid(c):
    return (c.get("id") if isinstance(c, dict) else c) or 0


def _name(cards, cid):
    return cards.name(cid) if cid else "-"


def _zone_card(cards, s, area, index, player, deck):
    if not isinstance(index, int) or index < 0:
        return 0
    if area == 1:
        arr = deck
    elif area == 7:
        arr = s.get("stadium")
    elif area == 12:
        arr = s.get("looking")
    else:
        zone = {2: "hand", 3: "discard", 4: "active", 5: "bench"}.get(area)
        players = s.get("players") or []
        if zone is None or not (0 <= player < len(players)):
            return 0
        arr = players[player].get(zone)
    if not arr or index >= len(arr):
        return 0
    return _cid(arr[index])


def describe(cards, o, s, me, deck):
    t = o.get("type")
    label = TYPE.get(t, f"OPT{t}")
    # source card the option acts on
    area = o.get("area")
    if area is None and t == 7:                 # PLAY: index is a hand slot
        area = 2
    cid = o.get("cardId") or _zone_card(cards, s, area, o.get("index"), o.get("playerIndex", me), deck)
    src = _name(cards, cid) if cid else ""
    # target (in-play)
    tgt = ""
    ia, ii = o.get("inPlayArea"), o.get("inPlayIndex")
    if ia == 4:
        tgt = " -> active"
    elif ia == 5:
        tgt = f" -> bench[{ii}]"
    extra = ""
    if t == 13:                                  # attack
        a = _ATTACKS.get(o.get("attackId"))
        if a:
            dmg, var, cost, eff = a
            extra = f" ({dmg}{'+' if var else ''} dmg)"
    elif t == 0 and o.get("number") is not None:  # NUMBER (COUNT select): show the chosen value
        extra = f" = {o.get('number')}"
    parts = [label]
    if src:
        parts.append(src)
    return " ".join(parts) + tgt + extra


def render_pk(cards, pk, slot=""):
    if not pk:
        return f"  {slot}(empty)"
    eng = "".join(ENERGY[e] for e in (pk.get("energies") or []) if 0 <= e < len(ENERGY))
    tools = ",".join(_name(cards, _cid(x)) for x in (pk.get("tools") or []))
    s = f"  {slot}{_name(cards, pk.get('id')):16s} HP {pk.get('hp')}/{pk.get('maxHp')}"
    if eng:
        s += f"  E[{eng}]"
    if tools:
        s += f"  tool[{tools}]"
    if pk.get("appearThisTurn"):
        s += "  (new)"
    return s


def render(cards, obs, me):
    s = obs["current"]; pl = s["players"]; opp = 1 - me
    turn = s.get("turn")
    out = [f"\n{'='*60}", f" TURN {turn}   you=P{me}"]
    for who, idx, tag in ((pl[opp], opp, "OPPONENT"), (pl[me], me, "YOU")):
        out.append(f"{tag} (P{idx})  prizes:{len(who.get('prize') or [])}  "
                   f"deck:{who.get('deckCount')}  hand:{who.get('handCount', len(who.get('hand') or []))}  "
                   f"discard:{len(who.get('discard') or [])}")
        act = (who.get("active") or [None])
        out.append(render_pk(cards, act[0] if act else None, "Active: "))
        for i, b in enumerate(who.get("bench") or []):
            out.append(render_pk(cards, b, f"Bench[{i}]: "))
    hand = pl[me].get("hand") or []
    out.append("YOUR HAND: " + (", ".join(_name(cards, _cid(c)) for c in hand) or "(empty)"))
    out.append("-" * 60)
    print("\n".join(out))


def make_ai(ckpt, mode, n_sims, n_det, deck_holder):
    ck = torch.load(ckpt, map_location="cpu")
    arch = ck.get("net_config", {}).get("arch", "mlp")
    cards = get_card_table()
    if arch != "transformer2":
        raise SystemExit(f"checkpoint arch {arch!r} unsupported -- the v1 mlp/transformer nets were "
                         f"removed; use a transformer2 checkpoint")
    from rl.encoding import TokenEncoder, GameTracker, AbilityTracker
    from rl.policy import build_token_net
    from rl.policy import load_compatible
    from rl import search_agent as SA
    enc = TokenEncoder(cards)
    net = build_token_net(cards, ck["net_config"]); load_compatible(net, ck["net"]); net.eval()
    tr, ab = GameTracker(), AbilityTracker()
    wk_on = bool(ck.get("net_config", {}).get("would_ko"))       # net trained with the would_KO feature?

    def ai(obs):
        sel = obs.get("select")
        if sel is None:
            tr.reset(); ab.reset(); return [int(c) for c in deck_holder["deck"]]
        tr.update(obs); ab.note_turn((obs.get("current") or {}).get("turn"))
        if mode == "mcts":
            pick = SA.mcts_select(obs, net, enc, deck_holder["deck"], tr, ab.slots, n_sims=n_sims, n_det=n_det)
        else:                                                    # greedy (annotate would_KO feature iff net trained w/ it)
            if wk_on:
                SA.annotate_would_ko(obs, deck_holder["deck"], enc)
            pick = SA._net_greedy_select(obs, net, enc, deck_holder["deck"], tr, ab.slots)
        ab.record(sel, pick); return pick
    return ai, arch


def human(cards, obs, me, deck):
    sel = obs["select"]
    if sel is None:
        return [int(c) for c in deck]
    render(cards, obs, me)
    opts = sel["option"]; k = sel.get("maxCount", 1)
    print(f"DECISION (type {sel.get('type')}): choose {k} of {len(opts)} option(s)")
    for i, o in enumerate(opts):
        print(f"  [{i}] {describe(cards, o, obs['current'], me, deck)}")
    picked = []
    while len(picked) < k:
        raw = input(f"  pick option# ({len(picked)}/{k} chosen, 'd'=done, 'q'=quit) > ").strip()
        if raw == "q":
            print("bye."); sys.exit(0)
        if raw == "d" and picked:
            break
        if raw.isdigit() and 0 <= int(raw) < len(opts) and int(raw) not in picked:
            picked.append(int(raw))
        else:
            print("    invalid.")
    return sorted(picked)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--mode", choices=["greedy", "mcts"], default="mcts")
    p.add_argument("--n-sims", type=int, default=80)
    p.add_argument("--n-det", type=int, default=2)
    p.add_argument("--your-deck", default=None, help="deck name from rl.decks (default: same as AI)")
    p.add_argument("--ai-deck", default=None, help="deck name (default: first meta deck)")
    p.add_argument("--side", type=int, default=0, choices=[0, 1], help="which player you are")
    p.add_argument("--demo", action="store_true", help="AI vs AI (render only, no input)")
    p.add_argument("--seed", type=int, default=None)
    a = p.parse_args()

    cards = get_card_table()
    ai_deck = DECKS.get(a.ai_deck) if a.ai_deck else list(DECKS.values())[0]
    your_deck = DECKS.get(a.your_deck) if a.your_deck else ai_deck
    ai_hold = {"deck": ai_deck}
    ai, arch = make_ai(a.ckpt, a.mode, a.n_sims, a.n_det, ai_hold)
    print(f"opponent: {arch} {a.mode}" + (f"({a.n_sims}x{a.n_det})" if a.mode == "mcts" else "")
          + f"  | you=P{a.side}  demo={a.demo}", flush=True)

    you = a.side
    demo_ai = make_ai(a.ckpt, "greedy", 0, 0, {"deck": your_deck})[0] if a.demo else None
    rng = random.Random(a.seed)
    try: battle_finish()
    except Exception: pass
    d0 = your_deck if you == 0 else ai_deck
    d1 = ai_deck if you == 0 else your_deck
    obs = battle_start(d0, d1)[0]
    steps = 0
    while obs["current"]["result"] < 0 and steps < 4000:
        cur = obs["current"]; turn_of = cur["yourIndex"]
        if obs.get("select") is None:
            obs = battle_select([int(c) for c in (d0 if turn_of == 0 else d1)]); continue
        if turn_of == you and not a.demo:
            pick = human(cards, obs, you, your_deck)
        elif turn_of == you and a.demo:
            pick = demo_ai(obs)
        else:
            pick = ai(obs)
        obs = battle_select(pick); steps += 1

    r = obs["current"]["result"]
    render(cards, obs, you)
    msg = "DRAW" if r == 2 else ("YOU WIN!" if r == you else "you lose.")
    print(f"\n{'='*60}\n  GAME OVER -> {msg}  (result={r})\n{'='*60}")
    try: battle_finish()
    except Exception: pass


if __name__ == "__main__":
    main()
