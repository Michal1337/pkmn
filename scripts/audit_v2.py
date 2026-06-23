"""Audit the v2 (encoding/policy) encoding + action space on REAL battle obs.

Plays greedy self-play games (dual per-side trackers, as in training/inference) and at
EVERY decision checks a battery of invariants, accumulating any violations. Targets both
generic correctness (shapes/dtypes/finiteness/id-range/mask/determinism/privacy) and the
specific things we've fixed/worried about (PLAY card-blindness, option src/tgt resolution,
tool capture, picked-set masking, net masked-argmax legality).

  PYTHONPATH=. python scripts/audit_v2.py --ckpt _tourney/v2_small/latest.pt --games 6
"""
from __future__ import annotations
import argparse
import collections
import random

import numpy as np
import torch

torch.set_grad_enabled(False)

from rl.card_features import get_card_table
from rl.encoding import build_mask, SUBMIT_ACTION, MAX_OPTIONS, N_ACTIONS
from rl.encoding import TokenEncoder, GameTracker, AbilityTracker, N_BENCH, MAX_HAND, UNIT_ATTR, _card_id
from rl.policy import build_token_net
from rl import search_agent as SA2
from rl.decks import DECKS
try:
    from rl.decks_generated import GENERATED
except Exception:
    GENERATED = {}
from sdk_cg.game import battle_start, battle_select, battle_finish

FAILS: list[str] = []
STATS = collections.Counter()
OPT_TYPES = collections.Counter()


def fail(msg):
    STATS["fails"] += 1
    if len(FAILS) < 40:
        FAILS.append(msg)


def audit(obs, enc, net, deck, tr, ab):
    sel = obs["select"]; s = obs["current"]; me = s["yourIndex"]; opp = 1 - me
    n_opt = len(sel["option"]); nleg = min(n_opt, MAX_OPTIONS)
    STATS["decisions"] += 1
    for op in sel["option"]:
        OPT_TYPES[op.get("type")] += 1
    o = enc.encode(obs, set(), self_deck=deck, tracker=tr, ability_slots=ab.slots)
    sh, ik, UNK = enc.shapes, enc.int_keys, enc.UNK

    # 1. shape / key / dtype contract
    for k, shp in sh.items():
        if k not in o:
            fail(f"missing key {k}"); continue
        if tuple(o[k].shape) != tuple(shp):
            fail(f"{k} shape {tuple(o[k].shape)} != {tuple(shp)}")
        if k in ik and o[k].dtype != np.int64:
            fail(f"{k} dtype {o[k].dtype} != int64")
    for k in o:
        if k not in sh:
            fail(f"extra key {k}")

    # 2. finiteness + id range
    for k, v in o.items():
        if k in ik:
            if v.size and (v.min() < 0 or v.max() > UNK):
                fail(f"{k} id out of [0,{UNK}] ({v.min()}..{v.max()})")
        elif not np.isfinite(v).all():
            fail(f"{k} has NaN/inf")

    # 3. action_mask == build_mask, and legality of option slots
    bm = np.asarray(build_mask(sel, set()), np.float32)
    if not np.array_equal(o["action_mask"], bm):
        fail("action_mask != build_mask(sel, {})")
    for i in range(MAX_OPTIONS):
        if (o["action_mask"][i] > 0.5) != (i < nleg):
            fail(f"mask[{i}] legality wrong (n_opt={n_opt})"); break

    # 4. padded option slots carry no token ref / no attr
    for i in range(nleg, MAX_OPTIONS):
        if o["opt_attr"][i].any() or o["opt_src_pos"][i] >= 0 or o["opt_tgt_pos"][i] >= 0:
            fail(f"pad option slot {i} not zero"); break

    # 5. option src/tgt token POSITIONS are valid (index into the pre-encoder state seq; -1 = none)
    from rl.encoding import N_STATE_TOKENS
    for i, op in enumerate(sel["option"][:MAX_OPTIONS]):
        for key in ("opt_src_pos", "opt_tgt_pos"):
            pos = int(o[key][i])
            if pos < -1 or pos >= N_STATE_TOKENS:
                fail(f"{key}[{i}]={pos} out of [-1,{N_STATE_TOKENS}) (type {op.get('type')})"); break

    # 6. PLAY (type 7) card-blindness regression: a PLAY option must point at its hand-card token
    for i, op in enumerate(sel["option"][:MAX_OPTIONS]):
        if op.get("type") == 7:
            STATS["play_opts"] += 1
            if int(o["opt_src_pos"][i]) < 0:
                fail(f"PLAY opt[{i}] card-blind (src_pos=-1)")
            break

    # 7. unit streams match the board + tool capture
    for side, key in ((me, "self"), (opp, "opp")):
        pls = s["players"][side]; act = (pls.get("active") or [None])[0]
        top, um = o[f"{key}_unit_top_id"], o[f"{key}_unit_mask"]
        if int(top[0]) != min(_card_id(act), UNK):
            fail(f"{key} active top {int(top[0])} != {min(_card_id(act), UNK)}")
        if (um[0] > 0.5) != (act is not None):
            fail(f"{key} active mask mismatch")
        for j, b in enumerate((pls.get("bench") or [])[:N_BENCH]):
            if int(top[1 + j]) != min(_card_id(b), UNK):
                fail(f"{key} bench[{j}] top mismatch"); break
        for slot, pk in enumerate([act] + list(pls.get("bench") or [])[:N_BENCH]):
            if pk and pk.get("tools"):
                STATS["tool_units"] += 1
                if not o[f"{key}_unit_tool_id"][slot].any():
                    fail(f"{key} unit[{slot}] has tools but tool_id all 0")
                break

    # 8. hand privacy: self hand matches; opp hand only EMPTY/UNK (no leak)
    hand = s["players"][me].get("hand") or []
    for j, c in enumerate(hand[:MAX_HAND]):
        if int(o["self_hand_id"][j]) != min(_card_id(c), UNK):
            fail(f"self_hand[{j}] mismatch"); break
    oh = o["opp_hand_id"]
    if ((oh != 0) & (oh != UNK)).any():
        fail("opp_hand leaks a real card id")

    # 9. determinism (same inputs -> identical output)
    o2 = enc.encode(obs, set(), self_deck=deck, tracker=tr, ability_slots=ab.slots)
    for k in o:
        if not np.array_equal(o[k], o2[k]):
            fail(f"non-deterministic key {k}"); break

    # 10. picked-set masks the picked option
    if n_opt > 1:
        o3 = enc.encode(obs, {0}, self_deck=deck, tracker=tr, ability_slots=ab.slots)
        if o3["action_mask"][0] > 0.5:
            fail("picked option 0 still legal after picked={0}")

    # 11. net forward -> finite, shape, masked-argmax legal
    t = {k: torch.as_tensor(np.asarray(v)[None], dtype=(torch.long if k in ik else torch.float32))
         for k, v in o.items()}
    logits, value = net.logits_value(t)
    if tuple(logits.shape) != (1, N_ACTIONS):
        fail(f"logits shape {tuple(logits.shape)} != (1,{N_ACTIONS})")
    if not torch.isfinite(logits).all() or not torch.isfinite(value).all():
        fail("net logits/value NaN/inf")
    ml = logits[0].clone(); ml[~torch.as_tensor(bm, dtype=torch.bool)] = -1e9
    a = int(ml.argmax())
    if not (a < nleg or a == SUBMIT_ACTION):
        fail(f"masked argmax {a} illegal (n_opt={n_opt})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--games", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    ct = get_card_table(); enc = TokenEncoder(ct)
    ck = torch.load(a.ckpt, map_location="cpu")
    net = build_token_net(ct, ck["net_config"]); net.load_state_dict(ck["net"]); net.eval()
    pool = list(DECKS.values()) + list(GENERATED.values())
    rng = random.Random(a.seed)
    print(f"auditing v2 encode+action-space over {a.games} games ({len(pool)} decks)...", flush=True)
    for g in range(a.games):
        decks = {0: rng.choice(pool), 1: rng.choice(pool)}
        tr = {0: GameTracker(), 1: GameTracker()}; ab = {0: AbilityTracker(), 1: AbilityTracker()}
        try: battle_finish()
        except Exception: pass
        obs = battle_start(decks[0], decks[1])[0]; steps = 0
        while obs["current"]["result"] < 0 and steps < 4000:
            sel = obs.get("select"); pp = obs["current"]["yourIndex"]
            if sel is None:
                obs = battle_select([int(c) for c in decks[pp]]); continue
            tr[pp].update(obs); ab[pp].note_turn((obs.get("current") or {}).get("turn"))
            audit(obs, enc, net, decks[pp], tr[pp], ab[pp])
            pick = SA2._net_greedy_select(obs, net, enc, decks[pp], tr[pp], ab[pp].slots)
            ab[pp].record(sel, pick)
            obs = battle_select(pick); steps += 1
        print(f"  game {g+1}/{a.games} done ({STATS['decisions']} decisions audited)", flush=True)
    try: battle_finish()
    except Exception: pass

    print("\n===== AUDIT SUMMARY =====")
    print(f"decisions audited : {STATS['decisions']}")
    print(f"option types seen : {dict(sorted(OPT_TYPES.items()))}")
    print(f"PLAY options      : {STATS['play_opts']}  | units-with-tools seen: {STATS['tool_units']}")
    print(f"total failures    : {STATS['fails']}")
    if FAILS:
        print("---- violations (first 40) ----")
        for f in FAILS:
            print("  FAIL:", f)
    else:
        print("ALL INVARIANTS PASS ✓")


if __name__ == "__main__":
    main()
