"""Build a behavioral-cloning dataset from PTCG episode replays -> a .npz of stacked
encoded-obs arrays + action labels (WINNING side only).

Mirrors inference-time encoding EXACTLY (rl.search_agent._net_greedy_select): per side a
GameTracker + AbilityTracker; each decision's recorded action (a list of option indices) is
expanded into per-pick rows `enc.encode(obs, picked=set(action[:k]), self_deck, tracker, ability_slots)`
with label=action[k], plus a SUBMIT row (label=SUBMIT_ACTION) when the side submitted before
maxCount. Deck step (select is None, action == 60 ids) resets the trackers and sets self_deck.

  python scripts/build_bc_dataset.py [ep_dir] [out.npz] [max_eps]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from rl.card_features import get_card_table
from rl.encoding import TokenEncoder, GameTracker, AbilityTracker, SUBMIT_ACTION

EP_DIR = sys.argv[1] if len(sys.argv) > 1 else "_kaggle_scout/ep"
OUT = sys.argv[2] if len(sys.argv) > 2 else "_kaggle_scout/bc.npz"
MAX_EPS = int(sys.argv[3]) if len(sys.argv) > 3 else 0

ct = get_card_table()
enc = TokenEncoder(ct)


def rows_from_episode(ep):
    """Yield (encoded_obs_dict, label) for the winning side's decisions."""
    rewards = ep.get("rewards") or []
    if len(rewards) != 2 or rewards[0] is None or rewards[1] is None or rewards[0] == rewards[1]:
        return                                   # draw / malformed -> skip
    win = 0 if rewards[0] > rewards[1] else 1
    steps = ep.get("steps") or []
    tr, ab = GameTracker(), AbilityTracker()
    deck = None
    for st in steps:
        if len(st) <= win:
            continue
        ag = st[win]
        obs = ag.get("observation") or {}
        action = ag.get("action")
        sel = obs.get("select")
        # DECK step = a 60-id action, regardless of whether select is None or SET (some replays
        # present the deck choice as a real select). -> set self_deck, reset trackers, don't clone it.
        if isinstance(action, list) and len(action) == 60:
            deck = [int(c) for c in action]
            tr.reset(); ab.reset()
            continue
        if sel is None:                          # inactive / non-decision step
            continue
        if deck is None or not isinstance(action, list) or not action:
            continue
        opts = sel.get("option") or []
        n = len(opts)
        if not all(isinstance(i, int) and 0 <= i < n for i in action):
            continue                             # malformed action -> skip this decision
        try:
            tr.update(obs)
            ab.note_turn((obs.get("current") or {}).get("turn"))
            maxc = sel.get("maxCount", 1)
            for k, idx in enumerate(action):
                yield enc.encode(obs, set(action[:k]), self_deck=deck, tracker=tr,
                                 ability_slots=ab.slots), int(idx)
            if len(action) < maxc:
                yield enc.encode(obs, set(action), self_deck=deck, tracker=tr,
                                 ability_slots=ab.slots), SUBMIT_ACTION
            ab.record(sel, action)
        except Exception:
            return                               # malformed mid-episode -> drop the rest


def main():
    eps = sorted(glob.glob(os.path.join(EP_DIR, "*.json")))
    if MAX_EPS:
        eps = eps[:MAX_EPS]
    print(f"[bc] {len(eps)} episodes from {EP_DIR}", flush=True)
    rows, labels = [], []
    used = 0
    for i, f in enumerate(eps):
        try:
            ep = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        got = 0
        for row, lab in rows_from_episode(ep):
            rows.append(row); labels.append(lab); got += 1
        used += 1 if got else 0
        if (i + 1) % 25 == 0:
            print(f"[bc]   {i + 1}/{len(eps)} eps -> {len(rows)} rows", flush=True)
    print(f"[bc] {len(rows)} rows from {used} winning sides", flush=True)
    if not rows:
        print("[bc] NO ROWS — check episode format"); return
    int_keys = set(enc.int_keys)
    keys = list(rows[0].keys())
    out = {}
    for k in keys:
        dt = np.int32 if k in int_keys else np.float32
        out[k] = np.stack([r[k] for r in rows]).astype(dt)
    out["__labels__"] = np.array(labels, dtype=np.int64)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    np.savez_compressed(OUT, **out)
    import collections
    lc = collections.Counter(labels)
    print(f"[bc] saved {OUT}: {len(labels)} rows, {len(keys)} keys; "
          f"SUBMIT={lc.get(SUBMIT_ACTION,0)}, n_distinct_labels={len(lc)}", flush=True)


if __name__ == "__main__":
    main()
