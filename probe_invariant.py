import logging; logging.disable(logging.CRITICAL)
import random
from kaggle_environments.envs.cabt.cg.game import battle_start, battle_select
from kaggle_environments.envs.cabt.cabt import deck

violations = {"min_gt_n": 0, "max_gt_n": 0, "n0_min_pos": 0, "min_gt_max": 0}
total_decisions = 0
shapes = {}

def pick(sel):
    # buffer-style: pick minimal legal then submit, mimicking env semantics loosely
    opts = sel["option"]
    n = len(opts)
    mn = sel.get("minCount", 1)
    mx = sel.get("maxCount", 1)
    k = min(mx, n)
    idxs = list(range(k))
    return idxs

random.seed(0)
games = 0
import itertools
for g in range(4000):
    try:
        obs, sd = battle_start(deck, deck)
    except Exception:
        continue
    games += 1
    steps = 0
    while obs is not None and steps < 2000:
        sel = obs.get("select")
        if sel is None:
            break
        cur = obs.get("current", {})
        if cur.get("result", -1) != -1:
            break
        n = len(sel.get("option", []))
        mn = sel.get("minCount", 1)
        mx = sel.get("maxCount", 1)
        total_decisions += 1
        shapes[(mn, mx)] = shapes.get((mn, mx), 0) + 1
        if mn > n:
            violations["min_gt_n"] += 1
        if mx > n:
            violations["max_gt_n"] += 1
        if n == 0 and mn > 0:
            violations["n0_min_pos"] += 1
        if mn > mx:
            violations["min_gt_max"] += 1
        if n == 0:
            # nothing to pick; engine should handle, try empty submit
            try:
                obs = battle_select([])
            except Exception:
                break
            steps += 1
            continue
        try:
            obs = battle_select(pick(sel))
        except Exception:
            break
        steps += 1

print("games_played:", games)
print("total_decisions:", total_decisions)
print("violations:", violations)
print("shapes (minCount,maxCount):", dict(sorted(shapes.items())))
