"""Microbench: cost of would_ko_flags per attack-decision (gates the engine-sim-feature retrain).
Drives random-legal mega_lucario games and times the would-KO sim at each MAIN-select-with-attacks."""
import random
import statistics
import time

from sdk_cg import game
from rl.card_features import get_card_table
from rl.decks import DECKS
from rl.encoding import TokenEncoder
from rl.search_agent2 import would_ko_flags

enc = TokenEncoder(get_card_table())
deck = [int(c) for c in DECKS["mega_lucario"]]
rng = random.Random(0)
times, n_atk, steps, games = [], 0, 0, 0

for g in range(4):
    obs, start = game.battle_start(deck, deck)
    if obs is None:
        continue
    games += 1
    while obs is not None and obs.get("current", {}).get("result", -1) < 0 and steps < 8000:
        sel = obs.get("select")
        if sel is None:
            break
        if sel.get("type") == 0 and any(o.get("attackId") is not None for o in (sel.get("option") or [])):
            t = time.perf_counter()
            flags = would_ko_flags(obs, deck, enc, n_det=1, rng=rng)
            times.append(time.perf_counter() - t)
            n_atk += 1
            if n_atk <= 3:
                print(f"  sample would_ko @atk-dec {n_atk}: {flags}", flush=True)
        n, k = len(sel["option"]), sel.get("maxCount", 1)
        pick = rng.sample(range(n), min(k, n)) if n else []
        obs = game.battle_select(pick)
        steps += 1
    try:
        game.battle_finish()
    except Exception:
        pass

print(f"games={games} steps={steps} attack-decisions={n_atk}", flush=True)
if times:
    print(f"would_ko_flags: mean={statistics.mean(times)*1000:.1f}ms  p50={statistics.median(times)*1000:.1f}ms  "
          f"max={max(times)*1000:.0f}ms  total={sum(times):.1f}s  (n={len(times)})", flush=True)
    print(f"attack-decisions are ~{steps/max(n_atk,1):.0f} steps apart -> per-step overhead "
          f"~{sum(times)/max(steps,1)*1000:.2f}ms/step", flush=True)
