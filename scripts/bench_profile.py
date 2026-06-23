"""Where does per-step time go? Coarse split (engine vs encode vs tracker) + cProfile
of the v2 encode path. Single-process (compute only, no IPC) -> the relative breakdown
matches a worker. Run locally."""
import cProfile
import io
import pstats
import random
import time

from kaggle_environments.envs.cabt.cg import game
from rl.card_features import get_card_table
from rl.encoding import Encoder
from rl.encoding import TokenEncoder, GameTracker
import rl.decks_generated as dg

ct = get_card_table()
decks = list(dg.GENERATED.values())


def coarse(arch, n_decisions=400):
    enc = TokenEncoder(ct) if arch == "transformer2" else Encoder(ct)
    rng = random.Random(0)
    d0, d1 = rng.choice(decks), rng.choice(decks)
    obs, st = game.battle_start(d0, d1); tk = GameTracker()
    t_eng = t_enc = t_trk = 0.0; dec = 0; nsel = 0
    while dec < n_decisions:
        s = obs["current"]
        if s["result"] >= 0:
            d0, d1 = rng.choice(decks), rng.choice(decks)
            obs, st = game.battle_start(d0, d1); tk = GameTracker(); continue
        sel = obs["current"]; me = sel["yourIndex"]
        if me == 0:
            if arch == "transformer2":
                t0 = time.perf_counter(); tk.update(obs); t_trk += time.perf_counter() - t0
                t0 = time.perf_counter(); enc.encode(obs, tracker=tk, self_deck=d0); t_enc += time.perf_counter() - t0
            else:
                t0 = time.perf_counter(); enc.encode(obs); t_enc += time.perf_counter() - t0
            dec += 1
        opt = obs["select"]; n, k = len(opt["option"]), opt["maxCount"]
        picks = rng.sample(range(n), min(k, n)) if n else []
        t0 = time.perf_counter(); obs = game.battle_select(picks); t_eng += time.perf_counter() - t0; nsel += 1
    try:
        game.battle_finish()
    except Exception:
        pass
    tot = t_eng + t_enc + t_trk
    print(f"[{arch}]  {dec} decisions / {nsel} selects:")
    print(f"   engine battle_select : {t_eng:6.2f}s  {100*t_eng/tot:3.0f}%   {1000*t_eng/nsel:.2f} ms/select")
    print(f"   encode               : {t_enc:6.2f}s  {100*t_enc/tot:3.0f}%   {1000*t_enc/max(dec,1):.2f} ms/decision")
    if arch == "transformer2":
        print(f"   GameTracker.update   : {t_trk:6.2f}s  {100*t_trk/tot:3.0f}%   {1000*t_trk/max(dec,1):.2f} ms/decision")


print("=== coarse split: engine vs encode (single-process) ===")
coarse("mlp")
coarse("transformer2")

print("\n=== cProfile: v2 single-env step loop, top 20 by self-time ===")
env = __import__("rl.env", fromlist=["CabtEnv"]).CabtEnv(
    agent_decks=decks, opponent_decks=decks, encoder=TokenEncoder(ct), v2=True, seed=0)
obs, _ = env.reset()
for _ in range(40):
    m = env.action_masks(); obs, *_ = env.step(random.choice([i for i, x in enumerate(m) if x]))
pr = cProfile.Profile(); pr.enable()
for _ in range(400):
    m = env.action_masks(); obs, *_ = env.step(random.choice([i for i, x in enumerate(m) if x]))
pr.disable(); env.close()
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(20)
print(s.getvalue())
