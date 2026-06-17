"""Head-to-head between two agents, each either MCTS (search) or greedy (plain net).

Both play the sample deck (mirror), sides alternated. Reports side-A win-rate.
  --a-ckpt/--a-mode and --b-ckpt/--b-mode define the two agents (mode: mcts|greedy).
"""
import argparse, logging, random, time
logging.disable(logging.CRITICAL)
import torch
from rl.card_features import get_card_table
from rl.encoding import Encoder
from rl.policy import build_net
from rl import search_agent as SA
from rl.policy import load_compatible
from rl.env import load_deck
from sdk_cg.game import battle_start, battle_select, battle_finish

def load(path, enc, ct):
    ck = torch.load(path, map_location="cpu")
    net = build_net(enc.cf, ct.vocab_size, ck.get("net_config", {"emb_dim": 32}))
    skipped = load_compatible(net, ck["net"]); net.eval()
    if skipped:
        print(f"  [warn] {path}: reinit {skipped} (cross-encoding load -> imperfect reference)")
    return net

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-ckpt", required=True); p.add_argument("--a-mode", default="mcts", choices=["mcts", "greedy"])
    p.add_argument("--b-ckpt", required=True); p.add_argument("--b-mode", default="mcts", choices=["mcts", "greedy"])
    p.add_argument("--label", default="")
    p.add_argument("--games", type=int, default=16)
    p.add_argument("--n-sims", type=int, default=40)
    p.add_argument("--n-det", type=int, default=2)
    a = p.parse_args()

    ct = get_card_table(); enc = Encoder(ct)
    nets = {a.a_ckpt: load(a.a_ckpt, enc, ct), a.b_ckpt: load(a.b_ckpt, enc, ct)}
    deck = load_deck(); rng = random.Random(7)

    def act(side_ckpt, side_mode, obs):
        net = nets[side_ckpt]
        if side_mode == "mcts":
            return SA.mcts_select(obs, net, enc, deck, "cpu", n_sims=a.n_sims, n_det=a.n_det, rng=rng)
        return SA._net_greedy_select(obs, net, enc, "cpu")

    w = l = d = 0; t0 = time.time()
    for g in range(a.games):
        a_idx = g % 2                          # alternate who is side A
        try: battle_finish()
        except Exception: pass
        obs = battle_start(deck, deck)[0]; steps = 0
        while obs["current"]["result"] < 0 and steps < 4000:
            sel = obs.get("select")
            if sel is None:
                obs = battle_select([int(c) for c in deck]); continue
            if obs["current"]["yourIndex"] == a_idx:
                pick = act(a.a_ckpt, a.a_mode, obs)
            else:
                pick = act(a.b_ckpt, a.b_mode, obs)
            obs = battle_select(pick); steps += 1
        r = obs["current"]["result"]
        if r == 2: d += 1
        elif r == a_idx: w += 1
        else: l += 1
        print(f"  g{g}: A={'P'+str(a_idx)} r{r} -> A W{w} L{l} D{d}  ({(time.time()-t0)/(g+1):.1f}s/g)", flush=True)
    n = max(1, w + l + d)
    tag = a.label or f"A({a.a_mode}:{a.a_ckpt})  vs  B({a.b_mode}:{a.b_ckpt})"
    print(f"\n{tag} over {a.games}: A winrate {w/n:.3f}  (W{w} L{l} D{d})", flush=True)

if __name__ == "__main__":
    main()
