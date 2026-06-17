"""Trusted evaluation: cross-deck, vs a FROZEN reference, with significance.

The mirror lies (proved repeatedly). This plays a CANDIDATE agent against a frozen
REFERENCE across the deck pool: the candidate plays its own deck (our submission deck)
while the reference cycles through every opponent archetype -- the real ladder scenario.
Each agent is either 'mcts' (search + PIMC inference) or 'greedy' (plain net).

  python scripts/eval_suite.py --cand-ckpt ckpt_az.pt --cand-mode mcts \
      --ref-ckpt ckpt_baseline.pt --ref-mode mcts --games 40

Reports per-opponent and aggregate candidate win-rate with a normal-approx 95% CI.
"""
import argparse, logging, math, random, time
logging.disable(logging.CRITICAL)
import torch
from rl.card_features import get_card_table
from rl.encoding import Encoder
from rl.policy import build_net
from rl import search_agent as SA
from rl.decks import DECKS
from rl.env import load_deck
from sdk_cg.game import battle_start, battle_select, battle_finish

def load(path, enc, ct):
    from rl.policy import load_compatible
    ck = torch.load(path, map_location="cpu")
    net = build_net(enc.cf, ct.vocab_size, ck.get("net_config", {"emb_dim": 32}))
    skipped = load_compatible(net, ck["net"]); net.eval()
    if skipped:
        print(f"  [warn] {path}: reinit {skipped} (cross-encoding load -> imperfect reference)")
    return net

def ci95(w, n):
    if n == 0: return 0.0, 0.0
    p = w / n; se = math.sqrt(p * (1 - p) / n)
    return max(0, p - 1.96 * se), min(1, p + 1.96 * se)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand-ckpt", required=True); ap.add_argument("--cand-mode", default="mcts", choices=["mcts", "greedy"])
    ap.add_argument("--ref-ckpt", required=True);  ap.add_argument("--ref-mode", default="mcts", choices=["mcts", "greedy"])
    ap.add_argument("--cand-deck", default="sample", help="deck the candidate plays (sample|<archetype>)")
    ap.add_argument("--opp-decks", default="sample,mega_abomasnow,dragapult,iono,mega_lucario")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--n-sims", type=int, default=40); ap.add_argument("--n-det", type=int, default=2)
    a = ap.parse_args()

    ct = get_card_table(); enc = Encoder(ct)
    nets = {a.cand_ckpt: load(a.cand_ckpt, enc, ct), a.ref_ckpt: load(a.ref_ckpt, enc, ct)}
    getdeck = lambda name: load_deck() if name == "sample" else DECKS[name]
    cand_deck = getdeck(a.cand_deck)
    rng = random.Random(7)

    def act(ckpt, mode, deck, obs):
        net = nets[ckpt]
        if mode == "mcts":
            return SA.mcts_select(obs, net, enc, deck, "cpu", n_sims=a.n_sims, n_det=a.n_det, rng=rng)
        return SA._net_greedy_select(obs, net, enc, "cpu")

    def play(opp_deck, cand_idx):
        d0, d1 = (cand_deck, opp_deck) if cand_idx == 0 else (opp_deck, cand_deck)
        try: battle_finish()
        except Exception: pass
        obs = battle_start(d0, d1)[0]; steps = 0
        while obs["current"]["result"] < 0 and steps < 4000:
            sel = obs.get("select")
            if sel is None:
                deck = cand_deck if obs["current"]["yourIndex"] == cand_idx else opp_deck
                obs = battle_select([int(c) for c in deck]); continue
            if obs["current"]["yourIndex"] == cand_idx:
                pick = act(a.cand_ckpt, a.cand_mode, cand_deck, obs)
            else:
                pick = act(a.ref_ckpt, a.ref_mode, opp_deck, obs)
            obs = battle_select(pick); steps += 1
        return obs["current"]["result"], cand_idx

    print(f"CAND {a.cand_mode}:{a.cand_ckpt} (deck={a.cand_deck})  vs  REF {a.ref_mode}:{a.ref_ckpt}", flush=True)
    t0 = time.time(); tot_w = tot_n = 0
    for name in a.opp_decks.split(","):
        opp = getdeck(name); w = l = d = 0
        for g in range(a.games):
            r, ci = play(opp, g % 2)
            if r == 2: d += 1
            elif r == ci: w += 1
            else: l += 1
        n = w + l + d; lo, hi = ci95(w, n); tot_w += w; tot_n += n
        print(f"  vs {name:15s}: W{w} L{l} D{d}  winrate {w/max(1,n):.3f}  95%CI[{lo:.2f},{hi:.2f}]  ({(time.time()-t0)/max(1,tot_n):.1f}s/g)", flush=True)
    lo, hi = ci95(tot_w, tot_n)
    print(f"\nAGGREGATE candidate winrate {tot_w/max(1,tot_n):.3f}  95%CI[{lo:.2f},{hi:.2f}]  ({tot_w}/{tot_n})", flush=True)

if __name__ == "__main__":
    main()
