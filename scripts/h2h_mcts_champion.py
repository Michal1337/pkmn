"""Faithful MCTS head-to-head: our v1 net (fixed encoding) + rl.search_agent MCTS
vs the TRUE champion (old-encoding net + its own search_agent MCTS), played through
the kaggle cabt env. Both pilot the champion's deck.csv (mirror); sides alternated.

Each side runs its OWN encoding/net/search; the live battle is the kaggle cabt engine
while each MCTS searches on the SDK's separate agent_ptr (same arrangement the champion
uses in its Kaggle deployment, so search never touches the live battle).

    PYTHONPATH=. python scripts/h2h_mcts_champion.py --new-ckpt _tourney/v1_playfix/latest.pt \
        --old-bundle submission_mcts --games 10 --n-sims 40 --n-det 2
"""
from __future__ import annotations
import argparse
import os
import random
import sys

import torch

torch.set_grad_enabled(False)  # inference-only; NEVER @torch.no_grad() on agent fns
# (it rewrites the signature -> kaggle_environments calls with 0 args -> silent draws).


def _v1_mcts_agent(ckpt, deck_holder, n_sims, n_det, seed):
    from rl.card_features import get_card_table
    from rl.encoding import Encoder
    from rl.policy import build_net
    from rl import search_agent as SA
    ct = get_card_table(); enc = Encoder(ct)
    ck = torch.load(ckpt, map_location="cpu")
    net = build_net(enc.cf, ct.vocab_size, ck.get("net_config", {})); net.load_state_dict(ck["net"]); net.eval()
    rng = random.Random(seed)

    def agent(obs):
        return SA.mcts_select(obs, net, enc, deck_holder["deck"], "cpu", n_sims=n_sims, n_det=n_det, rng=rng)
    return agent, ck.get("global_step")


def _champion_mcts_agent(bundle, n_sims, n_det, seed):
    sys.path.insert(0, bundle)
    import card_features, encoding, policy, search_agent
    ct = card_features.get_card_table(os.path.join(bundle, "EN_Card_Data.csv"))
    enc = encoding.Encoder(ct)
    ck = torch.load(os.path.join(bundle, "model.pt"), map_location="cpu")
    net = policy.build_net(enc.cf, ct.vocab_size, ck.get("net_config", {})); net.load_state_dict(ck["net"]); net.eval()
    with open(os.path.join(bundle, "deck.csv")) as f:
        deck = [int(line) for line in f if line.strip()]
    rng = random.Random(seed)

    def agent(obs):
        return search_agent.mcts_select(obs, net, enc, deck, "cpu", n_sims=n_sims, n_det=n_det, rng=rng)
    return agent, deck, ck.get("global_step")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-ckpt", required=True)
    p.add_argument("--old-bundle", default="submission_mcts")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--n-sims", type=int, default=40)
    p.add_argument("--n-det", type=int, default=2)
    args = p.parse_args()

    from kaggle_environments import make

    # champion first (it owns the deck both sides mirror); our v1 shares that deck.
    old_agent, deck, o_step = _champion_mcts_agent(args.old_bundle, args.n_sims, args.n_det, seed=0)
    new_holder = {"deck": deck}
    new_agent, n_step = _v1_mcts_agent(args.new_ckpt, new_holder, args.n_sims, args.n_det, seed=7)
    print(f"v1 NEW step={n_step} MCTS  vs  OLD champion step={o_step} MCTS   "
          f"sims={args.n_sims} det={args.n_det}  deck=champion mirror", flush=True)

    env = make("cabt", debug=False)
    import time
    wins = draws = losses = 0
    t0 = time.time()
    for g in range(args.games):
        new_p0 = (g % 2 == 0)
        agents = [new_agent, old_agent] if new_p0 else [old_agent, new_agent]
        env.reset(); env.run(agents)
        r0, r1 = env.state[0]["reward"], env.state[1]["reward"]
        new_r, opp_r = (r0, r1) if new_p0 else (r1, r0)
        if new_r == opp_r:
            draws += 1
        elif new_r > opp_r:
            wins += 1
        else:
            losses += 1
        tot = wins + losses + draws
        print(f"  g{g}: new={'P0' if new_p0 else 'P1'} -> v1 {wins}-{draws}-{losses} "
              f"(wr={wins/max(tot,1):.3f})  [{(time.time()-t0)/(g+1):.1f}s/g]", flush=True)
    tot = wins + losses + draws
    print(f"\nv1+MCTS vs champion+MCTS: {wins}-{draws}-{losses}  win-rate={wins/max(tot,1):.3f}  "
          f"(excl. draws {wins/max(wins+losses,1):.3f})  over {tot} games", flush=True)


if __name__ == "__main__":
    main()
