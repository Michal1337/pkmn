"""Head-to-head: a NEW-encoding net vs the OLD 4M champion, each piloting through its
OWN encoder (they ship different encodings). Greedy play (no search) on purpose -- we
want to measure the POLICY difference the encoding fix targets; MCTS would bail out both
sides equally and hide it.

Both nets coexist in one process because the module paths differ:
  NEW -> rl.encoding / rl.policy / rl.card_features   (the package, HEAD encoding)
  OLD -> flat encoding / policy / card_features        (from submission_mcts/, old encoding)

Each game both players pilot the SAME deck (controlled matchup); we alternate who is
player 0 and count the NEW net's win rate over a deck list.

  PYTHONPATH=. python scripts/h2h_vs_champion.py --new-ckpt <new4M.pt> \
      --old-bundle submission_mcts --games 8
HONEST CAVEAT: a local h2h is NOT the leaderboard (only a submission settles that). It's
the cleanest no-submission read, and unlike mirror-Elo it's a direct strength comparison.
"""
from __future__ import annotations
import argparse, os, sys, random
import numpy as np, torch


def _greedy_factory(net, enc, build_mask, submit_action, obs_to_tensors, deck_holder):
    """Return an agent(obs)->list[int] that plays greedily with this net+encoder."""
    dev = torch.device("cpu")
    def agent(obs):
        sel = obs.get("select")
        if sel is None:
            return list(deck_holder["deck"])
        picked: list[int] = []
        for _ in range(sel.get("maxCount", 1) + 1):
            o = obs_to_tensors(enc.encode(obs, set(picked)), dev)
            o = {k: v[None] for k, v in o.items()}
            with torch.no_grad():
                logits, _ = net.logits_value(o)
            ml = logits[0].clone()
            mask = torch.as_tensor(np.asarray(build_mask(sel, set(picked))), dtype=torch.bool)
            if mask.any():
                ml[~mask] = -1e9
            a = int(ml.argmax())
            if a == submit_action:
                break
            picked.append(a)
            if len(picked) >= sel.get("maxCount", 1):
                break
        return sorted(set(picked))
    return agent


def _load_new(ckpt_path):
    from rl.card_features import get_card_table
    from rl.encoding import Encoder, SUBMIT_ACTION, build_mask
    from rl.policy import build_net, obs_to_tensors
    cards = get_card_table("EN_Card_Data.csv")
    enc = Encoder(cards)
    ck = torch.load(ckpt_path, map_location="cpu")
    net = build_net(enc.cf, cards.vocab_size, ck.get("net_config", {})); net.load_state_dict(ck["net"]); net.eval()
    return net, enc, build_mask, SUBMIT_ACTION, obs_to_tensors, ck.get("global_step")


def _load_old(bundle):
    sys.path.insert(0, bundle)                    # flat: encoding/policy/card_features
    import card_features, encoding, policy
    cards = card_features.get_card_table(os.path.join(bundle, "EN_Card_Data.csv"))
    enc = encoding.Encoder(cards)
    ck = torch.load(os.path.join(bundle, "model.pt"), map_location="cpu")
    net = policy.build_net(enc.cf, cards.vocab_size, ck.get("net_config", {})); net.load_state_dict(ck["net"]); net.eval()
    return net, enc, encoding.build_mask, encoding.SUBMIT_ACTION, policy.obs_to_tensors, ck.get("global_step")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-ckpt", required=True)
    p.add_argument("--old-bundle", default="submission_mcts")
    p.add_argument("--games", type=int, default=8, help="games per deck per side-assignment")
    p.add_argument("--decks", default="meta", choices=["meta", "meta+gen"])
    args = p.parse_args()

    from kaggle_environments import make
    from rl.decks import DECKS
    from rl.train import load_deck
    deck_list = list(DECKS.items()) + [("sample", load_deck())]
    if args.decks == "meta+gen":
        from rl.decks_generated import GENERATED
        deck_list += list(GENERATED.items())[:10]

    new_holder, old_holder = {"deck": None}, {"deck": None}
    n_net, n_enc, n_bm, n_sub, n_o2t, n_step = _load_new(args.new_ckpt)
    o_net, o_enc, o_bm, o_sub, o_o2t, o_step = _load_old(args.old_bundle)
    new_agent = _greedy_factory(n_net, n_enc, n_bm, n_sub, n_o2t, new_holder)
    old_agent = _greedy_factory(o_net, o_enc, o_bm, o_sub, o_o2t, old_holder)
    print(f"NEW step={n_step}  vs  OLD champion step={o_step}   decks={len(deck_list)}  greedy h2h")

    env = make("cabt", debug=False)
    wins = draws = losses = 0
    for name, deck in deck_list:
        new_holder["deck"] = old_holder["deck"] = deck
        for g in range(args.games):
            new_p0 = (g % 2 == 0)                 # alternate sides
            agents = [new_agent, old_agent] if new_p0 else [old_agent, new_agent]
            env.reset(); env.run(agents)
            r0, r1 = env.state[0]["reward"], env.state[1]["reward"]
            new_r = r0 if new_p0 else r1
            opp_r = r1 if new_p0 else r0
            if new_r == opp_r: draws += 1
            elif new_r > opp_r: wins += 1
            else: losses += 1
        tot = wins + losses + draws
        print(f"  after {name:16s}: NEW {wins}-{draws}-{losses}  (wr={wins/max(tot,1):.3f})")
    tot = wins + losses + draws
    print(f"\nNEW vs OLD champion: {wins}-{draws}-{losses}  win-rate={wins/max(tot,1):.3f}  "
          f"(excl. draws {wins/max(wins+losses,1):.3f})  over {tot} games")


if __name__ == "__main__":
    main()
