"""Head-to-head: a v2 TokenTransformer (encoding/policy2) vs the OLD champion bundle.
The v2 agent pilots with its true deck + GameTracker + AbilityTracker (matches the
submission's inference). Both sides greedy; same deck per matchup; alternate sides.

    PYTHONPATH=. python scripts/h2h_v2.py --new-ckpt <v2.pt> --old-bundle submission_mcts --games 6
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import torch

torch.set_grad_enabled(False)  # inference-only; NB: do NOT use @torch.no_grad() on the
# agent fns — it rewrites their signature to (*args, **kwargs) and kaggle_environments'
# arg-count introspection then calls them with zero args -> TypeError -> silent no-contest.


def _v2_agent(ckpt_path, deck_holder):
    from rl.card_features import get_card_table
    from rl.encoding import build_mask, SUBMIT_ACTION
    from rl.encoding import TokenEncoder, GameTracker, AbilityTracker
    from rl.policy2 import build_token_net
    cards = get_card_table("EN_Card_Data.csv")
    enc = TokenEncoder(cards)
    ck = torch.load(ckpt_path, map_location="cpu")
    net = build_token_net(cards, ck.get("net_config", {})); net.load_state_dict(ck["net"]); net.eval()
    tracker, ability = GameTracker(), AbilityTracker()
    int_keys = enc.int_keys

    def agent(obs):
        sel = obs.get("select")
        if sel is None:
            tracker.reset(); ability.reset()
            return list(deck_holder["deck"])
        ability.note_turn((obs.get("current") or {}).get("turn"))
        tracker.update(obs)
        picked: list[int] = []
        for _ in range(sel.get("maxCount", 1) + 1):
            o = enc.encode(obs, set(picked), self_deck=deck_holder["deck"], tracker=tracker, ability_slots=ability.slots)
            t = {k: torch.as_tensor(np.asarray(v)[None], dtype=(torch.long if k in int_keys else torch.float32))
                 for k, v in o.items()}
            logits, _ = net.logits_value(t)
            ml = logits[0].clone()
            mask = torch.as_tensor(np.asarray(build_mask(sel, set(picked))), dtype=torch.bool)
            if mask.any():
                ml[~mask] = -1e9
            a = int(ml.argmax())
            if a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel.get("maxCount", 1):
                break
        ability.record(sel, picked)
        return sorted(set(picked))
    return agent, ck.get("global_step")


def _old_agent(bundle, deck_holder):
    sys.path.insert(0, bundle)
    import card_features, encoding, policy
    cards = card_features.get_card_table(os.path.join(bundle, "EN_Card_Data.csv"))
    enc = encoding.Encoder(cards)
    ck = torch.load(os.path.join(bundle, "model.pt"), map_location="cpu")
    net = policy.build_net(enc.cf, cards.vocab_size, ck.get("net_config", {})); net.load_state_dict(ck["net"]); net.eval()
    dev = torch.device("cpu")

    def agent(obs):
        sel = obs.get("select")
        if sel is None:
            return list(deck_holder["deck"])
        picked: list[int] = []
        for _ in range(sel.get("maxCount", 1) + 1):
            o = policy.obs_to_tensors(enc.encode(obs, set(picked)), dev)
            o = {k: v[None] for k, v in o.items()}
            logits, _ = net.logits_value(o)
            ml = logits[0].clone()
            mask = torch.as_tensor(np.asarray(encoding.build_mask(sel, set(picked))), dtype=torch.bool)
            if mask.any():
                ml[~mask] = -1e9
            a = int(ml.argmax())
            if a == encoding.SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel.get("maxCount", 1):
                break
        return sorted(set(picked))
    return agent, ck.get("global_step")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--new-ckpt", required=True)
    p.add_argument("--old-bundle", default="submission_mcts")
    p.add_argument("--games", type=int, default=6)
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
    new_agent, n_step = _v2_agent(args.new_ckpt, new_holder)
    old_agent, o_step = _old_agent(args.old_bundle, old_holder)
    print(f"v2 NEW step={n_step}  vs  OLD champion step={o_step}   decks={len(deck_list)}  greedy h2h")

    env = make("cabt", debug=False)
    wins = draws = losses = 0
    for name, deck in deck_list:
        new_holder["deck"] = old_holder["deck"] = deck
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
        print(f"  after {name:16s}: v2 {wins}-{draws}-{losses}  (wr={wins/max(tot,1):.3f})")
    tot = wins + losses + draws
    print(f"\nv2 vs OLD champion: {wins}-{draws}-{losses}  win-rate={wins/max(tot,1):.3f}  "
          f"(excl. draws {wins/max(wins+losses,1):.3f})  over {tot} games")


if __name__ == "__main__":
    main()
