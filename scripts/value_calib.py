"""Value-head-for-search diagnostic: old champion vs new 11M net.

Question: the new net matches the old net as a PLAIN policy on the ladder (~563 vs
~564), but our PIMC+MCTS amplifies the old net by ~+150 LB and the new net by only
~+44. Hypothesis: the new net's VALUE HEAD is a worse oracle for determinized search
(over-trained / different discard encoding), even though its policy is no worse.

Two metrics, both on ONE shared pool of states (collected once, so the old/new
comparison uses identical inputs):

  M1  value calibration   : does V(state) predict the eventual game outcome?
                            (Pearson/Spearman corr, sign accuracy, MSE)
  M2  search SNR          : at each branchable MAIN decision, determinize N_DET times
                            and value every option from the mover's perspective.
                              signal = std across options of (mean-over-det value)
                              noise  = mean across options of (std-over-det value)
                            SNR = signal / noise. Search can only pick the best option
                            if signal >> noise. Higher SNR -> search helps more.

Usage (run each in its matching module env -- the two nets use DIFFERENT encoders):
  # 1) collect a shared pool (uses the OLD bundle's engine + determinization)
  cd submission_mcts && PYTHONPATH=. python ../scripts/value_calib.py collect \
      --out ../notes/vcalib_pool.pkl --states 80 --ndet 5
  # 2) score each net on that identical pool
  cd submission_mcts && PYTHONPATH=. python ../scripts/value_calib.py eval \
      --mode flat --pool ../notes/vcalib_pool.pkl --ckpt model.pt --out ../notes/vcalib_old.json
  PYTHONPATH=. python scripts/value_calib.py eval \
      --mode rl --pool notes/vcalib_pool.pkl --ckpt ckpt_ppo_gen.pt --out notes/vcalib_new.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pickle
import random

import numpy as np
import torch


# ---------------------------------------------------------------- value helper
def _make_value_fn(mode, ckpt_path):
    """Build (net, encoder, value_fn) for the given module env.
    value_fn(obs_dict, me) -> float in [-1,1] from player `me`'s perspective."""
    if mode == "flat":          # inside submission_mcts/: flat imports, old encoding
        from card_features import get_card_table
        from encoding import Encoder
        from policy import build_net, obs_to_tensors
        csv = "EN_Card_Data.csv"
    else:                        # repo root: rl package, new encoding
        from rl.card_features import get_card_table
        from rl.encoding import Encoder
        from rl.policy import build_net, obs_to_tensors
        csv = "EN_Card_Data.csv"

    cards = get_card_table(csv)
    enc = Encoder(cards)
    ck = torch.load(ckpt_path, map_location="cpu")
    net = build_net(enc.cf, cards.vocab_size, ck.get("net_config", {}))
    net.load_state_dict(ck["net"])
    net.eval()
    dev = torch.device("cpu")

    def value_fn(obs_dict, me):
        cur = obs_dict["current"]
        if cur["result"] >= 0:                       # terminal
            return 0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)
        o = obs_to_tensors(enc.encode(obs_dict), dev)
        o = {k: v[None] for k, v in o.items()}
        with torch.no_grad():
            v = float(net.get_value(o)[0])
        return v if cur["yourIndex"] == me else -v   # net value is for the acting player

    return net, enc, value_fn


# -------------------------------------------------------------------- collect
def collect(args):
    """Play old-net-greedy self-play; at each branchable MAIN single-pick decision,
    record the root obs + determinized children per option + the game outcome."""
    from kaggle_environments import make
    from card_features import get_card_table
    from encoding import Encoder
    from policy import build_net
    from search_agent import _determinize, _net_greedy_select
    from sdk_cg import api

    cards = get_card_table("EN_Card_Data.csv")
    enc = Encoder(cards)
    ck = torch.load(args.ckpt, map_location="cpu")
    net = build_net(enc.cf, cards.vocab_size, ck.get("net_config", {}))
    net.load_state_dict(ck["net"]); net.eval()
    with open("deck.csv") as f:
        DECK = [int(x) for x in f if x.strip()]
    rng = random.Random(args.seed)

    records = []                 # one per branchable decision
    game_of = []                 # game index per record (for outcome labelling)
    cur_game = [0]

    def rec_agent(obs):
        sel = obs.get("select")
        if sel is None:
            return DECK
        branchable = (sel.get("type") == 0 and sel.get("maxCount", 1) == 1
                      and len(sel["option"]) >= 2)
        if branchable and len(records) < args.states:
            me = obs["current"]["yourIndex"]
            oc = api.to_observation_class(obs)
            n_opt = min(len(sel["option"]), args.maxopt)
            per_opt = {i: [] for i in range(n_opt)}
            for _ in range(args.ndet):
                try:
                    root = api.search_begin(oc, **_determinize(obs, DECK, rng, enc))
                except Exception:
                    continue
                for i in range(n_opt):
                    try:
                        child = api.search_step(root.searchId, [i])
                        per_opt[i].append(dataclasses.asdict(child.observation))
                    except Exception:
                        pass
                try:
                    api.search_release(root.searchId)
                except Exception:
                    pass
            try:
                api.search_end()
            except Exception:
                pass
            # keep only options that produced >=2 determinized children (for variance)
            per_opt = {i: ch for i, ch in per_opt.items() if len(ch) >= 2}
            if len(per_opt) >= 2:
                records.append({"root": obs, "me": me, "options": per_opt})
                game_of.append(cur_game[0])
        return _net_greedy_select(obs, net, enc, "cpu")

    env = make("cabt", debug=False)
    n_games = 0
    while len(records) < args.states and n_games < args.maxgames:
        start = len(records)
        env.reset()
        env.run([rec_agent, rec_agent])
        r0 = env.state[0]["reward"]
        res = 2 if r0 == env.state[1]["reward"] else (0 if r0 > env.state[1]["reward"] else 1)
        for k in range(start, len(records)):     # label this game's records
            records[k]["outcome_me"] = (0.0 if res == 2
                                        else (1.0 if res == records[k]["me"] else -1.0))
        cur_game[0] += 1
        n_games += 1
        print(f"  game {n_games}: total recorded decisions = {len(records)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(records, f)
    n_child = sum(len(ch) for r in records for ch in r["options"].values())
    print(f"wrote {args.out}: {len(records)} decisions, {n_child} child states, "
          f"{n_games} games")


# ----------------------------------------------------------------------- eval
def eval_net(args):
    net, enc, value_fn = _make_value_fn(args.mode, args.ckpt)
    with open(args.pool, "rb") as f:
        records = pickle.load(f)

    # M1: value of the root vs eventual outcome
    v_root, y = [], []
    for r in records:
        v_root.append(value_fn(r["root"], r["me"]))
        y.append(r["outcome_me"])
    v_root = np.array(v_root); y = np.array(y)
    nz = y != 0.0                                 # drop draws for sign accuracy
    sign_acc = float(np.mean(np.sign(v_root[nz]) == np.sign(y[nz]))) if nz.any() else float("nan")
    pear = float(np.corrcoef(v_root, y)[0, 1]) if len(y) > 2 else float("nan")
    # Spearman via rank correlation
    rv = np.argsort(np.argsort(v_root)); ry = np.argsort(np.argsort(y))
    spear = float(np.corrcoef(rv, ry)[0, 1]) if len(y) > 2 else float("nan")
    mse = float(np.mean((v_root - y) ** 2))

    # M2: per-decision search SNR
    snrs, signals, noises = [], [], []
    for r in records:
        me = r["me"]
        opt_means, opt_stds = [], []
        for i, children in r["options"].items():
            vals = np.array([value_fn(c, me) for c in children])
            opt_means.append(vals.mean()); opt_stds.append(vals.std())
        opt_means = np.array(opt_means); opt_stds = np.array(opt_stds)
        signal = float(opt_means.std())          # spread across options (between-option)
        noise = float(opt_stds.mean())           # determinization jitter (within-option)
        signals.append(signal); noises.append(noise)
        if noise > 1e-9:
            snrs.append(signal / noise)
    out = {
        "mode": args.mode, "ckpt": args.ckpt, "n_decisions": len(records),
        "M1_sign_acc": sign_acc, "M1_pearson": pear, "M1_spearman": spear, "M1_mse": mse,
        "M1_mean_abs_root_value": float(np.mean(np.abs(v_root))),
        "M2_mean_snr": float(np.mean(snrs)) if snrs else float("nan"),
        "M2_median_snr": float(np.median(snrs)) if snrs else float("nan"),
        "M2_mean_signal": float(np.mean(signals)),
        "M2_mean_noise": float(np.mean(noises)),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def priors_eval(args):
    """M3: the POLICY-PRIOR probe (what PUCT actually leans on at n_sims=40).
    For each branchable root, softmax the net's option logits -> prior over options.
    A peaky prior + 40 sims => search barely explores. Also: does the prior put mass
    on the value-good option (the determinized-mean-best one)?"""
    if args.mode == "flat":
        from encoding import build_mask
        from policy import obs_to_tensors
    else:
        from rl.encoding import build_mask
        from rl.policy import obs_to_tensors
    net, enc, value_fn = _make_value_fn(args.mode, args.ckpt)
    dev = torch.device("cpu")
    with open(args.pool, "rb") as f:
        records = pickle.load(f)

    norm_ents, top1s, perps, agrees, vbest_ranks = [], [], [], [], []
    for r in records:
        root = r["root"]; sel = root["select"]; me = r["me"]
        opt_idx = sorted(r["options"].keys())
        if len(opt_idx) < 2:
            continue
        o = obs_to_tensors(enc.encode(root), dev)
        o = {k: v[None] for k, v in o.items()}
        with torch.no_grad():
            logits, _ = net.logits_value(o)
        ml = logits[0].clone()
        mask = torch.as_tensor(np.asarray(build_mask(sel, set())), dtype=torch.bool, device=ml.device)
        ml[~mask] = -1e9
        p_full = torch.softmax(ml, dim=0).cpu().numpy()
        pr = np.array([p_full[i] for i in opt_idx], dtype=np.float64)
        s = pr.sum()
        if s <= 0:
            continue
        pr = pr / s                                  # prior over the surviving options
        ent = -np.sum(pr * np.log(pr + 1e-12))
        norm_ents.append(ent / np.log(len(opt_idx)))  # 1.0 = uniform, 0 = one-hot
        top1s.append(float(pr.max()))
        perps.append(float(np.exp(ent)))              # effective # of options explored
        # value-oracle best option (determinized mean), and where the prior ranks it
        vmean = {i: float(np.mean([value_fn(c, me) for c in r["options"][i]])) for i in opt_idx}
        vbest = max(opt_idx, key=lambda i: vmean[i])
        prior_best = opt_idx[int(pr.argmax())]
        agrees.append(prior_best == vbest)
        order = sorted(range(len(opt_idx)), key=lambda k: -pr[k])  # options by prior desc
        vbest_ranks.append(order.index(opt_idx.index(vbest)))      # 0 = prior's top pick

    out = {
        "mode": args.mode, "ckpt": args.ckpt, "n_decisions": len(norm_ents),
        "M3_mean_norm_entropy": float(np.mean(norm_ents)),   # lower = peakier prior
        "M3_mean_top1_mass": float(np.mean(top1s)),          # higher = peakier
        "M3_mean_perplexity": float(np.mean(perps)),         # ~effective options searched
        "M3_mean_n_options": float(np.mean([len(sorted(r["options"].keys())) for r in records
                                            if len(r["options"]) >= 2])),
        "M3_prior_top1_eq_valuebest": float(np.mean(agrees)),  # prior agrees w/ value-best
        "M3_mean_valuebest_rank_under_prior": float(np.mean(vbest_ranks)),  # 0 = prior's #1
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--ckpt", default="model.pt")
    c.add_argument("--out", default="../notes/vcalib_pool.pkl")
    c.add_argument("--states", type=int, default=80)
    c.add_argument("--ndet", type=int, default=5)
    c.add_argument("--maxopt", type=int, default=16)
    c.add_argument("--maxgames", type=int, default=60)
    c.add_argument("--seed", type=int, default=12345)
    c.set_defaults(func=collect)

    e = sub.add_parser("eval")
    e.add_argument("--mode", choices=["flat", "rl"], required=True)
    e.add_argument("--pool", required=True)
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.set_defaults(func=eval_net)

    pr = sub.add_parser("priors")
    pr.add_argument("--mode", choices=["flat", "rl"], required=True)
    pr.add_argument("--pool", required=True)
    pr.add_argument("--ckpt", required=True)
    pr.add_argument("--out", required=True)
    pr.set_defaults(func=priors_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
