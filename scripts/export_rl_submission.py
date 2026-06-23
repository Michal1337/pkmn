"""Export a trained checkpoint into a Kaggle submission tarball.

Two backends (both v2 token-transformer; the v1 mlp torch/numpy/mcts backends were removed):
  --backend transformer2 (default): greedy policy -- bundles policy2.py + the raw
      checkpoint (model.pt) and runs torch inference. Requires torch on the runtime.
  --backend mcts2: decision-time PUCT MCTS (search_agent2) on top of the v2 net --
      additionally bundles search_agent/search_agent2 + decks + the sdk_cg forward model.

Archive top-level contents (flat, as Kaggle requires):
    main.py  deck.csv  EN_Card_Data.csv  card_features.py  enc_constants.py
    encoding.py  attack_data.py  policy2.py  buff_data.py  model.pt
    (+ mcts2: search_agent.py  search_agent2.py  decks.py  sdk_cg/)

    python scripts/export_rl_submission.py --ckpt path/to/latest.pt                 # transformer2 (greedy)
    python scripts/export_rl_submission.py --ckpt path/to/latest.pt --backend mcts2 # + MCTS
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RL = os.path.join(ROOT, "rl")

# Kaggle execs main.py with no __file__ but adds the agent dir to sys.path.
_DIR_FINDER = '''\
import os
import sys


def _agent_dir():
    candidates = list(sys.path) + ["/kaggle_simulations/agent", os.getcwd()]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, "deck.csv")):
            return d
    return os.getcwd()


_HERE = _agent_dir()
sys.path.insert(0, _HERE)
'''


TRANSFORMER2_HEAD = '''\
import torch
from card_features import get_card_table
from encoding import SUBMIT_ACTION
from encoding import TokenEncoder, GameTracker, AbilityTracker
from policy2 import build_token_net

_DEVICE = torch.device("cpu")
_CARDS = get_card_table(os.path.join(_agent_dir(), "EN_Card_Data.csv"))
_ENC = TokenEncoder(_CARDS)
_CK = torch.load(os.path.join(_agent_dir(), "model.pt"), map_location="cpu")
_NET = build_token_net(_CARDS, _CK.get("net_config", {}))
_NET.load_state_dict(_CK["net"])
_NET.eval()
# net trained with the engine-sim would_ko feature? -> annotate it at inference too, so opt_attr's
# would_ko column is populated exactly as in training (train==test). Else it would read all-zero.
_WK_ON = bool(_CK.get("net_config", {}).get("would_ko"))
if _WK_ON:
    import search_agent2 as _SA2

# our true 60-card decklist (threaded as self_deck) + per-game reveal/ability memory.
with open(os.path.join(_agent_dir(), "deck.csv")) as f:
    DECK = [int(line) for line in f if line.strip()]
_TRACKER = GameTracker()
_ABILITY = AbilityTracker()


@torch.no_grad()
def _select(obs, picked):
    o = _ENC.encode(obs, set(picked), self_deck=DECK, tracker=_TRACKER,
                    ability_slots=_ABILITY.slots)
    t = {k: torch.as_tensor(v[None], dtype=(torch.long if k in _ENC.int_keys else torch.float32),
                            device=_DEVICE) for k, v in o.items()}
    logits, _ = _NET.logits_value(t)
    return int(logits.argmax(-1).item())


def agent(obs):
    sel = obs.get("select")
    if sel is None:                       # engine asking for the deck == start of a new game
        _TRACKER.reset(); _ABILITY.reset()
        return DECK
    # update memories ONCE per decision obs (NOT per buffered pick) -- exactly what the
    # training env's learner does (decision-obs-only), so reveal/ability memory is train==test.
    _ABILITY.note_turn((obs.get("current") or {}).get("turn"))
    _TRACKER.update(obs)
    if _WK_ON:                            # engine-sim KO per attack option, ONCE per decision (matches env)
        _SA2.annotate_would_ko(obs, DECK, _ENC)
    picked = []
    max_count = sel.get("maxCount", 1)
    for _ in range(max_count + 1):        # buffer single picks into a full selection
        a = _select(obs, picked)
        if a == SUBMIT_ACTION:
            break
        picked.append(a)
        if len(picked) >= max_count:
            break
    _ABILITY.record(sel, picked)          # remember OUR ability picks for later decisions this turn
    return sorted(set(picked))
'''


MCTS2_HEAD = '''\
import random
import torch
from card_features import get_card_table
from encoding import TokenEncoder, GameTracker, AbilityTracker
from policy2 import build_token_net
import search_agent2 as SA2

_CARDS = get_card_table(os.path.join(_agent_dir(), "EN_Card_Data.csv"))
_ENC = TokenEncoder(_CARDS)
_CK = torch.load(os.path.join(_agent_dir(), "model.pt"), map_location="cpu")
_NET = build_token_net(_CARDS, _CK.get("net_config", {}))
_NET.load_state_dict(_CK["net"])
_NET.eval()
_WK_ON = bool(_CK.get("net_config", {}).get("would_ko"))   # annotate would_ko at inference iff trained with it
_RNG = random.Random(0)
_NSIMS = 40
_NDET = 2

with open(os.path.join(_agent_dir(), "deck.csv")) as f:
    DECK = [int(line) for line in f if line.strip()]
_TRACKER = GameTracker()
_ABILITY = AbilityTracker()


def agent(obs):
    sel = obs.get("select")
    if sel is None:                       # start of a new game: reset per-game memory + give deck
        _TRACKER.reset(); _ABILITY.reset()
        return DECK
    _ABILITY.note_turn((obs.get("current") or {}).get("turn"))
    _TRACKER.update(obs)                  # decision-obs-only reveal memory (train==test)
    if _WK_ON:                            # engine-sim KO on the ROOT attack options (train==test)
        SA2.annotate_would_ko(obs, DECK, _ENC)
    pick = SA2.mcts_select(obs, _NET, _ENC, DECK, _TRACKER, _ABILITY.slots,
                           n_sims=_NSIMS, n_det=_NDET, rng=_RNG)
    _ABILITY.record(sel, pick)
    return pick
'''


def copy_module(src_name, dst_path):
    """Copy an rl/ module to the flat bundle, fixing relative imports."""
    with open(os.path.join(RL, src_name), encoding="utf-8") as f:
        code = f.read()
    for m in ("card_features", "enc_constants", "encoding", "attack_data", "buff_data",
              "policy2", "search_agent", "search_agent2", "decks"):
        code = code.replace(f"from .{m} import", f"from {m} import")
    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(code)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--backend", choices=["transformer2", "mcts2"], default="transformer2")
    p.add_argument("--deck", default=os.path.join(ROOT, "agent", "deck.csv"))
    p.add_argument("--csv", default=os.path.join(ROOT, "EN_Card_Data.csv"))
    p.add_argument("--out", default=None)
    p.add_argument("--builddir", default=None)
    p.add_argument("--n-sims", type=int, default=40, help="MCTS sims/move (mcts2 head)")
    p.add_argument("--n-det", type=int, default=2, help="MCTS determinizations/move")
    args = p.parse_args()

    out = args.out or os.path.join(ROOT, f"submission_rl_{args.backend}.tar.gz")
    b = args.builddir or os.path.join(ROOT, f"submission_rl_{args.backend}")
    if os.path.exists(b):
        shutil.rmtree(b)
    os.makedirs(b)

    ck = torch.load(args.ckpt, map_location="cpu")

    # shared code + data (the v2 token encoder: encoding.py imports shape constants from
    # enc_constants.py + per-attack props from attack_data.py + buff tables from buff_data.py).
    copy_module("card_features.py", os.path.join(b, "card_features.py"))
    copy_module("enc_constants.py", os.path.join(b, "enc_constants.py"))
    copy_module("encoding.py", os.path.join(b, "encoding.py"))
    copy_module("attack_data.py", os.path.join(b, "attack_data.py"))
    copy_module("buff_data.py", os.path.join(b, "buff_data.py"))
    copy_module("policy2.py", os.path.join(b, "policy2.py"))
    torch.save(ck, os.path.join(b, "model.pt"))
    shutil.copy(args.deck, os.path.join(b, "deck.csv"))
    shutil.copy(args.csv, os.path.join(b, "EN_Card_Data.csv"))

    wk = bool(ck.get("net_config", {}).get("would_ko"))   # net trained with would_ko -> needs the engine-sim at inference

    def _bundle_engine_sim():
        # search_agent2 (MCTS + would_ko) reuses _determinize/_branchable/_Node from search_agent
        # (net-agnostic primitives); bundle both + the candidate decklists + the sdk_cg forward model.
        copy_module("decks.py", os.path.join(b, "decks.py"))
        copy_module("search_agent.py", os.path.join(b, "search_agent.py"))
        copy_module("search_agent2.py", os.path.join(b, "search_agent2.py"))
        shutil.copytree(os.path.join(ROOT, "sdk_cg"), os.path.join(b, "sdk_cg"),
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    if args.backend == "transformer2":
        if wk:                                            # greedy head annotates would_ko -> needs the sim
            _bundle_engine_sim()
        main_py = _DIR_FINDER + TRANSFORMER2_HEAD          # TRANSFORMER2_HEAD defines agent() itself
    else:  # mcts2: always needs the engine-sim (MCTS + would_ko annotation of the root)
        _bundle_engine_sim()
        main_py = (_DIR_FINDER + MCTS2_HEAD               # MCTS2_HEAD defines agent() itself
                   .replace("_NSIMS = 40", f"_NSIMS = {args.n_sims}")
                   .replace("_NDET = 2", f"_NDET = {args.n_det}"))

    with open(os.path.join(b, "main.py"), "w", encoding="utf-8") as f:
        f.write(main_py)

    with open(os.path.join(b, "deck.csv")) as f:
        n = len([ln for ln in f if ln.strip()])
    if n != 60:
        raise SystemExit(f"ERROR: deck.csv has {n} cards, expected 60.")

    files = sorted(os.listdir(b))
    with tarfile.open(out, "w:gz") as tar:
        for name in files:
            tar.add(os.path.join(b, name), arcname=name)

    size = os.path.getsize(out) / 1e6
    print(f"backend={args.backend}  trained to step {ck.get('global_step')}")
    print(f"wrote {out} ({size:.1f} MB)")
    print("top-level contents:", files)


if __name__ == "__main__":
    main()
