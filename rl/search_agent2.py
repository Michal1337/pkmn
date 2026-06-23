"""v2 (token-transformer) decision-time PUCT MCTS — the v2 counterpart of search_agent.py.

Drives the encoding/policy2 net. The v2 encoder is STATEFUL (it takes our true deck + a
GameTracker of opponent reveals + this-turn ability slots), but MCTS evaluates hypothetical
DETERMINIZED futures. We thread the agent's deck + a FROZEN root tracker/ability snapshot
through every leaf encode: the determinization already fills hidden info, and exact per-node
tracker state along hypothetical lines isn't worth the cost. Net-agnostic pieces
(determinization, branchable test, node) are reused from search_agent.

Search runs on the SDK's separate agent_ptr simulator (api.search_*), never the live battle.
"""
from __future__ import annotations

import dataclasses
import random

import numpy as np
import torch

from .encoding import SUBMIT_ACTION
from .encoding import N_ACTIONS
from .search_agent import _determinize, _branchable, _Node   # net-agnostic helpers

try:    # per-attack is_variable flag for variable-aware would_KO sampling (bundle-safe)
    from .attack_data import ATTACKS as _WK_ATTACKS
except Exception:
    _WK_ATTACKS = {}

WK_NDET_VAR = 6     # determinizations for a VARIABLE-damage attack (coin/conditional) -> KO probability;
                    # fixed-damage attacks are deterministic given the visible board -> 1 sim is exact.


def _tens(enc, enc_obs):
    return {k: torch.as_tensor(np.asarray(v)[None],
                               dtype=(torch.long if k in enc.int_keys else torch.float32))
            for k, v in enc_obs.items()}


@torch.no_grad()
def _net_greedy_select(obs, net, enc, deck, tracker, ability):
    """Full engine selection from the v2 net (buffered single-pick), no search."""
    sel = obs["select"]; picked: list[int] = []
    for _ in range(sel.get("maxCount", 1) + 1):
        o = _tens(enc, enc.encode(obs, set(picked), self_deck=deck, tracker=tracker, ability_slots=ability))
        a = int(net.logits_value(o)[0][0].argmax())
        if a == SUBMIT_ACTION:
            break
        picked.append(a)
        if len(picked) >= sel.get("maxCount", 1):
            break
    return sorted(set(picked))


@torch.no_grad()
def _value(net, enc, obs_dict, me, deck, tracker, ability) -> float:
    cur = obs_dict["current"]
    if cur["result"] >= 0:                                   # terminal
        return 0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)
    o = _tens(enc, enc.encode(obs_dict, set(), self_deck=deck, tracker=tracker, ability_slots=ability))
    v = float(net.get_value(o)[0])
    return v if cur["yourIndex"] == me else -v               # net value is for the acting player


@torch.no_grad()
def _priors_value(net, enc, obs_dict, me, deck, tracker, ability):
    o = _tens(enc, enc.encode(obs_dict, set(), self_deck=deck, tracker=tracker, ability_slots=ability))
    logits, val = net.logits_value(o)
    n = len(obs_dict["select"]["option"])
    p = torch.softmax(logits[0, :n], -1).cpu().numpy() if n else np.zeros(0)
    v = float(val[0]); cur = obs_dict["current"]
    return p, (v if cur["yourIndex"] == me else -v)


def _advance(api, sid, obs, net, enc, deck, tracker, ability):
    """Step through forced/multi-pick selects (net-greedy) to the next branchable node."""
    while obs["current"]["result"] < 0 and not _branchable(obs):
        st = api.search_step(sid, _net_greedy_select(obs, net, enc, deck, tracker, ability))
        sid = st.searchId; obs = dataclasses.asdict(st.observation)
    return sid, obs


def mcts_visits(obs, net, enc, deck, tracker=None, ability=None,
                n_sims=160, n_det=2, c_puct=1.5, rng=None):
    """PUCT MCTS over the v2 net; returns (root visit counts over options, ok)."""
    rng = rng or random.Random()
    sel = obs.get("select")
    if sel is None or not _branchable(obs) or sel.get("type") != 0:
        return None, False
    from sdk_cg import api
    me = obs["current"]["yourIndex"]
    n_opt = len(sel["option"])
    agg = np.zeros(n_opt)

    def simulate(node):
        if node.term:
            return node.tv
        if not node.exp:
            node.P, v = _priors_value(net, enc, node.obs, me, deck, tracker, ability)
            node.exp = True
            return v
        N, W, P = node.N, node.W, node.P
        sqrtsum = float(np.sqrt(N.sum() + 1e-8))
        Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
        score = (Q if node.me_turn else -Q) + c_puct * P * sqrtsum / (1.0 + N)
        a = int(score.argmax())
        if a not in node.kids:
            st = api.search_step(node.sid, [a])
            csid, cobs = _advance(api, st.searchId, dataclasses.asdict(st.observation),
                                  net, enc, deck, tracker, ability)
            node.kids[a] = _Node(csid, cobs, me)
        v = simulate(node.kids[a])
        node.N[a] += 1; node.W[a] += v
        return v

    for _ in range(n_det):
        try:
            root_ss = api.search_begin(api.to_observation_class(obs), **_determinize(obs, deck, rng, enc))
        except Exception:
            continue
        root = _Node(root_ss.searchId, dataclasses.asdict(root_ss.observation), me)
        if len(root.N) != n_opt:                              # alignment guard
            try: api.search_end()
            except Exception: pass
            return None, False
        for _ in range(n_sims):
            simulate(root)
        agg += root.N
        try: api.search_release(root_ss.searchId)
        except Exception: pass
    try: api.search_end()
    except Exception: pass
    if agg.sum() == 0:
        return None, False
    return agg, True


def _my_prizes(obs, me) -> int:
    pls = (obs.get("current") or {}).get("players") or []
    return len(pls[me].get("prize") or []) if 0 <= me < len(pls) else 6


def _advance_resolve(api, sid, obs, me):
    """Net-free: step (first-legal) through our remaining sub-selects until the attack resolves
    (turn passes to opp / terminal). Used by the would_KO sim -- no net needed, so it runs in
    env workers during collection."""
    while obs["current"]["result"] < 0 and obs["current"]["yourIndex"] == me:
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        k = sel.get("maxCount", 1) or 1
        pick = list(range(min(k, len(opts)))) if opts else []
        st = api.search_step(sid, pick)
        sid = st.searchId; obs = dataclasses.asdict(st.observation)
    return sid, obs


def would_ko_flags(obs, deck, enc, n_var=WK_NDET_VAR, rng=None) -> dict:
    """Engine-accurate would-KO per ATTACK option: simulate the attack 1 ply on the SDK sim and
    report the KO RATE (we take a prize / win). Net-free + minimal -> usable as a TRAINING FEATURE
    per attack-option (abilities/stadium/weakness/variable all resolved by the real engine).
    VARIABLE-aware: a fixed-damage attack is deterministic given the visible board -> 1 sim (exact);
    a VARIABLE (coin/conditional) attack is sampled `n_var` times -> KO probability (the engine
    re-rolls each determinization; manual_coin=False auto-flips off the persistent agent_ptr RNG).
    Returns {option_index: ko_rate in [0,1]}; {} if no attack options / not a MAIN select."""
    sel = obs.get("select")
    if sel is None or sel.get("type") != 0:
        return {}
    opts = sel.get("option") or []
    atk = [i for i, o in enumerate(opts) if o.get("attackId") is not None]
    if not atk:
        return {}
    from sdk_cg import api
    me = obs["current"]["yourIndex"]
    p0 = _my_prizes(obs, me)
    rng = rng or random.Random()
    out = {}
    for a in atk:
        av = _WK_ATTACKS.get(opts[a].get("attackId"))
        ndet = max(1, n_var) if (av and av[1]) else 1     # variable -> sample prob; fixed -> 1 exact
        kos = trials = 0
        for _ in range(ndet):
            try:
                ss = api.search_begin(api.to_observation_class(obs), **_determinize(obs, deck, rng, enc))
            except Exception:
                continue
            try:
                st = api.search_step(ss.searchId, [a])
                _, o2 = _advance_resolve(api, st.searchId, dataclasses.asdict(st.observation), me)
                trials += 1
                cur = o2["current"]
                if cur["result"] == me or (cur["result"] < 0 and _my_prizes(o2, me) < p0):
                    kos += 1
            except Exception:
                pass
            finally:
                try: api.search_release(ss.searchId)
                except Exception: pass
        if trials:
            out[a] = kos / trials
    try: api.search_end()
    except Exception: pass
    return out


def write_would_ko(obs, flags) -> None:
    """Write o['would_ko']=rate onto each attack option in-place (so the encoder emits the feature)."""
    opts = (obs.get("select") or {}).get("option") or []
    for i, r in flags.items():
        if 0 <= i < len(opts):
            opts[i]["would_ko"] = float(r)


def annotate_would_ko(obs, deck, enc, n_var=WK_NDET_VAR, rng=None) -> dict:
    """Compute would_ko_flags AND write them onto the attack options (the per-option TRAINING
    feature). Call ONCE per real (root) decision -- in the env at collection AND in the inference
    agent when net_config['would_ko'] -> train==test (both use the default n_var). Returns flags."""
    flags = would_ko_flags(obs, deck, enc, n_var=n_var, rng=rng)
    write_would_ko(obs, flags)
    return flags


def mcts_select(obs, net, enc, deck, tracker=None, ability=None,
                n_sims=160, n_det=2, c_puct=1.5, rng=None):
    """PUCT MCTS choice -> selection list[int] (net-greedy fallback / deck step)."""
    if obs.get("select") is None:                             # deck-selection step
        return [int(c) for c in deck]
    agg, ok = mcts_visits(obs, net, enc, deck, tracker, ability, n_sims, n_det, c_puct, rng)
    if not ok:
        return _net_greedy_select(obs, net, enc, deck, tracker, ability)
    return [int(agg.argmax())]


def mcts_policy(obs, net, enc, deck, tracker=None, ability=None,
                n_sims=160, n_det=2, c_puct=1.5, rng=None, temp=1.0):
    """For AlphaZero self-play on v2: returns (selection list[int], pi over N_ACTIONS)."""
    agg, ok = mcts_visits(obs, net, enc, deck, tracker, ability, n_sims, n_det, c_puct, rng)
    if not ok:
        return _net_greedy_select(obs, net, enc, deck, tracker, ability), None
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    if temp and temp != 1.0:
        agg = agg ** (1.0 / temp)
    pi[:len(agg)] = agg / agg.sum()
    a = int(np.random.choice(len(agg), p=pi[:len(agg)])) if temp > 0 else int(agg.argmax())
    return [a], pi
