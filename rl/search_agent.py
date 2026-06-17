"""Decision-time search (PIMC + value-net lookahead) on a trained net.

Uses the SDK forward model (sdk_cg.api: search_begin/search_step) to look ahead.
At a MAIN single-pick decision, for each legal option we simulate it across a few
DETERMINIZATIONS of the hidden cards (PIMC), evaluate each resulting state with the
trained value head, and pick the option with the best averaged value. Sub-selects
and multi-pick decisions defer to the net's greedy policy.

This is the inference-time AlphaZero-lite: net for value, engine for lookahead.
Needs the SDK in sdk_cg/ (api.py + cg.dll/libcg.so). The search runs on a separate
`agent_ptr` simulator, so it never touches the live battle.
"""

from __future__ import annotations

import dataclasses
import random
from collections import Counter

import numpy as np
import torch

from .decks import DECKS
from .encoding import MAX_OPTIONS, N_ACTIONS, SUBMIT_ACTION, build_mask
from .policy import obs_to_tensors

def _load_sample_deck():
    """The engine 'sample' deck (agent/deck.csv) -- part of the meta/training pool but
    not in DECKS. Bundle-safe: alongside this module in a submission, ../agent in repo."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "deck.csv"), os.path.join(here, "..", "agent", "deck.csv")):
        try:
            if os.path.exists(p):
                return [int(x) for x in open(p) if x.strip()]
        except Exception:
            pass
    return None


# Deck hypotheses for opponent inference: the known meta = official archetypes + the
# engine sample deck (which is also the multi-deck training pool). Covers every deck an
# opponent might play, so inference isn't forced into a same-archetype fallback.
_CANDIDATES = [list(d) for d in DECKS.values()]
_sample_deck = _load_sample_deck()
if _sample_deck:
    _CANDIDATES.append(_sample_deck)


def _net_greedy_select(obs, net, enc, device):
    """Full engine selection from the net (buffered single-pick), no search."""
    sel = obs["select"]
    picked: list[int] = []
    for _ in range(sel.get("maxCount", 1) + 1):
        o = obs_to_tensors(enc.encode(obs, set(picked)), device)
        o = {k: v[None] for k, v in o.items()}
        with torch.no_grad():
            logits, _ = net.logits_value(o)
        ml = logits[0].clone()
        mask = torch.as_tensor(np.asarray(build_mask(sel, set(picked))),
                               dtype=torch.bool, device=ml.device)
        if mask.any():                              # never pick an illegal action
            ml[~mask] = -1e9
        a = int(ml.argmax())
        if a == SUBMIT_ACTION:
            break
        picked.append(a)
        if len(picked) >= sel.get("maxCount", 1):
            break
    return sorted(set(picked))


def _value(net, enc, obs_dict, me, device) -> float:
    """Value of a (search) state from player `me`'s perspective, in [-1, 1]."""
    cur = obs_dict["current"]
    if cur["result"] >= 0:                       # terminal
        if cur["result"] == 2:
            return 0.0
        return 1.0 if cur["result"] == me else -1.0
    o = obs_to_tensors(enc.encode(obs_dict), device)
    o = {k: v[None] for k, v in o.items()}
    with torch.no_grad():
        v = float(net.get_value(o)[0])
    # net value is for the acting player; flip if it's the opponent's turn
    return v if cur["yourIndex"] == me else -v


def _cid(c):
    return c.get("id") if isinstance(c, dict) else c


def _observed(pl, stadium, owner, with_hand):
    """Counter of card ids of player `owner` that are publicly visible (or, for our
    own side, also our hand). These are hard constraints: definitely in their 60."""
    cnt = Counter()
    for grp in ("active", "bench"):
        for pk in (pl.get(grp) or []):
            if not pk:
                continue
            cnt[pk["id"]] += 1
            for key in ("preEvolution", "energyCards", "tools"):
                for c in (pk.get(key) or []):
                    cnt[_cid(c)] += 1
    for c in (pl.get("discard") or []):
        cnt[_cid(c)] += 1
    if with_hand:
        for c in (pl.get("hand") or []):
            cnt[_cid(c)] += 1
    for c in (stadium or []):
        if isinstance(c, dict) and c.get("playerIndex") == owner:
            cnt[c["id"]] += 1
    return cnt


def _fit(cards, n, full, rng):
    """Trim/pad a card list to exactly n (pad by sampling the full deck)."""
    cards = list(cards)
    rng.shuffle(cards)
    if len(cards) > n:
        return cards[:n]
    while len(cards) < n and full:
        cards.append(rng.choice(full))
    return cards


def _basic_pokemon(ids, enc):
    """A basic Pokemon id from `ids` (for a face-down opp active), else None."""
    for cid in ids:
        f = enc.cards.features(cid)
        if f is not None and f[0] > 0.5 and f[3] > 0.5:   # category=pokemon, stage=Basic
            return int(cid)
    return None


def _determinize(obs, deck, rng, enc):
    """Observation-conditioned PIMC. The opponent's visible cards (discard, board,
    attachments, their stadium) are hard constraints; we infer which known decklist
    is consistent with them and fill the hidden zones from `inferred_deck - observed`.
    Our own hidden cards (deck/prizes) come from `our_deck - seen - hand`."""
    s = obs["current"]; me = s["yourIndex"]
    mp, op = s["players"][me], s["players"][1 - me]
    stadium = s.get("stadium")

    # --- opponent: infer archetype, then deal the unseen remainder ---
    seen = _observed(op, stadium, 1 - me, with_hand=False)
    cands = [Counter(d) for d in (_CANDIDATES + [list(deck)])]
    consistent = [c for c in cands if all(seen[k] <= c[k] for k in seen)]
    if not consistent:                       # off-meta opponent: best-overlap match
        consistent = [max(cands, key=lambda c: sum(min(seen[k], c[k]) for k in seen))]
    D = rng.choice(consistent)               # ensemble across determinizations
    full_opp = list(D.elements())
    rem = []
    for cid, ct in D.items():
        rem += [cid] * max(0, ct - seen[cid])
    rng.shuffle(rem)

    dC = op["deckCount"]; pC = len(op.get("prize") or []); hC = op.get("handCount", 0)
    face = bool(op.get("active") and op["active"][0] is None)
    rem = _fit(rem, dC + pC + hC + (1 if face else 0), full_opp, rng)
    i = 0
    opp_deck = rem[i:i + dC]; i += dC
    opp_prize = rem[i:i + pC]; i += pC
    opp_hand = rem[i:i + hC]; i += hC
    opp_active = []
    if face:
        opp_active = [_basic_pokemon(rem[i:] + full_opp, enc) or rem[i]]

    # --- our side: deck/prizes are what's left of our deck after seen + hand ---
    seen_me = _observed(mp, stadium, me, with_hand=True)
    Dme = Counter(deck)
    rem_me = []
    for cid, ct in Dme.items():
        rem_me += [cid] * max(0, ct - seen_me[cid])
    rng.shuffle(rem_me)
    mdC = mp["deckCount"]; mpC = len(mp.get("prize") or [])
    rem_me = _fit(rem_me, mdC + mpC, list(deck), rng)

    return dict(
        your_deck=rem_me[:mdC],
        your_prize=rem_me[mdC:mdC + mpC],
        opponent_deck=opp_deck,
        opponent_prize=opp_prize,
        opponent_hand=opp_hand,
        opponent_active=opp_active,
    )


def search_select(obs, net, enc, deck, device, n_det=4, rng=None):
    """Choose a selection for `obs` using value-net lookahead. Returns list[int]."""
    rng = rng or random.Random()
    sel = obs.get("select")
    if sel is None:
        return [int(c) for c in deck]              # deck-selection step
    # only search MAIN single-pick decisions with a real choice
    if sel.get("type") != 0 or sel.get("maxCount", 1) != 1 or len(sel["option"]) < 2:
        return _net_greedy_select(obs, net, enc, device)

    from sdk_cg import api
    me = obs["current"]["yourIndex"]
    n_opt = len(sel["option"])
    vals = np.zeros(n_opt, dtype=np.float64)
    counts = np.zeros(n_opt, dtype=np.int64)
    oc = api.to_observation_class(obs)

    for _ in range(n_det):
        try:
            root = api.search_begin(oc, **_determinize(obs, deck, rng, enc))
        except Exception:
            continue                               # determinization rejected; skip
        for i in range(n_opt):
            try:
                child = api.search_step(root.searchId, [i])
            except Exception:
                continue
            cobs = dataclasses.asdict(child.observation)
            vals[i] += _value(net, enc, cobs, me, device)
            counts[i] += 1
        try:
            api.search_release(root.searchId)
        except Exception:
            pass
    try:
        api.search_end()
    except Exception:
        pass

    if counts.sum() == 0:                          # all determinizations failed
        return _net_greedy_select(obs, net, enc, device)
    avg = np.where(counts > 0, vals / np.maximum(counts, 1), -1e9)
    return [int(avg.argmax())]


# ---- proper PUCT MCTS (policy priors + value leaves + adversarial backup) ----
def _priors_value(net, enc, obs_dict, me, device):
    """Policy priors over the node's options + value (root `me` perspective)."""
    o = obs_to_tensors(enc.encode(obs_dict), device)
    o = {k: v[None] for k, v in o.items()}
    with torch.no_grad():
        logits, val = net.logits_value(o)
    n = len(obs_dict["select"]["option"])
    w = logits.shape[-1]                            # net can only score N_ACTIONS slots
    m = min(n, w)
    p = torch.softmax(logits[0, :m], -1).cpu().numpy() if m else np.zeros(0)
    if n > m:                                       # >N_ACTIONS options: keep P aligned to N
        p = np.concatenate([p, np.full(n - m, 1e-8, dtype=p.dtype)])
        p = p / p.sum()
    v = float(val[0])
    cur = obs_dict["current"]
    v = v if cur["yourIndex"] == me else -v
    return p, v


def _branchable(obs_dict):
    """A node we branch on: single-pick with >=2 real options, not terminal."""
    s = obs_dict.get("select")
    return (s is not None and s.get("maxCount", 1) == 1
            and 2 <= len(s["option"]) <= MAX_OPTIONS   # >MAX_OPTIONS: not encodable, net can't score
            and obs_dict["current"]["result"] < 0)


class _Node:
    __slots__ = ("sid", "obs", "me_turn", "term", "tv", "P", "N", "W", "kids", "exp")

    def __init__(self, sid, obs, me):
        self.sid = sid; self.obs = obs
        cur = obs["current"]
        self.term = cur["result"] >= 0
        self.tv = (0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)) if self.term else 0.0
        self.me_turn = cur["yourIndex"] == me
        self.P = None; self.exp = False
        n = len(obs["select"]["option"]) if obs.get("select") else 0
        self.N = np.zeros(n); self.W = np.zeros(n); self.kids = {}


def _advance(api, sid, obs, net, enc, device):
    """Step through forced/multi-pick selects (net-greedy) to the next branchable
    node or terminal, within the search tree. Returns (sid, obs)."""
    while obs["current"]["result"] < 0 and not _branchable(obs):
        pick = _net_greedy_select(obs, net, enc, device)
        st = api.search_step(sid, pick)
        sid = st.searchId; obs = dataclasses.asdict(st.observation)
    return sid, obs


def mcts_visits(obs, net, enc, deck, device, n_sims=40, n_det=2, c_puct=1.5, rng=None):
    """Run PUCT MCTS; return (root visit counts over options, ok). ok=False if the
    decision isn't searchable (caller should fall back to the net)."""
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
            node.P, v = _priors_value(net, enc, node.obs, me, device)
            node.exp = True
            return v
        N, W, P = node.N, node.W, node.P
        sqrtsum = float(np.sqrt(N.sum() + 1e-8))
        Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
        score = (Q if node.me_turn else -Q) + c_puct * P * sqrtsum / (1.0 + N)
        a = int(score.argmax())
        if a not in node.kids:
            st = api.search_step(node.sid, [a])
            csid, cobs = _advance(api, st.searchId, dataclasses.asdict(st.observation), net, enc, device)
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
        if len(root.N) != n_opt:                   # alignment guard
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


def mcts_select(obs, net, enc, deck, device, n_sims=40, n_det=2, c_puct=1.5, rng=None):
    """PUCT MCTS choice -> selection list[int] (net-greedy fallback)."""
    if obs.get("select") is None:                  # deck-selection step
        return [int(c) for c in deck]
    agg, ok = mcts_visits(obs, net, enc, deck, device, n_sims, n_det, c_puct, rng)
    if not ok:
        return _net_greedy_select(obs, net, enc, device)
    return [int(agg.argmax())]


def mcts_policy(obs, net, enc, deck, device, n_sims=40, n_det=2, c_puct=1.5, rng=None, temp=1.0):
    """For AlphaZero self-play: returns (selection list[int], pi over N_ACTIONS).
    pi is the MCTS visit distribution (policy training target); None if unsearchable."""
    from .encoding import N_ACTIONS
    agg, ok = mcts_visits(obs, net, enc, deck, device, n_sims, n_det, c_puct, rng)
    if not ok:
        return _net_greedy_select(obs, net, enc, device), None
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    if temp and temp != 1.0:
        agg = agg ** (1.0 / temp)
    pi[:len(agg)] = agg / agg.sum()
    # sample proportionally to visits during self-play (exploration)
    a = int(np.random.choice(len(agg), p=pi[:len(agg)])) if temp > 0 else int(agg.argmax())
    return [a], pi
