"""Shared decision-time search primitives (independent of the net).

Observation-conditioned PIMC determinization (`_determinize`), the branchable-node
test (`_branchable`), and the MCTS node (`_Node`) -- imported by the live v2 MCTS in
search_agent2.py. The v1-net MCTS that used to live here (search_select / mcts_* /
the obs_to_tensors-based value + priors) was removed along with the v1 policy; only
these net-agnostic primitives remain.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .decks import DECKS
from .encoding import MAX_OPTIONS


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
try:    # generated archetypes -> match the --decks all+gen training pool (absent in the submission bundle, which stays at the 5 meta decks)
    from .decks_generated import GENERATED
    _CANDIDATES.extend(list(d) for d in GENERATED.values())
except Exception:
    pass


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
