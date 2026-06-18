"""Observation encoder: cabt ``obs`` dict -> fixed-shape numpy arrays.

Design (see rl/STATE_ACTION_SPACE.md for the full spec):

* Card identity is provided BOTH ways ("static features now, embedding hook"):
  every card surface emits a static feature array (from EN_Card_Data.csv) AND a
  raw integer id, so a CleanRL net can concat the static features with a learned
  ``nn.Embedding`` lookup.
* The action space is per-option: each candidate option gets its own feature row,
  the policy scores rows, and an action mask hides illegal/padded slots.

All arrays are float32 except ``*_id`` (int64). Everything is encoded from the
ACTING player's perspective (``current.yourIndex`` is "self").

v2 (post-audit) fixes:
* ACTIVE Pokemon is encoded as its own vector, BENCH pooled separately, so the
  policy can identify the in-combat Pokemon (previously active+bench were pooled
  together order-invariantly, hiding which one was active).
* DECK-SEARCH options carry no cardId; the chosen card lives in ``sel['deck']``.
  We resolve it (area==1 -> sel['deck'][index]) so search picks aren't blind.
* MAX_OPTIONS 64->96 (engine can emit up to ~69) and MAX_HAND 15->20.
* per-Pokemon count/energy features are clipped to [0,1] (no saturation).
"""

from __future__ import annotations

import numpy as np

from .card_features import CardTable, N_ENERGY, get_card_table

# ---- shape constants (tune here) -------------------------------------------
N_BENCH = 5          # bench slots (active is encoded separately)
MAX_HAND = 20        # observed max hand ~17
MAX_DISCARD = 30     # per-card discard window (rest summarized by the count scalar)
MAX_OPTIONS = 128    # ladder emitted 98-option selects; 128 gives headroom (was 96).
                     # Pointer-scored, so widening costs no weights; search_agent still
                     # caps/pads/masks beyond this as a bound-free backstop.
                     # index MAX_OPTIONS == "submit"
N_OPT_TYPES = 16     # OptionType range 0..15
N_SELECT_TYPES = 16  # SelectType range
N_SELECT_CTX = 64    # SelectContext embedding vocab cap

SUBMIT_ACTION = MAX_OPTIONS  # the extra action index meaning "stop / submit set"
N_ACTIONS = MAX_OPTIONS + 1

# normalisation scales
_T = 50.0; _CNT = 15.0; _DECK = 60.0; _PRIZE = 6.0; _DISCARD = 40.0


def _f(x) -> float:
    return float(x) if x is not None else 0.0


class Encoder:
    """Stateless (given a CardTable) obs -> arrays encoder."""

    def __init__(self, card_table: CardTable | None = None):
        self.cards = card_table or get_card_table()
        self.cf = self.cards.feat_dim  # static card feature width

    # ---- public shape description (for building the net / gym spaces) -----
    @property
    def shapes(self) -> dict[str, tuple]:
        cf = self.cf
        return {
            "scalars": (14,),
            "self_player": (10,), "opp_player": (10,),
            "self_active_dyn": (1, 18), "opp_active_dyn": (1, 18),
            "self_active_static": (1, cf), "opp_active_static": (1, cf),
            "self_active_id": (1,), "opp_active_id": (1,),
            "self_bench_dyn": (N_BENCH, 18), "opp_bench_dyn": (N_BENCH, 18),
            "self_bench_static": (N_BENCH, cf), "opp_bench_static": (N_BENCH, cf),
            "self_bench_id": (N_BENCH,), "opp_bench_id": (N_BENCH,),
            "hand_static": (MAX_HAND, cf), "hand_id": (MAX_HAND,), "hand_mask": (MAX_HAND,),
            "self_discard_static": (MAX_DISCARD, cf), "self_discard_id": (MAX_DISCARD,), "self_discard_mask": (MAX_DISCARD,),
            "opp_discard_static": (MAX_DISCARD, cf), "opp_discard_id": (MAX_DISCARD,), "opp_discard_mask": (MAX_DISCARD,),
            "stadium_static": (cf,), "stadium_id": (1,),
            "opt_dyn": (MAX_OPTIONS, 28),
            "opt_card_static": (MAX_OPTIONS, cf), "opt_card_id": (MAX_OPTIONS,),
            "opt_tgt_static": (MAX_OPTIONS, cf), "opt_tgt_id": (MAX_OPTIONS,),
            "select_type": (1,), "select_context": (1,),
            "action_mask": (N_ACTIONS,),
        }

    @property
    def int_keys(self):
        return {"self_active_id", "opp_active_id", "self_bench_id", "opp_bench_id",
                "hand_id", "self_discard_id", "opp_discard_id",
                "stadium_id", "opt_card_id", "opt_tgt_id", "select_type", "select_context"}

    # ---- helpers ----------------------------------------------------------
    def _energy_hist(self, energies) -> np.ndarray:
        v = np.zeros(N_ENERGY, dtype=np.float32)
        for e in (energies or []):
            if 0 <= e < N_ENERGY:
                v[e] += 1.0
        return v

    def _pokemon(self, pk) -> tuple[np.ndarray, np.ndarray, int]:
        """-> (dynamic[18], static[cf], card_id). Count features clipped to [0,1]."""
        if pk is None:
            return np.zeros(18, np.float32), np.zeros(self.cf, np.float32), 0
        eh = np.minimum(self._energy_hist(pk.get("energies")) / 10.0, 1.0)
        maxhp = _f(pk.get("maxHp"))
        dyn = np.array([
            1.0,                                              # present
            _f(pk.get("hp")) / max(maxhp, 1.0),              # hp ratio (<=1)
            min(maxhp / 400.0, 1.0),
            *eh,                                              # 11
            min(len(pk.get("energies") or []) / 10.0, 1.0),
            min(len(pk.get("tools") or []) / 3.0, 1.0),
            1.0 if pk.get("appearThisTurn") else 0.0,
            min(len(pk.get("preEvolution") or []) / 2.0, 1.0),
        ], dtype=np.float32)
        cid = pk.get("id") or 0
        return dyn, self.cards.features(cid), cid

    def _fill(self, seq, n):
        """seq of Pokemon -> (dyn[n,18], static[n,cf], ids[n])."""
        dyn = np.zeros((n, 18), np.float32)
        static = np.zeros((n, self.cf), np.float32)
        ids = np.zeros(n, np.int64)
        for i, pk in enumerate(seq[:n]):
            dyn[i], static[i], ids[i] = self._pokemon(pk)
        return dyn, static, ids

    def _active(self, pl):
        return self._fill(list(pl.get("active") or [])[:1], 1)

    def _bench(self, pl):
        return self._fill(list(pl.get("bench") or []), N_BENCH)

    def _player(self, pl) -> np.ndarray:
        return np.array([
            _f(pl.get("handCount")) / _CNT,
            _f(pl.get("deckCount")) / _DECK,
            len(pl.get("prize") or []) / _PRIZE,
            len(pl.get("discard") or []) / _DISCARD,
            _f(pl.get("benchMax")) / 5.0,
            1.0 if pl.get("poisoned") else 0.0,
            1.0 if pl.get("burned") else 0.0,
            1.0 if pl.get("asleep") else 0.0,
            1.0 if pl.get("paralyzed") else 0.0,
            1.0 if pl.get("confused") else 0.0,
        ], dtype=np.float32)

    def _pile(self, cards, n):
        """A card pile (hand/discard) -> (static[n,cf], ids[n], mask[n]), per-card."""
        static = np.zeros((n, self.cf), np.float32)
        ids = np.zeros(n, np.int64)
        mask = np.zeros(n, np.float32)
        for i, c in enumerate((cards or [])[:n]):
            cid = (c.get("id") if isinstance(c, dict) else c) or 0
            static[i] = self.cards.features(cid); ids[i] = cid; mask[i] = 1.0
        return static, ids, mask

    def _option_row(self, o: dict, cid: int) -> tuple[np.ndarray, np.ndarray, int]:
        """``cid`` is the option's resolved card id (0 if none)."""
        cid = cid or 0
        t = np.zeros(N_OPT_TYPES, np.float32)
        ot = o.get("type", 0)
        if 0 <= ot < N_OPT_TYPES:
            t[ot] = 1.0
        dyn = np.array([
            *t,                                    # 16
            _f(o.get("area")) / _CNT,
            _f(o.get("index")) / _CNT,
            _f(o.get("playerIndex")),
            _f(o.get("inPlayArea")) / _CNT,
            _f(o.get("inPlayIndex")) / _CNT,
            _f(o.get("energyIndex")) / _CNT,
            _f(o.get("count")) / 5.0,
            _f(o.get("number")) / _CNT,
            1.0 if cid else 0.0,                   # references a known card
            1.0 if o.get("attackId") is not None else 0.0,
            1.0 if o.get("serial") is not None else 0.0,
            _f(o.get("attackId")) / 2000.0,
        ], dtype=np.float32)
        return dyn, self.cards.features(cid), cid

    def _zone_card(self, s, area, index, player, deck_list) -> int:
        """Card id at (area, index) in `player`'s zone (cabt AreaType). 0 if unresolvable."""
        if not isinstance(index, int) or index < 0:
            return 0
        if area == 1:                                   # DECK (visible via sel['deck'])
            arr = deck_list
        elif area == 7:                                 # STADIUM
            arr = s.get("stadium")
        elif area == 12:                                # LOOKING
            arr = s.get("looking")
        else:
            zone = {2: "hand", 3: "discard", 4: "active", 5: "bench"}.get(area)
            players = s.get("players") or []
            if zone is None or not (0 <= player < len(players)):
                return 0
            arr = players[player].get(zone)
        if not arr or index >= len(arr):
            return 0
        c = arr[index]
        if c is None:
            return 0
        return (c.get("id") if isinstance(c, dict) else c) or 0

    def _resolve_card(self, o: dict, s, me, deck_list) -> int:
        """The SOURCE card an option acts on: explicit cardId, else the card at the
        option's (area,index) in its owner's zone (hand/deck/discard/active/bench).
        Previously deck-only -> now any zone, so PLAY/ATTACH/EVOLVE/ABILITY/DISCARD
        options carry the actual card identity instead of just a positional index."""
        cid = o.get("cardId")
        if cid is not None:
            return cid
        p = o.get("playerIndex")
        return self._zone_card(s, o.get("area"), o.get("index"), me if p is None else p, deck_list)

    def _resolve_target(self, o: dict, s, me) -> int:
        """The TARGET Pokemon an option affects: the in-play card at
        (inPlayArea, inPlayIndex) -- e.g. which Pokemon gets the energy / evolves."""
        p = o.get("playerIndex")
        return self._zone_card(s, o.get("inPlayArea"), o.get("inPlayIndex"), me if p is None else p, None)

    # ---- main entry -------------------------------------------------------
    def encode(self, obs: dict, picked: set[int] | None = None) -> dict[str, np.ndarray]:
        """Encode a cabt observation. ``picked`` = option indices already chosen
        this decision (for multi-select buffering)."""
        picked = picked or set()
        s = obs["current"]
        me = s["yourIndex"]
        opp = 1 - me
        mp, op = s["players"][me], s["players"][opp]
        sel = obs["select"]

        out: dict[str, np.ndarray] = {}

        # board: active (own vector) + bench (pooled), per side
        out["self_active_dyn"], out["self_active_static"], out["self_active_id"] = self._active(mp)
        out["opp_active_dyn"], out["opp_active_static"], out["opp_active_id"] = self._active(op)
        out["self_bench_dyn"], out["self_bench_static"], out["self_bench_id"] = self._bench(mp)
        out["opp_bench_dyn"], out["opp_bench_static"], out["opp_bench_id"] = self._bench(op)
        out["self_player"] = self._player(mp)
        out["opp_player"] = self._player(op)

        # discard: per-card (identity-aware), both sides public. Own hand per-card too
        # (opponent hand is hidden -> only handCount, already in the player vec).
        out["self_discard_static"], out["self_discard_id"], out["self_discard_mask"] = self._pile(mp.get("discard"), MAX_DISCARD)
        out["opp_discard_static"], out["opp_discard_id"], out["opp_discard_mask"] = self._pile(op.get("discard"), MAX_DISCARD)
        out["hand_static"], out["hand_id"], out["hand_mask"] = self._pile(mp.get("hand"), MAX_HAND)

        # stadium
        stad = s.get("stadium") or []
        stad_card = stad[0] if stad else None
        out["stadium_static"] = self.cards.features(stad_card.get("id") if stad_card else None)
        out["stadium_id"] = np.array([stad_card.get("id") if stad_card else 0], np.int64)

        # select-context scalars
        n_opt = len(sel["option"])
        out["scalars"] = np.array([
            _f(s.get("turn")) / _T,
            _f(s.get("turnActionCount")) / 20.0,
            1.0 if s.get("firstPlayer") == me else 0.0,
            1.0 if s.get("supporterPlayed") else 0.0,
            1.0 if s.get("stadiumPlayed") else 0.0,
            1.0 if s.get("energyAttached") else 0.0,
            1.0 if s.get("retreated") else 0.0,
            _f(sel.get("remainDamageCounter")) / 20.0,
            _f(sel.get("remainEnergyCost")) / 5.0,
            _f(sel.get("minCount")) / 3.0,
            _f(sel.get("maxCount")) / 3.0,
            len(picked) / 3.0,
            min(n_opt, MAX_OPTIONS) / float(MAX_OPTIONS),
            1.0 if stad_card else 0.0,
        ], dtype=np.float32)
        out["select_type"] = np.array([min(sel.get("type", 0), N_SELECT_TYPES - 1)], np.int64)
        out["select_context"] = np.array([min(sel.get("context", 0), N_SELECT_CTX - 1)], np.int64)

        # options: resolve each option's SOURCE card (any zone) + TARGET Pokemon, so the
        # net scores by full content (which card / which target), not a positional index.
        deck_list = sel.get("deck") if isinstance(sel.get("deck"), list) else None
        opt_dyn = np.zeros((MAX_OPTIONS, 28), np.float32)
        opt_static = np.zeros((MAX_OPTIONS, self.cf), np.float32)
        opt_id = np.zeros(MAX_OPTIONS, np.int64)
        opt_tgt_static = np.zeros((MAX_OPTIONS, self.cf), np.float32)
        opt_tgt_id = np.zeros(MAX_OPTIONS, np.int64)
        for i, o in enumerate(sel["option"][:MAX_OPTIONS]):
            src = self._resolve_card(o, s, me, deck_list)
            opt_dyn[i], opt_static[i], opt_id[i] = self._option_row(o, src)
            tgt = self._resolve_target(o, s, me)
            opt_tgt_static[i], opt_tgt_id[i] = self.cards.features(tgt), tgt
        out["opt_dyn"], out["opt_card_static"], out["opt_card_id"] = opt_dyn, opt_static, opt_id
        out["opt_tgt_static"], out["opt_tgt_id"] = opt_tgt_static, opt_tgt_id

        out["action_mask"] = build_mask(sel, picked)
        return out


def build_mask(sel: dict, picked: set[int]) -> np.ndarray:
    """Legal-action mask over [0..MAX_OPTIONS] (last index == submit).

    * an option index is legal if it exists, fits MAX_OPTIONS, and isn't picked;
    * submit is legal once at least ``minCount`` options are picked (and we are
      not auto-submitting, which the env handles when picked == maxCount).
    """
    mask = np.zeros(N_ACTIONS, np.float32)
    n = min(len(sel["option"]), MAX_OPTIONS)
    min_c, max_c = sel.get("minCount", 1), sel.get("maxCount", 1)
    if len(picked) < max_c:
        for i in range(n):
            if i not in picked:
                mask[i] = 1.0
    if min_c <= len(picked) < max_c:
        mask[SUBMIT_ACTION] = 1.0
    # safety: never return an all-zero mask
    if mask.sum() == 0:
        mask[SUBMIT_ACTION] = 1.0
    return mask
