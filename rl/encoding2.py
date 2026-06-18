"""Token-based observation encoder (v2): cabt ``obs`` dict -> token streams.

This is an ADDITIVE sibling of ``rl/encoding.py``. It does NOT replace it. Where
``encoding.py`` emits a handful of pooled / per-slot feature blocks for an MLP,
this encoder emits a *set of tokens* to be consumed by a Transformer
(``rl/policy2.TokenTransformer``):

  * every card surface (decklists, prizes, hands, discards, stadium) is a flat
    list of card-id tokens (+ a pad mask);
  * every in-play Pokemon (active + bench, both players) is ONE "unit" token
    carrying its top id, pre-evolution ids, tool ids, and a fixed 23-dim attr
    vector;
  * each candidate option is an "option" token (source id + target id + the
    SAME structural/attack/special-condition features ``encoding.py`` uses);
  * a single CLS token carries the global scalar vector.

The action space is REUSED from ``encoding.py`` (MAX_OPTIONS / SUBMIT_ACTION /
N_ACTIONS / build_mask) so the two encoders are interchangeable at the action
layer; only the *observation* representation differs.

Placeholder ids (shared with the embedding table in policy2):
  * EMPTY = 0  -> ``nn.Embedding(padding_idx=0)`` (no card / pad slot).
  * UNK   = vocab_size -> "a card exists here but its identity is hidden"
    (face-down prize, opponent hand). The embedding table is therefore sized
    ``vocab_size + 1`` so index ``vocab_size`` is a learnable "unknown" vector.

All id arrays are int64; all attr/mask arrays are float32. No batch dim is added
(``obs_to_tensors2`` in policy2 batches/stacks). Encoded from the ACTING player's
perspective (``current.yourIndex`` == "self").
"""

from __future__ import annotations

import numpy as np

from .card_features import CardTable, get_card_table

# REUSE the action-space contract + option-feature helpers from encoding.py so the
# two encoders score the same action set. We intentionally import these (do not
# re-derive them) -- if encoding.py changes the action space, this follows.
from .encoding import (
    MAX_OPTIONS, SUBMIT_ACTION, N_ACTIONS, OPT_DYN,  # noqa: F401  (SUBMIT_ACTION re-exported as part of the action contract)
    N_SELECT_TYPES, N_SELECT_CTX,
    build_mask, Encoder as _MlpEncoder,
)

try:    # transient turn-buff tables (bundle-safe; empty if missing -> feature stays 0)
    from .buff_data import DEFENSE_BUFF_ATTACKS, OFFENSE_BUFF_CARDS
except Exception:
    DEFENSE_BUFF_ATTACKS, OFFENSE_BUFF_CARDS = {}, {}

# ---- token-stream shape constants (tune here) ------------------------------
N_ENERGY_BINS = 12   # EnergyType bins 0..11 (spec: histogram over 12 types)
UNIT_ATTR = 24       # unit-attr layout below (idx 23 = next-turn damage-reduction buff)
OPT_UNIT_DIM = UNIT_ATTR + 1  # gathered referenced-unit attr + a present flag (per option, src & tgt)
N_PREEVO = 2         # pre-evolution stack ids per unit (padded)
N_TOOLS = 2          # attached tool ids per unit (padded)

N_BENCH = 5          # bench unit slots per side (active is its own slot)
DECK_SIZE = 60       # our/opp "decklist" token budget
N_PRIZE = 6          # prize slots per side
MAX_HAND = 20        # hand token budget (observed max ~17; encoding.py uses 20)
MAX_DISCARD = 60     # discard token budget (public)
N_STADIUM = 2        # slot0 = self-owned stadium, slot1 = opp-owned (position=owner)

G = 19               # CLS scalars: 13 board/turn + 5 select-dynamics + 1 our-this-turn offensive buff

# normalisation scales (match the spec's CLS layout)
_T = 50.0


def _f(x) -> float:
    return float(x) if x is not None else 0.0


def _card_id(c) -> int:
    """A cabt Card (dict with .id) or bare int -> its id (0 if none)."""
    if c is None:
        return 0
    if isinstance(c, dict):
        return c.get("id") or 0
    return c or 0


def _id_serial(c):
    """A cabt Card dict -> (id, serial); a bare int -> (id, None). serial is a unique
    per-physical-card id (present on every card surface + reveal log), used to count
    distinct copies without over-counting a single card recycled through zones."""
    if isinstance(c, dict):
        return c.get("id"), c.get("serial")
    return c, None


def _board_slot(area, index):
    """cabt AreaType ACTIVE=4 / BENCH=5 (+index) -> unit slot (0=active, 1+i=bench[i]).
    None if the reference isn't an in-play Pokemon (hand/deck/discard/stadium/...) -- used
    to gather the live state of the board Pokemon an option's source/target points at."""
    if area == 4:
        return 0
    if area == 5 and isinstance(index, int) and 0 <= index < N_BENCH:
        return 1 + index
    return None


class GameTracker:
    """Per-game memory of which cards (and HOW MANY distinct copies) each player has revealed.

    For each ``playerIndex`` we store, per cardId, the SET OF SERIALS seen. A serial is a
    unique per-physical-card id, so ``len(serials)`` is the true number of distinct copies
    that player has shown -- a single card cycling play->discard->hand->play keeps ONE serial
    (no over-count), while two real copies contribute two. Serials are populated from two
    sources, unioned (so the set dedups automatically):
      (1) reveal LOGS (PLAY/ATTACH/EVOLVE/DEVOLVE/MOVE_CARD/MOVE_ATTACHED/ATTACK -- all carry
          cardId+serial) -> catches copies that have since returned to a hidden zone;
      (2) CURRENTLY-VISIBLE cards (discard, board Pokemon + their preevo/tools/energy, stadium)
          -> catches copies whose move-log we never observed.

    IMPORTANT: ``obs['logs']`` is an INCREMENTAL delta (verified -- log length is non-monotonic),
    NOT cumulative, so ``update(obs)`` must run on every decision obs this side sees; the serial
    SETS persist across the game. Attribution is by ``playerIndex`` (perspective-independent), so
    one tracker serves both sides -- each reads ``copies_for(its_opponent_index)``. ``reset()`` per game.

    NOTE: the engine has NO 'ability used' log event -- that is handled separately by
    AbilityTracker (observed from our own selections, not from logs).
    """
    _REVEAL_TYPES = frozenset({6, 10, 11, 12, 13, 14, 15})  # MOVE_CARD,PLAY,ATTACH,EVOLVE,DEVOLVE,MOVE_ATTACHED,ATTACK

    def __init__(self):
        self.reset()

    def reset(self):
        # playerIndex -> {cardId: set(serial)}; len(set) == distinct copies seen.
        self.serials: dict[int, dict] = {0: {}, 1: {}}
        # transient turn-buffs (reset each owner turn): see DEFENSE/OFFENSE buff tables.
        self._buff_turn = None
        self.opp_def_buff: dict = {}    # serial -> reduction: OPP units shielded during MY current turn
        self.offense_buff: float = 0.0  # MY this-turn extra attack damage (from a trainer I played)

    def _add(self, pi, cid, ser):
        if pi in (0, 1) and cid and ser is not None:
            self.serials[pi].setdefault(cid, set()).add(ser)

    def update(self, obs: dict):
        cur = obs.get("current") or {}
        me = cur.get("yourIndex", 0); opp = 1 - me
        turn = cur.get("turn")
        if turn != self._buff_turn:          # new (owner) turn -> transient buffs expire
            self._buff_turn = turn
            self.opp_def_buff = {}
            self.offense_buff = 0.0
        # (1) reveal logs -- the incremental delta since the previous obs (run on every obs)
        for lg in (obs.get("logs") or []):
            t = lg.get("type")
            if t in self._REVEAL_TYPES:
                self._add(lg.get("playerIndex"), lg.get("cardId"), lg.get("serial"))
            # OPP used a defensive-buff attack on its just-finished turn -> that Pokemon (by serial)
            # takes less damage during MY current turn.
            if t == 15 and lg.get("playerIndex") == opp and lg.get("serial") is not None:
                mag = DEFENSE_BUFF_ATTACKS.get(lg.get("attackId"))
                if mag:
                    self.opp_def_buff[lg["serial"]] = mag
            # WE played an offensive-buff trainer this turn -> our attacks do more THIS turn.
            elif t == 10 and lg.get("playerIndex") == me:
                b = OFFENSE_BUFF_CARDS.get(lg.get("cardId"))
                if b:
                    self.offense_buff = max(self.offense_buff, b)
        # (2) currently-visible public cards (both players' discard/board/stadium)
        for pi, pl in enumerate(cur.get("players") or []):
            if not isinstance(pl, dict):
                continue
            for c in (pl.get("discard") or []):
                self._add(pi, *_id_serial(c))
            for grp in ("active", "bench"):
                for pk in (pl.get(grp) or []):
                    if not pk:
                        continue
                    self._add(pi, pk.get("id"), pk.get("serial"))
                    for c in (pk.get("preEvolution") or []) + (pk.get("tools") or []) \
                            + (pk.get("energyCards") or []):
                        self._add(pi, *_id_serial(c))
        for c in (cur.get("stadium") or []):
            if isinstance(c, dict):
                self._add(c.get("playerIndex"), c.get("id"), c.get("serial"))

    def copies_for(self, player_idx: int) -> dict:
        """{cardId: distinct copies seen} for ``player_idx`` (deduped by serial)."""
        return {cid: len(s) for cid, s in self.serials.get(int(player_idx), {}).items()}


class AbilityTracker:
    """Per-side, per-turn record of which of THIS player's in-play slots used an ability.

    cabt has NO ABILITY log event (the opponent's ability use is therefore unobservable),
    but a player always OBSERVES ITS OWN selections: when it picks an ABILITY option
    (``type==10``) sourced from a board Pokemon, we mark that Pokemon's unit slot. ABILITY
    options identify their source via ``area``/``index`` (verified): ACTIVE area=4 -> slot 0,
    BENCH area=5 -> slot 1+index; area=7 is a Stadium ability (no Pokemon -> skipped).

    Auto-resets when the turn number changes (abilities are once-per-turn). Slot convention
    matches ``TokenEncoder._units``: 0=active, 1+i=bench[i]. ONE tracker per side; each side
    feeds only its own picks and the encoder applies it to that side's own units only (the
    opponent's units always stay 0 -- train==test, since neither train nor inference can see
    the opponent's ability use).
    """
    _ABILITY = 10
    _AREA_ACTIVE, _AREA_BENCH = 4, 5

    def __init__(self):
        self.reset()

    def reset(self):
        self._turn = None
        self.slots: set[int] = set()

    def note_turn(self, turn):
        """Clear the per-turn record when a new turn starts. Call before reading ``slots``."""
        if turn != self._turn:
            self._turn = turn
            self.slots = set()

    def record(self, sel: dict, picked):
        """Mark unit slots for any ABILITY options among ``picked`` (option indices into sel)."""
        opts = (sel or {}).get("option") or []
        for idx in picked:
            if 0 <= idx < len(opts):
                o = opts[idx]
                if o.get("type") == self._ABILITY:
                    a, i = o.get("area"), o.get("index")
                    if a == self._AREA_ACTIVE:
                        self.slots.add(0)
                    elif a == self._AREA_BENCH and isinstance(i, int) and 0 <= i < N_BENCH:
                        self.slots.add(1 + i)


class TokenEncoder:
    """Stateless (given a CardTable) cabt obs -> token-stream numpy arrays.

    ``encode(obs, picked=set())`` mirrors ``encoding.Encoder.encode`` -- it takes
    the set of option indices already chosen this decision so the action mask is
    correct for multi-pick selects.
    """

    def __init__(self, card_table: CardTable | None = None):
        self.cards = card_table or get_card_table()
        self.vocab_size = self.cards.vocab_size
        self.UNK = self.vocab_size          # "unknown card" id (index vocab_size)
        # Reuse encoding.py's option-feature logic verbatim (type one-hot +
        # structural + attack feats + special-condition + zone resolution).
        self._mlp = _MlpEncoder(card_table=self.cards)

    # ---- public shape description (for obs_to_tensors2 / building the net) --
    @property
    def shapes(self) -> dict[str, tuple]:
        return {
            # CLS / global scalars
            "cls_scalars": (G,),
            "select_type": (1,), "select_context": (1,),
            "effect_id": (2,), "effect_mask": (2,),   # [0]=effect card, [1]=contextCard (masked if absent)

            # ---- card-list token streams (id + pad mask) ----
            "self_deck_id": (DECK_SIZE,), "self_deck_mask": (DECK_SIZE,),
            "opp_deck_id": (DECK_SIZE,), "opp_deck_mask": (DECK_SIZE,),
            "self_prize_id": (N_PRIZE,), "self_prize_mask": (N_PRIZE,),
            "opp_prize_id": (N_PRIZE,), "opp_prize_mask": (N_PRIZE,),
            "self_hand_id": (MAX_HAND,), "self_hand_mask": (MAX_HAND,),
            "opp_hand_id": (MAX_HAND,), "opp_hand_mask": (MAX_HAND,),
            "self_discard_id": (MAX_DISCARD,), "self_discard_mask": (MAX_DISCARD,),
            "opp_discard_id": (MAX_DISCARD,), "opp_discard_mask": (MAX_DISCARD,),
            "stadium_id": (N_STADIUM,), "stadium_mask": (N_STADIUM,),

            # ---- in-play unit tokens (active + bench, both sides) ----
            # one row per unit: top id + preevo ids + tool ids + 23-dim attr.
            "self_unit_top_id": (1 + N_BENCH,),
            "self_unit_preevo_id": (1 + N_BENCH, N_PREEVO),
            "self_unit_tool_id": (1 + N_BENCH, N_TOOLS),
            "self_unit_attr": (1 + N_BENCH, UNIT_ATTR),
            "self_unit_mask": (1 + N_BENCH,),
            "opp_unit_top_id": (1 + N_BENCH,),
            "opp_unit_preevo_id": (1 + N_BENCH, N_PREEVO),
            "opp_unit_tool_id": (1 + N_BENCH, N_TOOLS),
            "opp_unit_attr": (1 + N_BENCH, UNIT_ATTR),
            "opp_unit_mask": (1 + N_BENCH,),

            # ---- option tokens (the action candidates) ----
            "opt_src_id": (MAX_OPTIONS,),
            "opt_tgt_id": (MAX_OPTIONS,),
            "opt_attr": (MAX_OPTIONS, OPT_DYN),
            # live state of the board Pokemon each option references (attr + present flag):
            # src = the Pokemon ACTING (ability/attack/retreat); tgt = the Pokemon ACTED ON
            # (attach/evolve/attack-defender). All-zero + flag=0 when none (e.g. trainers).
            "opt_src_unit": (MAX_OPTIONS, OPT_UNIT_DIM),
            "opt_tgt_unit": (MAX_OPTIONS, OPT_UNIT_DIM),

            # legal-action mask over [0..MAX_OPTIONS] (last == submit)
            "action_mask": (N_ACTIONS,),
        }

    @property
    def int_keys(self) -> set[str]:
        return {
            "select_type", "select_context", "effect_id",
            "self_deck_id", "opp_deck_id",
            "self_prize_id", "opp_prize_id",
            "self_hand_id", "opp_hand_id",
            "self_discard_id", "opp_discard_id",
            "stadium_id",
            "self_unit_top_id", "self_unit_preevo_id", "self_unit_tool_id",
            "opp_unit_top_id", "opp_unit_preevo_id", "opp_unit_tool_id",
            "opt_src_id", "opt_tgt_id",
        }

    # ---- helpers ----------------------------------------------------------
    def _id_list(self, cards, n, *, fill_unk=0):
        """A list of cabt Cards -> (ids[n] int64, mask[n] float32).

        Real ids fill from the front; ``fill_unk`` extra UNK tokens follow (for
        face-down / hidden cards whose count is known but identity isn't); the
        rest stay EMPTY(0). The mask is 1 wherever a token is present (real OR
        UNK) so the transformer attends to "a hidden card exists here".
        """
        ids = np.zeros(n, np.int64)
        mask = np.zeros(n, np.float32)
        i = 0
        for c in (cards or []):
            if i >= n:
                break
            ids[i] = _card_id(c)
            mask[i] = 1.0
            i += 1
        for _ in range(int(fill_unk)):
            if i >= n:
                break
            ids[i] = self.UNK
            mask[i] = 1.0
            i += 1
        return ids, mask

    def _unk_list(self, count, n):
        """``count`` hidden cards (UNK) padded to n -> (ids[n], mask[n])."""
        return self._id_list([], n, fill_unk=min(int(count or 0), n))

    def _energy_hist12(self, energies) -> np.ndarray:
        v = np.zeros(N_ENERGY_BINS, np.float32)
        for e in (energies or []):
            if isinstance(e, int) and 0 <= e < N_ENERGY_BINS:
                v[e] += 1.0
        return v

    def _unit(self, pk, *, is_active: bool, player):
        """One in-play Pokemon -> (top_id, preevo_ids[2], tool_ids[2], attr[23], present).

        ``player`` is the owning PlayerState (status booleans live there, applying
        to the ACTIVE Pokemon only).
        """
        top_id = np.zeros((), np.int64)
        preevo = np.zeros(N_PREEVO, np.int64)
        tools = np.zeros(N_TOOLS, np.int64)
        attr = np.zeros(UNIT_ATTR, np.float32)
        if pk is None:
            return int(top_id), preevo, tools, attr, 0.0

        top_id = pk.get("id") or 0
        for i, c in enumerate((pk.get("preEvolution") or [])[:N_PREEVO]):
            preevo[i] = _card_id(c)
        for i, c in enumerate((pk.get("tools") or [])[:N_TOOLS]):
            tools[i] = _card_id(c)

        maxhp = _f(pk.get("maxHp"))
        hp = _f(pk.get("hp"))
        eh = self._energy_hist12(pk.get("energies"))
        n_energy = float(len(pk.get("energies") or []))

        attr[0] = (hp / maxhp) if maxhp > 0 else 0.0          # hp / maxHp
        attr[1] = maxhp / 340.0                                # maxHp / 340
        attr[2:2 + N_ENERGY_BINS] = eh / 8.0                   # 12-bin energy hist /8
        attr[14] = n_energy / 12.0                             # total energy / 12
        # status one-hot (ACTIVE only): poisoned,burned,asleep,paralyzed,confused
        if is_active and player is not None:
            attr[15] = 1.0 if player.get("poisoned") else 0.0
            attr[16] = 1.0 if player.get("burned") else 0.0
            attr[17] = 1.0 if player.get("asleep") else 0.0
            attr[18] = 1.0 if player.get("paralyzed") else 0.0
            attr[19] = 1.0 if player.get("confused") else 0.0
        attr[20] = 1.0 if pk.get("appearThisTurn") else 0.0    # summoning sickness
        # tera flag from static card features if derivable; else 0.
        attr[21] = self._tera_flag(top_id)
        attr[22] = 0.0                                         # ability_used (set by _units from AbilityTracker)
        return int(top_id), preevo, tools, attr, 1.0

    def _tera_flag(self, card_id) -> float:
        """1.0 if the card is a Tera Pokemon, else 0.0 (from static features).

        card_features puts a 4-dim SPECIAL_TAGS one-hot [Ancient,Future,Tera,
        Trainer's] at a fixed offset: 3 (category) + 9 (stage) + 11 (type) +
        11 (weakness) + 4 (rule) = 38, so Tera is feature index 38 + 2 = 40.
        Guarded so a future feature-layout change can't crash the encoder.
        """
        idx = 3 + 9 + 11 + 11 + 4 + 2  # = 40
        feats = self.cards.features(card_id)
        if 0 <= idx < feats.shape[0]:
            return float(feats[idx] > 0.5)
        return 0.0

    def _units(self, pl, ability_slots=None, def_buff=None):
        """A PlayerState -> stacked unit arrays for [active] + N_BENCH bench slots.

        ``ability_slots``: set of unit-slot indices (0=active, 1+i=bench[i]) that used an
        ability THIS turn (this player's own observable use only); sets attr[22]=1 there.
        ``def_buff``: {serial: reduction} of units shielded by a next-turn defensive attack;
        matched by the unit's serial (precise even with duplicate cards) -> attr[23].
        """
        n = 1 + N_BENCH
        top = np.zeros(n, np.int64)
        preevo = np.zeros((n, N_PREEVO), np.int64)
        tools = np.zeros((n, N_TOOLS), np.int64)
        attr = np.zeros((n, UNIT_ATTR), np.float32)
        mask = np.zeros(n, np.float32)
        serials = [None] * n

        active_list = list(pl.get("active") or [])
        act = active_list[0] if active_list else None
        top[0], preevo[0], tools[0], attr[0], mask[0] = self._unit(act, is_active=True, player=pl)
        if act:
            serials[0] = act.get("serial")

        bench = list(pl.get("bench") or [])
        for i in range(N_BENCH):
            pk = bench[i] if i < len(bench) else None
            top[1 + i], preevo[1 + i], tools[1 + i], attr[1 + i], mask[1 + i] = \
                self._unit(pk, is_active=False, player=pl)
            if pk:
                serials[1 + i] = pk.get("serial")

        for slot in (ability_slots or ()):           # ability_used flag (present slots only)
            if 0 <= slot < n and mask[slot] > 0.5:
                attr[slot, 22] = 1.0
        if def_buff:                                  # next-turn damage-reduction (by serial)
            for slot in range(n):
                ser = serials[slot]
                if ser is not None and mask[slot] > 0.5 and ser in def_buff:
                    attr[slot, 23] = min(def_buff[ser], 200) / 200.0
        return top, preevo, tools, attr, mask

    def _visible_self_cards(self, s, mp) -> list:
        """Approximate our 60-card decklist as the UNION of cards we currently see
        we own: hand + discard + board pokemon (top + preevo + tools + energyCards)
        + our taken prizes (which are face-down -> not identifiable, so excluded
        here -- they contribute only to the UNK prize tokens). This is a v1 stand-in.

        TODO: the TRUE 60-card decklist is not in the obs; it should be threaded in
        from the deck registry (rl/decks.py) keyed by which deck we piloted. Until
        then this under-counts (hidden deck + face-down prizes are missing).
        """
        out = []
        out += list(mp.get("hand") or [])
        out += list(mp.get("discard") or [])
        me = s["yourIndex"]
        for grp in ("active", "bench"):
            for pk in (mp.get(grp) or []):
                if pk is None:
                    continue
                if pk.get("id"):
                    out.append(pk.get("id"))
                out += list(pk.get("preEvolution") or [])
                out += list(pk.get("tools") or [])
                out += list(pk.get("energyCards") or [])
        # our stadium in play (if we own it)
        for c in (s.get("stadium") or []):
            if isinstance(c, dict) and c.get("playerIndex") == me:
                out.append(c)
        return out

    def _visible_opp_cards(self, s, op) -> list:
        """Approximate the OPP "revealed so far" set from what we can see them own:
        their discard + their board pokemon (top/preevo/tools/energyCards) + their
        stadium. Their hand/deck/prizes stay hidden (UNK / count-only elsewhere).

        TODO: true log-derived played-card counts belong here.
        """
        out = []
        out += list(op.get("discard") or [])
        opp_idx = 1 - s["yourIndex"]
        for grp in ("active", "bench"):
            for pk in (op.get(grp) or []):
                if pk is None:
                    continue
                if pk.get("id"):
                    out.append(pk.get("id"))
                out += list(pk.get("preEvolution") or [])
                out += list(pk.get("tools") or [])
                out += list(pk.get("energyCards") or [])
        for c in (s.get("stadium") or []):
            if isinstance(c, dict) and c.get("playerIndex") == opp_idx:
                out.append(c)
        return out

    # ---- main entry -------------------------------------------------------
    def encode(self, obs: dict, picked: set[int] | None = None,
               self_deck: list | None = None, tracker: "GameTracker | None" = None,
               ability_slots: "set[int] | None" = None) -> dict[str, np.ndarray]:
        """``self_deck``: our true 60-card decklist ids (else a visible-cards proxy).
        ``tracker``: a GameTracker (updated each step) adding opp cards revealed earlier
        but no longer visible (else only currently-visible opp cards).
        ``ability_slots``: our own unit slots (0=active, 1+i=bench[i]) that used an ability
        this turn -> sets attr[22] on our units only (opp ability use is unobservable)."""
        picked = picked or set()
        s = obs["current"]
        me = s["yourIndex"]
        opp = 1 - me
        mp, op = s["players"][me], s["players"][opp]
        sel = obs["select"]

        out: dict[str, np.ndarray] = {}

        # ---- CLS / global scalars (spec layout) ----
        own_deck = _f(mp.get("deckCount")); opp_deck = _f(op.get("deckCount"))
        own_prize = len(mp.get("prize") or []); opp_prize = len(op.get("prize") or [])
        own_hand = _f(mp.get("handCount")); opp_hand = _f(op.get("handCount"))
        out["cls_scalars"] = np.array([
            _f(s.get("turn")) / _T,
            _f(s.get("turnActionCount")) / 20.0,
            1.0 if s.get("firstPlayer") == me else 0.0,
            1.0 if s.get("supporterPlayed") else 0.0,
            1.0 if s.get("stadiumPlayed") else 0.0,
            1.0 if s.get("energyAttached") else 0.0,
            1.0 if s.get("retreated") else 0.0,
            own_deck / 60.0,
            opp_deck / 60.0,
            own_prize / 6.0,
            opp_prize / 6.0,
            own_hand / 12.0,
            opp_hand / 12.0,
            # select-dynamics (what's going on in THIS decision): restore v1 parity
            _f(sel.get("remainDamageCounter")) / 20.0,
            _f(sel.get("remainEnergyCost")) / 5.0,
            _f(sel.get("minCount")) / 3.0,
            _f(sel.get("maxCount")) / 3.0,
            len(picked) / 3.0,
            (tracker.offense_buff if tracker is not None else 0.0) / 50.0,  # our this-turn attack buff
        ], dtype=np.float32)
        out["select_type"] = np.array([min(sel.get("type", 0), N_SELECT_TYPES - 1)], np.int64)
        out["select_context"] = np.array([min(sel.get("context", 0), N_SELECT_CTX - 1)], np.int64)
        # source-effect tokens: the card driving this select (effect) + the card it concerns
        # (contextCard). Populated only during effect resolution -> masked otherwise.
        eff_id = np.zeros(2, np.int64); eff_mask = np.zeros(2, np.float32)
        for j, key in enumerate(("effect", "contextCard")):
            cid = _card_id(sel.get(key))
            if cid:
                eff_id[j] = cid; eff_mask[j] = 1.0
        out["effect_id"], out["effect_mask"] = eff_id, eff_mask

        # ---- card-list streams ----
        # decklists (approximate; see _visible_*_cards TODO)
        # our decklist: the TRUE 60-card list if threaded in, else the visible-cards proxy
        self_cards = list(self_deck) if self_deck else self._visible_self_cards(s, mp)
        out["self_deck_id"], out["self_deck_mask"] = self._id_list(self_cards, DECK_SIZE)
        # opp "revealed deck composition": with a tracker, emit each card as many tokens as
        # DISTINCT SERIALS seen = true copies (logs + visible, serial-deduped) -- so a recycled
        # card counts once and a discarded card isn't double-counted vs its reveal log. Without
        # a tracker (e.g. self-tests), fall back to the currently-visible cards.
        if tracker is not None:
            opp_cards = []
            for cid, n in tracker.copies_for(opp).items():
                opp_cards.extend([cid] * n)
        else:
            opp_cards = self._visible_opp_cards(s, op)
        out["opp_deck_id"], out["opp_deck_mask"] = self._id_list(opp_cards, DECK_SIZE)

        # prizes: face-down -> UNK tokens, count = remaining prize slots
        out["self_prize_id"], out["self_prize_mask"] = self._unk_list(own_prize, N_PRIZE)
        out["opp_prize_id"], out["opp_prize_mask"] = self._unk_list(opp_prize, N_PRIZE)

        # hands: ours visible (real ids); opp hidden -> handCount UNK tokens
        out["self_hand_id"], out["self_hand_mask"] = self._id_list(mp.get("hand"), MAX_HAND)
        out["opp_hand_id"], out["opp_hand_mask"] = self._unk_list(opp_hand, MAX_HAND)

        # discards: both public, real ids
        out["self_discard_id"], out["self_discard_mask"] = self._id_list(mp.get("discard"), MAX_DISCARD)
        out["opp_discard_id"], out["opp_discard_mask"] = self._id_list(op.get("discard"), MAX_DISCARD)

        # stadium: slot0 = self-owned, slot1 = opp-owned (position encodes owner)
        stad_id = np.zeros(N_STADIUM, np.int64)
        stad_mask = np.zeros(N_STADIUM, np.float32)
        for c in (s.get("stadium") or []):
            if not isinstance(c, dict):
                continue
            slot = 0 if c.get("playerIndex") == me else 1
            stad_id[slot] = c.get("id") or 0
            stad_mask[slot] = 1.0
        out["stadium_id"], out["stadium_mask"] = stad_id, stad_mask

        # ---- in-play unit tokens ----  (ability_used -> OUR units; def_buff matches by serial,
        # so only the OPP's shielded units light up; passing it to both is harmless)
        def_buff = tracker.opp_def_buff if tracker is not None else None
        (out["self_unit_top_id"], out["self_unit_preevo_id"], out["self_unit_tool_id"],
         out["self_unit_attr"], out["self_unit_mask"]) = self._units(mp, ability_slots, def_buff)
        (out["opp_unit_top_id"], out["opp_unit_preevo_id"], out["opp_unit_tool_id"],
         out["opp_unit_attr"], out["opp_unit_mask"]) = self._units(op, None, def_buff)

        # ---- option tokens (reuse encoding.py's resolution + feature logic) ----
        deck_list = sel.get("deck") if isinstance(sel.get("deck"), list) else None
        opt_src = np.zeros(MAX_OPTIONS, np.int64)
        opt_tgt = np.zeros(MAX_OPTIONS, np.int64)
        opt_attr = np.zeros((MAX_OPTIONS, OPT_DYN), np.float32)
        opt_src_unit = np.zeros((MAX_OPTIONS, OPT_UNIT_DIM), np.float32)
        opt_tgt_unit = np.zeros((MAX_OPTIONS, OPT_UNIT_DIM), np.float32)
        # side -> (unit attr array, unit present mask) for gathering referenced-unit live state
        unit_attr = {me: out["self_unit_attr"], opp: out["opp_unit_attr"]}
        unit_mask = {me: out["self_unit_mask"], opp: out["opp_unit_mask"]}
        for i, o in enumerate(sel["option"][:MAX_OPTIONS]):
            src = self._mlp._resolve_card(o, s, me, deck_list)
            dyn, _static, sid = self._mlp._option_row(o, src)
            opt_src[i] = sid
            opt_attr[i] = dyn
            opt_tgt[i] = self._mlp._resolve_target(o, s, me)
            # gather live state of the board Pokemon this option references:
            #   source via (area,index), target via (inPlayArea,inPlayIndex); skip non-board refs.
            p = o.get("playerIndex")
            side = me if p is None else p
            src_slot, src_side = _board_slot(o.get("area"), o.get("index")), side
            tgt_slot, tgt_side = _board_slot(o.get("inPlayArea"), o.get("inPlayIndex")), side
            if o.get("type") == 13:           # ATTACK: attacker/defender are implicit actives (no area)
                if src_slot is None:
                    src_slot, src_side = 0, me     # our active attacks
                if tgt_slot is None:
                    tgt_slot, tgt_side = 0, opp    # opp's active defends (slot 0)
            for slot, sd, dst in ((src_slot, src_side, opt_src_unit),
                                  (tgt_slot, tgt_side, opt_tgt_unit)):
                attr_arr = unit_attr.get(sd)
                if (slot is not None and attr_arr is not None and slot < attr_arr.shape[0]
                        and unit_mask[sd][slot] > 0.5):
                    dst[i, :UNIT_ATTR] = attr_arr[slot]
                    dst[i, UNIT_ATTR] = 1.0          # present flag
        out["opt_src_id"], out["opt_tgt_id"], out["opt_attr"] = opt_src, opt_tgt, opt_attr
        out["opt_src_unit"], out["opt_tgt_unit"] = opt_src_unit, opt_tgt_unit

        # ---- legal-action mask (reused contract) ----
        out["action_mask"] = build_mask(sel, picked)

        # clamp every CARD id into [0, UNK]: a card absent from EN_Card_Data.csv
        # (id >= vocab_size) maps to the learnable UNK row instead of crashing the
        # embedding or colliding past it. EMPTY(0) and UNK(vocab_size) are preserved.
        for k in self.int_keys:
            if k in ("select_type", "select_context"):   # select indices, not card ids
                continue
            np.clip(out[k], 0, self.UNK, out=out[k])
        return out


if __name__ == "__main__":
    # ---- SELF-TEST: encode real cabt obs from notes/vcalib_pool.pkl ----
    import os
    import pickle

    enc = TokenEncoder()
    pool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "notes", "vcalib_pool.pkl")
    pool = pickle.load(open(pool_path, "rb"))
    print(f"loaded {len(pool)} records; vocab_size={enc.vocab_size} UNK={enc.UNK}")

    shapes = enc.shapes
    int_keys = enc.int_keys

    n_test = min(3, len(pool))
    for ri in range(n_test):
        out = enc.encode(pool[ri]["root"], picked=set())
        # every declared key present, no extras
        assert set(out.keys()) == set(shapes.keys()), (
            f"key mismatch: missing={set(shapes) - set(out)} extra={set(out) - set(shapes)}")
        for k, arr in out.items():
            assert arr.shape == shapes[k], f"[{k}] shape {arr.shape} != {shapes[k]}"
            want = np.int64 if k in int_keys else np.float32
            assert arr.dtype == want, f"[{k}] dtype {arr.dtype} != {want}"
            if k in int_keys:
                assert (arr >= 0).all() and (arr <= enc.UNK).all(), f"[{k}] id out of [0,UNK]"
        print(f"  record {ri}: OK  ({len(out)} keys)")

    print("\nfull .shapes layout (key -> shape, dtype):")
    for k in shapes:
        dt = "int64" if k in int_keys else "f32"
        print(f"  {k:22s} {str(shapes[k]):14s} {dt}")
    print("\nint_keys:", sorted(int_keys))
    print("\nSELF-TEST PASSED")
