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

try:    # per-attack properties (base_damage / is_variable / energy_cost / has_effect); bundle-safe
    from .attack_data import ATTACKS as _ATTACKS
except Exception:
    _ATTACKS = {}

try:    # transient turn-buff tables for the v2 token encoder (bundle-safe; empty -> feature stays 0)
    from .buff_data import DEFENSE_BUFF_ATTACKS, OFFENSE_BUFF_CARDS
except Exception:
    DEFENSE_BUFF_ATTACKS, OFFENSE_BUFF_CARDS = {}, {}

# ---- shape/layout constants live in enc_constants.py (single source of truth) ----
# re-exported here so existing `from .encoding import <CONST>` callers keep working.
from .enc_constants import (        # noqa: F401,E402
    N_BENCH, MAX_HAND, MAX_DISCARD_MLP, MAX_OPTIONS, N_OPT_TYPES, N_SELECT_TYPES, N_SELECT_CTX,
    SUBMIT_ACTION, N_ACTIONS, OPT_STRUCT, MAX_ATTACK, OPT_DYN,
    _T, _CNT, _DECK, _PRIZE, _DISCARD,
    N_ENERGY_BINS, UNIT_ATTR, N_PREEVO, N_TOOLS, N_ENERGY_CARDS,
    DECK_SIZE, N_PRIZE, MAX_DISCARD, N_STADIUM, G,
    _TOKEN_LAYOUT, _OFF, N_STATE_TOKENS,
)


# int64 obs keys that are NOT card ids (so they're exempt from the [0, UNK] card-id clamp):
# select indices, token positions (may be -1 == none), OptionType id, attackId (indexes attack_emb).
_NON_CARD_INT_KEYS = frozenset({
    "select_type", "select_context", "opt_src_pos", "opt_tgt_pos", "opt_verb", "opt_attack_id",
})


def _f(x) -> float:
    return float(x) if x is not None else 0.0


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


# ===========================================================================
#  TOKEN ENCODER  (the live encoder; the legacy mlp Encoder was removed)
#  TokenTransformer's pointer-list encoder + per-game GameTracker/AbilityTracker.
#  Uses the module-level option-resolution helpers above (attack_feats/option_row/
#  resolve_card/zone_card/zone_pokemon).
# ===========================================================================

# (token constants + _TOKEN_LAYOUT/_OFF/N_STATE_TOKENS live in enc_constants.py,
#  imported at the top of this module.)


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


def _unit_pos(area, index, owner_self) -> int:
    """Global token position of the in-play Pokemon at (area,index) in owner's board; -1 if none."""
    slot = _board_slot(area, index)
    if slot is None:
        return -1
    return _OFF["self_units" if owner_self else "opp_units"] + slot


def _ref_pos(area, index, owner_self) -> int:
    """Global token position of the card/unit at (area,index) in owner's zone; -1 if unresolvable.
    Maps a cabt AreaType to the matching token stream so an option can point at the EXACT
    pre-encoder token of the card it references (board unit / hand / deck / discard / stadium)."""
    slot = _board_slot(area, index)
    if slot is not None:                                  # ACTIVE/BENCH -> unit token
        return _OFF["self_units" if owner_self else "opp_units"] + slot
    if not isinstance(index, int) or index < 0:
        return -1
    if area == 2:                                         # HAND
        return _OFF["self_hand" if owner_self else "opp_hand"] + min(index, MAX_HAND - 1)
    # DECK (area 1) is intentionally absent: a deck-search option indexes the SHOWN subset
    # (sel['deck']), NOT the self_deck stream order -> no aligned token; the caller synths from id.
    if area == 3:                                         # DISCARD
        return _OFF["self_discard" if owner_self else "opp_discard"] + min(index, MAX_DISCARD - 1)
    # STADIUM (area 7) is intentionally absent: its slot is set by the stadium's REAL owner, not the
    # option's owner_self (stadium-ability options carry playerIndex=None -> owner_self would always
    # pick the self slot even when the OPPONENT owns the stadium). The encode loop resolves it.
    return -1


# ===========================================================================
# Option resolution -- stateless helpers over a cabt observation.
# (Formerly methods of the v1 Encoder, borrowed by the v2 TokenEncoder via
#  self._mlp; extracted to module functions so the token encoder stands alone.)
# ===========================================================================
def attack_feats(o: dict) -> list[float]:
    """ATTACK option -> [base_dmg/350, is_variable, energy_cost/5, has_effect]; zeros otherwise.
    Variable/conditional damage is FLAGGED (not faked); the engine-sim would_ko feature (a
    SEPARATE appended column, see option_row) resolves the realized KO with abilities/stadium/
    weakness/variable that base_dmg can't."""
    aid = o.get("attackId")
    a = _ATTACKS.get(aid) if aid is not None else None
    if a is None:
        return [0.0, 0.0, 0.0, 0.0]
    dmg, var, cost, eff = a
    return [min(dmg, 350) / 350.0, float(var), min(cost, 5) / 5.0, float(eff)]


def option_row(o: dict, cid: int, cards) -> tuple[np.ndarray, np.ndarray, int]:
    """``cid`` is the option's resolved (source) card id (0 if none); ``cards`` = the CardTable."""
    cid = cid or 0
    t = np.zeros(N_OPT_TYPES, np.float32)
    ot = o.get("type", 0)
    if 0 <= ot < N_OPT_TYPES:
        t[ot] = 1.0
    sc = [0.0] * 5                              # special-condition one-hot (poison..confuse)
    sct = o.get("specialConditionType")
    if isinstance(sct, int) and 0 <= sct < 5:
        sc[sct] = 1.0
    dyn = np.array([
        *t,                                    # 16  option type (-> opt_verb; dropped from opt_attr)
        min(_f(o.get("count")) / 5.0, 1.0),    # 2   action params -- the only structural dims kept.
        min(_f(o.get("number")) / _CNT, 1.0),  #     positions are gathered via opt_src/tgt_pos and
                                               #     attack identity via the attack_emb (opt_attack_id),
                                               #     so the old positional/flag/attackId scalars are gone.
        *attack_feats(o),                      # 4   attack: dmg / variable / cost / effect
        *sc,                                   # 5   special condition
        float(o.get("would_ko", 0.0)),         # 1   engine-sim would-KO
    ], dtype=np.float32)
    return dyn, cards.features(cid), cid


def zone_card(s, area, index, player, deck_list) -> int:
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


def zone_pokemon(s, area, index, player):
    """The in-play Pokemon dict at (area, index) in ``player``'s board, else None.
    area 4 == ACTIVE (active[0]); area 5 == BENCH (bench[index])."""
    players = s.get("players") or []
    if not (0 <= player < len(players)):
        return None
    pl = players[player]
    if area == 4:
        act = pl.get("active") or []
        return act[0] if act else None
    if area == 5 and isinstance(index, int):
        bench = pl.get("bench") or []
        if 0 <= index < len(bench):
            return bench[index]
    return None


def resolve_card(o: dict, s, me, deck_list) -> int:
    """The SOURCE card an option acts on: explicit cardId, else the card at the option's
    (area,index) in its owner's zone (hand/deck/discard/active/bench). For TOOL_CARD(4)/
    ENERGY_CARD(5)/ENERGY(6), area/index point at the HOST Pokemon and toolIndex/energyIndex
    select WHICH attached card -> resolve the SELECTED card's identity. PLAY (type 7) carries
    only a bare ``index`` (a HAND slot) with NO ``area`` -> default area=HAND so the policy
    isn't blind to which card it plays (the most common option type)."""
    cid = o.get("cardId")
    if cid is not None:
        return cid
    ot = o.get("type")
    p = o.get("playerIndex")
    owner = me if p is None else p
    if ot in (4, 5, 6):
        pk = zone_pokemon(s, o.get("area"), o.get("index"), owner)
        if pk is not None:
            if ot == 4:
                sub, j = (pk.get("tools") or []), o.get("toolIndex")
            else:
                sub, j = (pk.get("energyCards") or []), o.get("energyIndex")
            if isinstance(j, int) and 0 <= j < len(sub):
                return _card_id(sub[j])
        # else fall through to the host-Pokemon id (indices missing/out of range)
    area = o.get("area")
    if area is None and ot == 7:              # PLAY: index is a HAND slot (AreaType.HAND==2)
        area = 2
    return zone_card(s, area, o.get("index"), owner, deck_list)


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
        # playerIndex -> {serial: last-known AreaType} and {serial: cardId}. The zone lets us
        # tell a revealed card the player is HOLDING IN HAND (about to play) from one buried in
        # deck/discard -- the most actionable hidden-state signal. HAND zones come from MOVE_CARD
        # logs; every currently-visible card overwrites its serial with ground truth.
        self.zone: dict[int, dict] = {0: {}, 1: {}}
        self.card_of: dict[int, dict] = {0: {}, 1: {}}
        # transient turn-buffs (reset each owner turn): see DEFENSE/OFFENSE buff tables.
        self._buff_turn = None
        self.opp_def_buff: dict = {}    # serial -> reduction: OPP units shielded during MY current turn
        self.offense_buff: float = 0.0  # MY this-turn extra attack damage (from a trainer I played)

    def _add(self, pi, cid, ser):
        if pi in (0, 1) and cid and ser is not None:
            self.serials[pi].setdefault(cid, set()).add(ser)
            self.card_of[pi][ser] = cid

    def _set_zone(self, pi, ser, area):
        if pi in (0, 1) and ser is not None and area is not None:
            self.zone[pi][ser] = int(area)

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
            if t == 6:                            # MOVE_CARD: toArea tells us where it went (e.g. -> HAND)
                self._set_zone(lg.get("playerIndex"), lg.get("serial"), lg.get("toArea"))
            if t == 7 and lg.get("fromArea") == 2:   # MOVE_CARD_REVERSE: a card left HAND face-down (no
                # serial -- e.g. Iono/Judge shuffle-hand). Our tracked known-in-hand cards for that
                # player are now stale, so evict them (-> UNK) rather than ghost them into opp_hand.
                pi = lg.get("playerIndex")
                if pi in (0, 1):
                    self.zone[pi] = {ser: a for ser, a in self.zone[pi].items() if a != 2}  # 2 == HAND
            # OPP used a defensive-buff attack on its just-finished turn -> that Pokemon (by serial)
            # takes less damage during MY current turn.
            if t == 15 and lg.get("playerIndex") == opp and lg.get("serial") is not None:
                mag = DEFENSE_BUFF_ATTACKS.get(lg.get("attackId"))
                if mag:
                    self.opp_def_buff[lg["serial"]] = mag
            # WE played an offensive-buff trainer this turn -> our attacks do more THIS turn.
            # ACCUMULATE (not max): these can be played multiple times in a turn, and the bonus stacks.
            elif t == 10 and lg.get("playerIndex") == me:
                b = OFFENSE_BUFF_CARDS.get(lg.get("cardId"))
                if b:
                    self.offense_buff += b
        # (2) currently-visible public cards (both players' discard/board/stadium). Each visible
        # card is GROUND TRUTH for its zone, so it overrides any stale log-derived zone (a card
        # played out of hand re-appears here on board/discard and stops counting as in-hand).
        for pi, pl in enumerate(cur.get("players") or []):
            if not isinstance(pl, dict):
                continue
            for c in (pl.get("discard") or []):
                cid, ser = _id_serial(c)
                self._add(pi, cid, ser); self._set_zone(pi, ser, 3)            # DISCARD
            for grp, area in (("active", 4), ("bench", 5)):                    # ACTIVE / BENCH
                for pk in (pl.get(grp) or []):
                    if not pk:
                        continue
                    self._add(pi, pk.get("id"), pk.get("serial"))
                    self._set_zone(pi, pk.get("serial"), area)
                    for c in (pk.get("preEvolution") or []) + (pk.get("tools") or []) \
                            + (pk.get("energyCards") or []):
                        cid, ser = _id_serial(c)
                        self._add(pi, cid, ser); self._set_zone(pi, ser, area)
        for c in (cur.get("stadium") or []):
            if isinstance(c, dict):
                self._add(c.get("playerIndex"), c.get("id"), c.get("serial"))
                self._set_zone(c.get("playerIndex"), c.get("serial"), 7)       # STADIUM

    def copies_for(self, player_idx: int) -> dict:
        """{cardId: distinct copies seen} for ``player_idx`` (deduped by serial)."""
        return {cid: len(s) for cid, s in self.serials.get(int(player_idx), {}).items()}

    def hand_ids_for(self, player_idx: int) -> list:
        """cardIds ``player_idx`` is KNOWN to currently hold in hand (last-known zone == HAND).
        These are revealed cards (e.g. searched/drawn-to-hand with a visible id) that the player
        is HOLDING and may play next turn -- distinct from cards merely known to be in their deck."""
        pi = int(player_idx)
        z, co = self.zone.get(pi, {}), self.card_of.get(pi, {})
        return [co[ser] for ser, area in z.items() if area == 2 and ser in co]   # 2 == AreaType.HAND


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

    ``encode(obs, picked=set())`` takes the set of option indices already chosen
    this decision so the action mask is correct for multi-pick selects.
    """

    def __init__(self, card_table: CardTable | None = None):
        self.cards = card_table or get_card_table()
        self.vocab_size = self.cards.vocab_size
        self.UNK = self.vocab_size          # "unknown card" id (index vocab_size)

    # ---- public shape description (for obs_to_tensors / building the net) --
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
            "self_unit_energy_id": (1 + N_BENCH, N_ENERGY_CARDS),
            "self_unit_attr": (1 + N_BENCH, UNIT_ATTR),
            "self_unit_mask": (1 + N_BENCH,),
            "opp_unit_top_id": (1 + N_BENCH,),
            "opp_unit_preevo_id": (1 + N_BENCH, N_PREEVO),
            "opp_unit_tool_id": (1 + N_BENCH, N_TOOLS),
            "opp_unit_energy_id": (1 + N_BENCH, N_ENERGY_CARDS),
            "opp_unit_attr": (1 + N_BENCH, UNIT_ATTR),
            "opp_unit_mask": (1 + N_BENCH,),

            # ---- option tokens (the action candidates) ----
            "opt_attr": (MAX_OPTIONS, OPT_STRUCT),   # structural + attack + condition (verb is opt_verb)
            "opt_verb": (MAX_OPTIONS,),              # OptionType id -> its own learned embedding in the net
            # token POSITION (into the pre-encoder state sequence, 0..N_STATE_TOKENS-1; -1 == none)
            # of the card/unit each option references as source/target. The net gathers that EXACT
            # pre-encoder token and feeds it through opt_src/tgt_proj -- the option's view of a card
            # is identical to that card's own token. src = the card driving the action; tgt = the
            # board Pokemon affected. (which-energy/count etc. are disambiguated by opt_attr.)
            "opt_src_pos": (MAX_OPTIONS,),
            "opt_tgt_pos": (MAX_OPTIONS,),
            # resolved card id for a referenced card with NO sequence token (an attached energy/tool);
            # the net synthesizes that card's token from this id. 0 == none (the pos points at a token).
            "opt_src_card": (MAX_OPTIONS,),
            "opt_tgt_card": (MAX_OPTIONS,),
            "opt_attack_id": (MAX_OPTIONS,),  # attackId per option -> learned attack_emb (0 = no-attack)

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
            "self_unit_top_id", "self_unit_preevo_id", "self_unit_tool_id", "self_unit_energy_id",
            "opp_unit_top_id", "opp_unit_preevo_id", "opp_unit_tool_id", "opp_unit_energy_id",
            "opt_src_pos", "opt_tgt_pos",     # positions (clamp-skipped); the *_card below are card ids
            "opt_src_card", "opt_tgt_card",
            "opt_verb",                       # OptionType id (clamp-skipped; not a card id)
            "opt_attack_id",                  # attackId (clamp-skipped; indexes attack_emb, not card_emb)
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
        """One in-play Pokemon -> (top_id, preevo_ids[2], tool_ids[2], energy_ids[4], attr[24], present).

        ``player`` is the owning PlayerState (status booleans live there, applying
        to the ACTIVE Pokemon only).
        """
        top_id = np.zeros((), np.int64)
        preevo = np.zeros(N_PREEVO, np.int64)
        tools = np.zeros(N_TOOLS, np.int64)
        energy = np.zeros(N_ENERGY_CARDS, np.int64)
        attr = np.zeros(UNIT_ATTR, np.float32)
        if pk is None:
            return int(top_id), preevo, tools, energy, attr, 0.0

        top_id = pk.get("id") or 0
        for i, c in enumerate((pk.get("preEvolution") or [])[:N_PREEVO]):
            preevo[i] = _card_id(c)
        for i, c in enumerate((pk.get("tools") or [])[:N_TOOLS]):
            tools[i] = _card_id(c)
        # attached energy CARD identities (special energies differ from basic of the same color);
        # the color histogram (attr[2:14]) + count (attr[14]) stay -- this adds IDENTITY.
        for i, c in enumerate((pk.get("energyCards") or [])[:N_ENERGY_CARDS]):
            energy[i] = _card_id(c)

        maxhp = _f(pk.get("maxHp"))
        hp = _f(pk.get("hp"))
        eh = self._energy_hist12(pk.get("energies"))
        n_energy = float(len(pk.get("energies") or []))

        attr[0] = (hp / maxhp) if maxhp > 0 else 0.0          # hp / maxHp
        attr[1] = maxhp / 500.0                                # effective in-play maxHp /500 (tool-boostable; base printed maxHp is a static feature in policy)
        attr[2:2 + N_ENERGY_BINS] = np.minimum(eh / 4.0, 1.0)  # 12-bin energy hist /4, clipped (was /8 -> wasted range)
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
        return int(top_id), preevo, tools, energy, attr, 1.0

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
        energy = np.zeros((n, N_ENERGY_CARDS), np.int64)
        attr = np.zeros((n, UNIT_ATTR), np.float32)
        mask = np.zeros(n, np.float32)
        serials = [None] * n

        active_list = list(pl.get("active") or [])
        act = active_list[0] if active_list else None
        top[0], preevo[0], tools[0], energy[0], attr[0], mask[0] = self._unit(act, is_active=True, player=pl)
        if act:
            serials[0] = act.get("serial")

        bench = list(pl.get("bench") or [])
        for i in range(N_BENCH):
            pk = bench[i] if i < len(bench) else None
            top[1 + i], preevo[1 + i], tools[1 + i], energy[1 + i], attr[1 + i], mask[1 + i] = \
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
        return top, preevo, tools, energy, attr, mask

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
            _f(sel.get("remainDamageCounter")) / 20.0,   # ceiling ~13 (Dusknoir); keep the larger scale
            _f(sel.get("remainEnergyCost")) / 5.0,        # ceiling 5 (max attack cost) -> exactly 1.0
            # counts: /5 (shared scale w/ energy cost; obs max maxCount=5 -> 1.0) + clamp, since
            # maxCount is hard-capped at len(option) (~17 on detach-all-energy) -> bound the rare tail.
            min(_f(sel.get("minCount")) / 5.0, 1.0),
            min(_f(sel.get("maxCount")) / 5.0, 1.0),
            min(len(picked) / 5.0, 1.0),
            min((tracker.offense_buff if tracker is not None else 0.0) / 100.0, 1.0),  # our this-turn attack buff (stacks; /100 + clamp)
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
        # Symmetric to our known 60-card decklist: the opp deck is a FULL 60 slots that start
        # all-UNK and fill in with revealed ids as the game plays (the rest stay UNK, not empty),
        # so "we know N of the opp's 60 cards, the other 60-N are unknown" is represented.
        out["opp_deck_id"], out["opp_deck_mask"] = self._id_list(
            opp_cards[:DECK_SIZE], DECK_SIZE, fill_unk=DECK_SIZE - min(len(opp_cards), DECK_SIZE))

        # prizes: face-down -> UNK tokens, count = remaining prize slots
        out["self_prize_id"], out["self_prize_mask"] = self._unk_list(own_prize, N_PRIZE)
        out["opp_prize_id"], out["opp_prize_mask"] = self._unk_list(opp_prize, N_PRIZE)

        # hands: ours visible (real ids); opp hidden -> handCount UNK tokens
        out["self_hand_id"], out["self_hand_mask"] = self._id_list(mp.get("hand"), MAX_HAND)
        # opp hand: cards we KNOW they're holding (zone==HAND, from reveal logs) placed first,
        # then UNK up to handCount -- so "holding a known threat, about to play it" is visible
        # (distinct from a card merely known to be in their deck, which lives in opp_deck).
        opp_known_hand = tracker.hand_ids_for(opp) if tracker is not None else []
        _nkn = min(len(opp_known_hand), int(opp_hand))
        out["opp_hand_id"], out["opp_hand_mask"] = self._id_list(
            opp_known_hand[:_nkn], MAX_HAND, fill_unk=max(0, min(int(opp_hand), MAX_HAND) - _nkn))

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
         out["self_unit_energy_id"], out["self_unit_attr"], out["self_unit_mask"]) = \
            self._units(mp, ability_slots, def_buff)
        (out["opp_unit_top_id"], out["opp_unit_preevo_id"], out["opp_unit_tool_id"],
         out["opp_unit_energy_id"], out["opp_unit_attr"], out["opp_unit_mask"]) = \
            self._units(op, None, def_buff)

        # ---- option tokens (reuse encoding.py's resolution + feature logic) ----
        deck_list = sel.get("deck") if isinstance(sel.get("deck"), list) else None
        opt_attr = np.zeros((MAX_OPTIONS, OPT_STRUCT), np.float32)
        opt_verb = np.zeros(MAX_OPTIONS, np.int64)    # OptionType id -> its own embedding in the net
        opt_attack_id = np.zeros(MAX_OPTIONS, np.int64)   # attackId -> learned attack_emb (0 = no-attack/padding)
        # Each option POINTS at the exact pre-encoder token of the card/unit it references:
        #   SOURCE = the card driving the action: ATTACK -> our active; an effect-driven select
        #     (e.g. "discard an energy") -> the EFFECT card that caused it; else the played/used
        #     card at (area,index).
        #   TARGET = the board Pokemon affected: (inPlayArea,inPlayIndex), or the (area,index) host
        #     for attached-card selects; ATTACK -> opp active.
        # Fine distinctions among options sharing the same src/tgt (which energy, count, ...) are
        # carried by opt_attr (the per-option structural table), not by a separate card token.
        opt_src_pos = np.full(MAX_OPTIONS, -1, np.int64)
        opt_tgt_pos = np.full(MAX_OPTIONS, -1, np.int64)
        # resolved id of a referenced card that has NO sequence token (an ATTACHED energy/tool,
        # summed into its unit's bag) -> the net synthesizes that card's token from this id. 0 = none.
        opt_src_card = np.zeros(MAX_OPTIONS, np.int64)
        opt_tgt_card = np.zeros(MAX_OPTIONS, np.int64)
        has_effect = _card_id(sel.get("effect")) != 0
        # the single in-play stadium's token position, by its REAL owner (slot 0 self / 1 opp) --
        # used for stadium-ability options, whose playerIndex is None (owner_self can't tell).
        _stad0 = (s.get("stadium") or [None])[0]
        stad_pos = (_OFF["stadium"] + (0 if (isinstance(_stad0, dict) and _stad0.get("playerIndex") == me) else 1)) \
            if isinstance(_stad0, dict) else -1
        for i, o in enumerate(sel["option"][:MAX_OPTIONS]):
            src_cid = resolve_card(o, s, me, deck_list)
            opt_attr[i] = option_row(o, src_cid, self.cards)[0][N_OPT_TYPES:]   # drop verb one-hot
            ot = o.get("type")
            opt_verb[i] = min(int(ot) if isinstance(ot, int) else 0, N_OPT_TYPES - 1)
            aid = o.get("attackId")
            opt_attack_id[i] = min(int(aid), MAX_ATTACK - 1) if isinstance(aid, int) and aid > 0 else 0
            p = o.get("playerIndex")
            owner_self = (p is None or p == me)
            if ot in (4, 5, 6):                           # discard/select an ATTACHED energy/tool
                # MIRROR of ATTACH: src = the Pokemon we remove it FROM (its unit token); tgt = the
                # SPECIFIC attached card -- it has no sequence token, so carry its id for synthesis.
                opt_src_pos[i] = _unit_pos(o.get("area"), o.get("index"), owner_self)
                pk = zone_pokemon(s, o.get("area"), o.get("index"), me if p is None else p)
                if pk is not None:
                    sub = (pk.get("tools") or []) if ot == 4 else (pk.get("energyCards") or [])
                    j = o.get("toolIndex") if ot == 4 else o.get("energyIndex")
                    if isinstance(j, int) and 0 <= j < len(sub):
                        opt_tgt_card[i] = _card_id(sub[j])
            elif ot == 13:                                # ATTACK: our active -> opp active
                opt_src_pos[i] = _OFF["self_units"]
                opt_tgt_pos[i] = _OFF["opp_units"]
            else:
                # SOURCE = the card the option acts on/with -> its OWN sequence token when the zone is
                # index-aligned (HAND / DISCARD / board / stadium, incl. "select from discard" and
                # "discard a hand card"); a DECK-search pick indexes the shown subset (no aligned
                # token) -> synth from the resolved id; an option with NO object card (YES/NO/COUNT)
                # -> the effect card driving the select.
                area = o.get("area")
                if area is None and ot == 7:              # PLAY: index is a HAND slot
                    area = 2
                pos = stad_pos if area == 7 else _ref_pos(area, o.get("index"), owner_self)
                if pos >= 0:
                    opt_src_pos[i] = pos
                elif src_cid:
                    opt_src_card[i] = src_cid             # deck-search pick -> synth from id
                elif has_effect:
                    opt_src_pos[i] = _OFF["effect"]       # no object card -> effect context
                # TARGET: the affected board Pokemon
                opt_tgt_pos[i] = _unit_pos(o.get("inPlayArea"), o.get("inPlayIndex"), owner_self)
        out["opt_attr"], out["opt_verb"] = opt_attr, opt_verb
        out["opt_src_pos"], out["opt_tgt_pos"] = opt_src_pos, opt_tgt_pos
        out["opt_src_card"], out["opt_tgt_card"] = opt_src_card, opt_tgt_card
        out["opt_attack_id"] = opt_attack_id

        # ---- legal-action mask (reused contract) ----
        out["action_mask"] = build_mask(sel, picked)

        # clamp every CARD id into [0, UNK]: a card absent from EN_Card_Data.csv
        # (id >= vocab_size) maps to the learnable UNK row instead of crashing the
        # embedding or colliding past it. EMPTY(0) and UNK(vocab_size) are preserved.
        for k in self.int_keys:
            if k in _NON_CARD_INT_KEYS:                    # not card ids -> exempt from the card-id clamp
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
            if k in int_keys and k not in _NON_CARD_INT_KEYS:
                assert (arr >= 0).all() and (arr <= enc.UNK).all(), f"[{k}] id out of [0,UNK]"
        print(f"  record {ri}: OK  ({len(out)} keys)")

    print("\nfull .shapes layout (key -> shape, dtype):")
    for k in shapes:
        dt = "int64" if k in int_keys else "f32"
        print(f"  {k:22s} {str(shapes[k]):14s} {dt}")
    print("\nint_keys:", sorted(int_keys))
    print("\nSELF-TEST PASSED")
