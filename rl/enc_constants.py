"""Shape/layout constants for the encoders + token model -- the single source of truth.

Two groups: v1 (mlp Encoder) and v2 (TokenEncoder / TokenTransformer). They share several
names (MAX_HAND, MAX_OPTIONS, N_OPT_TYPES, ...). `MAX_DISCARD_MLP` (v1 window) is intentionally
distinct from `MAX_DISCARD` (v2 public discard budget) -- do not conflate. Imported by
rl/encoding.py (which re-exports them) and rl/policy.py.
"""

# ---- v1 (mlp) + shared shape constants -------------------------------------
N_BENCH = 5          # bench slots (active is encoded separately)
MAX_HAND = 20        # observed max hand ~17
MAX_DISCARD_MLP = 30     # v1 mlp discard window (v2 token encoder uses its own MAX_DISCARD below)
MAX_OPTIONS = 128    # ladder emitted 98-option selects; 128 gives headroom (was 96).
                     # Pointer-scored, so widening costs no weights; search_agent still
                     # caps/pads/masks beyond this as a bound-free backstop.
                     # index MAX_OPTIONS == "submit"
N_OPT_TYPES = 17     # OptionType range 0..16 (incl SPECIAL_CONDITION=16)
N_SELECT_TYPES = 16  # SelectType range
N_SELECT_CTX = 64    # SelectContext embedding vocab cap

SUBMIT_ACTION = MAX_OPTIONS  # the extra action index meaning "stop / submit set"
N_ACTIONS = MAX_OPTIONS + 1

# opt feature layout: 2 action-params(count/number) + 4 attack(dmg/var/cost/effect) + 5 special-
# condition + 1 would_ko. Positional refs are gathered via opt_src/tgt_pos; option-type via the
# opt_verb embedding; ATTACK IDENTITY via a learned attack_emb (opt_attack_id) -- so the old
# positional/flag/attackId-scalar dims were dropped (a uninterpretable normalized-id scalar).
OPT_STRUCT = 2 + 4 + 5 + 1
MAX_ATTACK = 2048    # attack-id embedding table size (>max attackId ~1556); index 0 = no-attack (padding)
OPT_DYN = N_OPT_TYPES + OPT_STRUCT

# normalisation scales
_T = 50.0; _CNT = 15.0; _DECK = 60.0; _PRIZE = 6.0; _DISCARD = 40.0

# ---- v2 (token) constants --------------------------------------------------
N_ENERGY_BINS = 12   # EnergyType bins 0..11 (spec: histogram over 12 types)
UNIT_ATTR = 24       # unit-attr layout (idx 23 = next-turn damage-reduction buff)
N_PREEVO = 2         # pre-evolution stack ids per unit (padded)
N_TOOLS = 2          # attached tool ids per unit (padded)
N_ENERGY_CARDS = 4   # attached energy CARD ids per unit (padded) -- special-energy identity
                     # (Double Turbo / Team Rocket / Jet / ...), distinct from the color histogram

DECK_SIZE = 60       # our/opp "decklist" token budget
N_PRIZE = 6          # prize slots per side
MAX_DISCARD = 60     # discard token budget (public)
N_STADIUM = 2        # slot0 = self-owned stadium, slot1 = opp-owned (position=owner)

G = 19               # CLS scalars: 13 board/turn + 5 select-dynamics + 1 our-this-turn offensive buff

# Canonical PRE-ENCODER token order -- MUST match policy TokenTransformer._encode's state-token
# build order. An option points at the EXACT sequence token of the card/unit it references
# (gathered by global position), so its src/tgt embedding IS that token, not a rebuilt copy.
_TOKEN_LAYOUT = [
    ("cls", 1), ("sel_type", 1), ("sel_ctx", 1),
    ("self_deck", DECK_SIZE), ("opp_deck", DECK_SIZE),
    ("self_prize", N_PRIZE), ("opp_prize", N_PRIZE),
    ("self_hand", MAX_HAND), ("opp_hand", MAX_HAND),
    ("self_discard", MAX_DISCARD), ("opp_discard", MAX_DISCARD),
    ("stadium", N_STADIUM), ("effect", 2),
    ("self_units", 1 + N_BENCH), ("opp_units", 1 + N_BENCH),
]
_OFF, _pos = {}, 0
for _name, _sz in _TOKEN_LAYOUT:
    _OFF[_name] = _pos
    _pos += _sz
N_STATE_TOKENS = _pos          # number of non-option tokens (option positions index into these)
