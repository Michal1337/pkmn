"""Effect-feature encoder fork (``--arch transformer2_fx``) -- EXPERIMENTAL, A/B'd
against the base ``transformer2``. The base rl/encoding.py is left UNTOUCHED.

Two additions over the base v2 encoding, both driven by the frozen rl/effect_data.py
coarse effect-category tables (generated offline from the SDK card/attack text):

  (1) per-ATTACK effect multi-hot (``attack_multihot`` by attackId) APPENDED to
      ``opt_attr`` -> width OPT_STRUCT + N_ATTACK_FX (the net's opt_attr_proj sizes to it
      via net_config['opt_struct']);
  (2) per-CARD ability + trainer effect multi-hot APPENDED to the static card-feature
      matrix (consumed by the net's static_proj, which auto-sizes). Requires --static.

Plus the #3 select tweak: a NUMBER option's count is rescaled by the select's OWN
ceiling (maxCount / remainDamageCounter) instead of a fixed /15, and YES/NO/NUMBER
options point their source at the effect token when an effect is driving the select.

NOTE: MAX_HAND stays at the base value here -- raising it forks the whole token layout
(_OFF offsets / positions), so it's intentionally deferred to keep this encoder a thin,
low-risk post-processor over the base encode(). It's a non-lossy change for a later pass.
"""

from __future__ import annotations

import numpy as np

from .encoding import TokenEncoder
from .enc_constants import OPT_STRUCT, MAX_OPTIONS, _OFF
from . import effect_data

OPT_STRUCT_FX = OPT_STRUCT + effect_data.N_ATTACK_FX        # widened opt_attr in the fork


class TokenEncoderFX(TokenEncoder):
    """Base TokenEncoder + per-attack effect multi-hot on opt_attr + the #3 select tweak.
    The ability/trainer static multi-hot lives on the NET side (see _AugmentedCardTable)."""

    def encode(self, obs, picked=None, **kw):
        out = super().encode(obs, picked, **kw)
        # (1) append the per-attack effect multi-hot (from opt_attack_id) to opt_attr.
        aid = out["opt_attack_id"]
        amh = np.asarray([effect_data.attack_multihot(int(a)) for a in aid], dtype=np.float32)
        out["opt_attr"] = np.concatenate([out["opt_attr"].astype(np.float32), amh], axis=1)
        # (3) NUMBER/YES-NO select tweak (robust; guards on missing fields).
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        eff_present = bool(sel.get("effect"))
        ceiling = max(sel.get("maxCount") or 1, sel.get("remainDamageCounter") or 0, 1)
        for i, o in enumerate(opts[:MAX_OPTIONS]):
            t = o.get("type")
            if t == 0:                                       # NUMBER: rescale by the select's own ceiling
                out["opt_attr"][i, 1] = min((o.get("number") or 0) / ceiling, 1.0)
            if t in (0, 1, 2) and eff_present and out["opt_src_pos"][i] < 0:
                out["opt_src_pos"][i] = _OFF["effect"]       # point YES/NO/NUMBER src at the effect card
        return out

    @property
    def shapes(self):
        s = dict(super().shapes)
        s["opt_attr"] = (MAX_OPTIONS, OPT_STRUCT_FX)
        return s


class _AugmentedCardTable:
    """Wraps a CardTable so the NET's static matrix carries the per-card ability/trainer
    effect multi-hot as appended columns; static_proj auto-sizes to the wider matrix.
    All other attributes (vocab_size, features, cf, ...) delegate to the base table."""

    def __init__(self, base):
        self._base = base
        extra = np.asarray([effect_data.ability_multihot(i) + effect_data.trainer_multihot(i)
                            for i in range(base.matrix.shape[0])], dtype=np.float32)
        self.matrix = np.concatenate([base.matrix.astype(np.float32), extra], axis=1)

    def __getattr__(self, name):                             # vocab_size/features/cf/... -> base
        return getattr(self._base, name)


def encoder_for(net_config, card_table):
    """Return the encoder for this net_config's arch (FX fork or base)."""
    if (net_config or {}).get("arch") == "transformer2_fx":
        return TokenEncoderFX(card_table)
    return TokenEncoder(card_table)


def build_net_for(net_config, card_table):
    """Build the net, handling the FX fork's wider opt_attr + augmented static matrix.
    For the base arch this is exactly build_token_net(card_table, net_config)."""
    from .policy import build_token_net
    if (net_config or {}).get("arch") == "transformer2_fx":
        cfg = {**net_config, "opt_struct": OPT_STRUCT_FX}
        return build_token_net(_AugmentedCardTable(card_table), cfg)
    return build_token_net(card_table, net_config)
