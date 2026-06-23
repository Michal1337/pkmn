"""Checkpoint-loading helper.

The v1 mlp / transformer ActorCritic that used to live here (plus obs_to_tensors,
jit_wrap, greedy_action, build_net) has been removed -- the live model is the v2
token-transformer in rl/policy2.py (build_token_net / obs_to_tensors2). Only the
arch-agnostic checkpoint loader remains, kept here so existing
``from .policy import load_compatible`` call sites are unchanged.
"""

from __future__ import annotations


def load_compatible(net, state_dict):
    """STRICT load: load_state_dict(strict=True), so ANY mismatch (missing/extra/shape)
    RAISES instead of silently dropping or zero-padding layers. Returns [] (the old
    lenient warm-start returned a list of skipped keys; kept for call-site compatibility)."""
    net.load_state_dict(state_dict)
    return []
