"""Validate the token-based net (rl/policy2.TokenTransformer) on a SYNTHETIC batch.

Builds a small TokenTransformer and forwards a batch (B=2) of tensors whose shapes
EXACTLY match TokenEncoder.shapes -- no real obs / pickle needed, so this is a pure
net<->encoder-contract check. Asserts the full interface:
  * logits_value(o) -> logits[B, N_ACTIONS], value[B]
  * get_value(o)    -> value[B]   (equal to logits_value's value)
  * get_action_and_value(o) -> action[B], logp[B], entropy[B], value[B]
  * illegal actions are masked to -inf and legal ones are finite;
  * every sampled action is legal.

Run:
    PYTHONPATH=. /c/Users/mgrom/miniconda3/python scripts/validate_policy2.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from rl.encoding import MAX_OPTIONS, N_ACTIONS, N_SELECT_TYPES, N_SELECT_CTX  # noqa: E402
from rl.encoding2 import TokenEncoder  # noqa: E402
from rl.policy2 import build_token_net  # noqa: E402

B = 2
DEV = "cpu"


def synth_batch(enc: TokenEncoder, rng) -> dict:
    """A batch of B tensors per encoder key, shapes == enc.shapes (with a leading
    batch dim). int64 ids span [0, UNK]; select_type/context use their vocabs."""
    shapes, int_keys, UNK = enc.shapes, enc.int_keys, enc.UNK
    o = {}
    for k, shp in shapes.items():
        full = (B,) + shp
        if k in int_keys:
            if k == "select_type":
                hi = N_SELECT_TYPES
            elif k == "select_context":
                hi = N_SELECT_CTX
            else:
                hi = UNK + 1  # ids are in [0, vocab_size] inclusive (UNK == vocab_size)
            o[k] = torch.from_numpy(rng.integers(0, hi, size=full).astype(np.int64))
        else:
            o[k] = torch.from_numpy(rng.standard_normal(full).astype(np.float32))

    # A realistic legal-action mask: first few options legal + submit; rest illegal.
    am = np.zeros((B, N_ACTIONS), np.float32)
    am[:, :5] = 1.0          # 5 legal options
    am[:, MAX_OPTIONS] = 1.0  # submit legal
    o["action_mask"] = torch.from_numpy(am)
    return o


def main() -> int:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    enc = TokenEncoder()
    net = build_token_net(enc.cards, {"emb_dim": 48, "d_model": 128,
                                      "nhead": 4, "nlayers": 2, "ff": 256})
    net.eval()
    n_params = sum(p.numel() for p in net.parameters())
    print(f"vocab_size={enc.vocab_size} UNK={enc.UNK} card_emb_rows={enc.vocab_size + 1} "
          f"N_ACTIONS={N_ACTIONS} params={n_params:,}")

    o = synth_batch(enc, rng)
    am = o["action_mask"]

    with torch.no_grad():
        logits, value = net.logits_value(o)
        v_only = net.get_value(o)
        action, logp, ent, v2 = net.get_action_and_value(o)

    # ---- shape asserts (the task's headline requirement) ----
    assert logits.shape == (B, N_ACTIONS), f"logits {tuple(logits.shape)} != {(B, N_ACTIONS)}"
    assert value.shape == (B,), f"value {tuple(value.shape)} != {(B,)}"
    assert v_only.shape == (B,), f"get_value {tuple(v_only.shape)} != {(B,)}"
    for nm, t in [("action", action), ("logp", logp), ("entropy", ent), ("value2", v2)]:
        assert t.shape == (B,), f"{nm} {tuple(t.shape)} != {(B,)}"

    # ---- masking + legality ----
    assert torch.isfinite(logits[am > 0.5]).all(), "legal-action logits not finite"
    assert (logits[am < 0.5] <= -1e8).all(), "illegal actions not masked to -inf"
    for b in range(B):
        assert am[b, int(action[b])] > 0.5, f"row {b} sampled illegal action {int(action[b])}"

    # ---- value-only path must equal the joint path (same CLS encode) ----
    assert torch.allclose(value, v_only, atol=1e-5), "get_value != logits_value value"
    assert torch.allclose(value, v2, atol=1e-5), "get_action_and_value value != logits_value value"

    print(f"logits.shape={tuple(logits.shape)}  value.shape={tuple(value.shape)}")
    print(f"get_value.shape={tuple(v_only.shape)}  action={action.tolist()}  "
          f"logp.shape={tuple(logp.shape)}  entropy.shape={tuple(ent.shape)}")
    print("VALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
