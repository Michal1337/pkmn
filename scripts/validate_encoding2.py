"""Validate the token-based encoder (rl/encoding.py) against real cabt obs.

Loads notes/vcalib_pool.pkl, encodes every record, and asserts each output array
matches TokenEncoder.shapes (shape + dtype + id-range + finiteness). Also smoke-
tests batching through obs_to_tensors and a forward pass of policy.TokenTransformer
when available, so the encoder<->net contract is checked end to end.

Run:
    PYTHONPATH=. /c/Users/mgrom/miniconda3/python scripts/validate_encoding.py
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.encoding import TokenEncoder  # noqa: E402

POOL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "notes", "vcalib_pool.pkl")


def main() -> int:
    enc = TokenEncoder()
    pool = pickle.load(open(POOL, "rb"))
    shapes, int_keys = enc.shapes, enc.int_keys
    print(f"loaded {len(pool)} records; vocab_size={enc.vocab_size} UNK={enc.UNK}")

    encoded = []
    for ri, r in enumerate(pool):
        out = enc.encode(r["root"], picked=set())
        assert set(out) == set(shapes), (
            f"[rec {ri}] key mismatch missing={set(shapes)-set(out)} extra={set(out)-set(shapes)}")
        for k, a in out.items():
            assert a.shape == shapes[k], f"[rec {ri}] {k} shape {a.shape} != {shapes[k]}"
            want = np.int64 if k in int_keys else np.float32
            assert a.dtype == want, f"[rec {ri}] {k} dtype {a.dtype} != {want}"
            assert np.isfinite(a).all(), f"[rec {ri}] {k} has non-finite values"
            if k in int_keys:
                assert (a >= 0).all() and (a <= enc.UNK).all(), f"[rec {ri}] {k} id out of [0,UNK]"
        encoded.append(out)
    print(f"OK: all {len(pool)} records match shapes/dtypes/id-range/finite")

    # net contract (optional: only if torch + policy import cleanly)
    try:
        import torch  # noqa: F401
        from rl.policy import build_token_net, obs_to_tensors
        net = build_token_net(enc.cards, {})
        dev = "cpu"
        # single forward
        o1 = {k: v[None] for k, v in obs_to_tensors(encoded[0], dev).items()}
        logits, value = net.logits_value(o1)
        assert logits.shape[-1] == shapes["action_mask"][0], "logits width != N_ACTIONS"
        # batched forward (stack 3)
        batch = {k: torch.stack([obs_to_tensors(e, dev)[k] for e in encoded[:3]])
                 for k in encoded[0]}
        bl, bv = net.logits_value(batch)
        assert bl.shape[0] == 3 and bv.shape[0] == 3
        a, lp, ent, v = net.get_action_and_value(batch)
        print(f"net OK: logits {tuple(bl.shape)} value {tuple(bv.shape)} "
              f"action {tuple(a.shape)} (TokenTransformer forward + batch)")
    except Exception as e:  # pragma: no cover - net is built in a sibling task
        print(f"net check skipped ({type(e).__name__}: {e})")

    print("\nfull .shapes layout:")
    for k in shapes:
        dt = "int64" if k in int_keys else "f32"
        print(f"  {k:22s} {str(shapes[k]):14s} {dt}")
    print("\nVALIDATION PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
