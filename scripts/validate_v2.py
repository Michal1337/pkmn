"""End-to-end validation for the token-based v2 encoder + transformer.

Exercises the full v2 path against REAL cabt observations:

  1. get_card_table("EN_Card_Data.csv") -> TokenEncoder; load notes/vcalib_pool.pkl;
     encode the first ~6 records' root obs with picked=set(); obs_to_tensors +
     stack into a batch of 6.
  2. build_token_net(small dims); forward logits_value -> assert
     logits.shape == (6, N_ACTIONS), value.shape == (6,), no NaNs, and that
     illegal actions (action_mask < 0.5) are -1e9.
  3. get_value matches the value head from logits_value.
  4. one BACKWARD step: loss = masked cross-entropy to a random LEGAL action +
     value.pow(2).mean(); loss.backward(); assert card_emb and value_head grads
     are non-None and finite; an optimizer step runs.

Run (from repo root):
    PYTHONPATH=. /c/Users/mgrom/miniconda3/python scripts/validate_v2.py

Touches only rl/encoding.py + rl/policy.py (never encoding.py / policy.py).
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

# repo root on path (so `import rl...` works regardless of cwd)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rl.card_features import get_card_table
from rl.encoding import TokenEncoder, MAX_OPTIONS, N_ACTIONS
from rl.policy import build_token_net, obs_to_tensors

NEG = -1e9
B = 6


def main() -> int:
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 1. encoder + real obs -> batch ----
    csv_path = os.path.join(_ROOT, "EN_Card_Data.csv")
    card_table = get_card_table(csv_path)
    enc = TokenEncoder(card_table=card_table)
    print(f"[1] card_table: vocab_size={card_table.vocab_size} feat_dim={card_table.feat_dim}")
    print(f"    TokenEncoder: UNK id={enc.UNK}  #streams={len(enc.shapes)}")

    pool_path = os.path.join(_ROOT, "notes", "vcalib_pool.pkl")
    pool = pickle.load(open(pool_path, "rb"))
    n = min(B, len(pool))
    assert n == B, f"need >= {B} pool records, have {len(pool)}"
    print(f"    loaded {len(pool)} pool records; encoding first {n}")

    encoded = [enc.encode(pool[i]["root"], picked=set()) for i in range(n)]

    # validate per-record shapes/dtypes/id-range against the declared layout
    shapes, int_keys = enc.shapes, enc.int_keys
    for ri, out in enumerate(encoded):
        assert set(out) == set(shapes), (
            f"record {ri} key mismatch: missing={set(shapes)-set(out)} extra={set(out)-set(shapes)}")
        for k, arr in out.items():
            assert arr.shape == shapes[k], f"record {ri} [{k}] shape {arr.shape} != {shapes[k]}"
            want = np.int64 if k in int_keys else np.float32
            assert arr.dtype == want, f"record {ri} [{k}] dtype {arr.dtype} != {want}"
            if k in int_keys:
                assert int(arr.min()) >= 0 and int(arr.max()) <= enc.UNK, \
                    f"record {ri} [{k}] id out of [0, UNK={enc.UNK}]"
    print(f"    per-record encode OK ({len(shapes)} streams each)")

    dev = torch.device("cpu")
    tens = [obs_to_tensors(e, dev) for e in encoded]
    batch = {k: torch.stack([t[k] for t in tens], dim=0) for k in tens[0]}
    print(f"    batched: action_mask {tuple(batch['action_mask'].shape)}, "
          f"opt_attr {tuple(batch['opt_attr'].shape)}")

    # ---- 2. build net + forward logits_value ----
    net = build_token_net(card_table, {"emb_dim": 48, "d_model": 128,
                                        "nhead": 4, "nlayers": 2, "ff": 256})
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[2] TokenTransformer params={n_params:,}  N_ACTIONS={N_ACTIONS}")

    logits, value = net.logits_value(batch)
    assert logits.shape == (B, N_ACTIONS), f"logits {tuple(logits.shape)} != {(B, N_ACTIONS)}"
    assert value.shape == (B,), f"value {tuple(value.shape)} != {(B,)}"
    assert torch.isfinite(value).all(), "value has NaN/Inf"
    mask = batch["action_mask"]
    legal = mask >= 0.5
    # finite on legal entries; exactly NEG on illegal entries
    assert torch.isfinite(logits[legal]).all(), "legal logits have NaN/Inf"
    illegal = ~legal
    if illegal.any():
        assert torch.equal(logits[illegal], torch.full_like(logits[illegal], NEG)), \
            "illegal actions are not exactly -1e9"
    print(f"    logits {tuple(logits.shape)} value {tuple(value.shape)} finite; "
          f"illegal -> {NEG:g}; legal/row(min..max)="
          f"{int(legal.sum(1).min())}..{int(legal.sum(1).max())}")

    # ---- 3. get_value matches logits_value's value ----
    gv = net.get_value(batch)
    assert gv.shape == (B,)
    assert torch.allclose(gv, value, atol=1e-5), \
        f"get_value != logits_value value (max diff {float((gv-value).abs().max()):.2e})"
    print(f"    get_value matches logits_value value (max diff "
          f"{float((gv - value).abs().max()):.2e})")

    # ---- 4. one backward + optimizer step ----
    # target = a random LEGAL action per row
    targets = torch.empty(B, dtype=torch.long)
    for b in range(B):
        legal_idx = torch.nonzero(legal[b], as_tuple=False).squeeze(-1)
        targets[b] = legal_idx[torch.randint(len(legal_idx), (1,)).item()]

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    opt.zero_grad()
    logits, value = net.logits_value(batch)
    policy_loss = F.cross_entropy(logits, targets)   # masked logits -> illegal get ~0 prob
    value_loss = value.pow(2).mean()
    loss = policy_loss + value_loss
    assert torch.isfinite(loss), "loss is NaN/Inf"
    loss.backward()

    ce, ve = net.card_emb.weight.grad, net.value_head.weight.grad
    assert ce is not None and torch.isfinite(ce).all(), "card_emb grad missing/non-finite"
    assert ve is not None and torch.isfinite(ve).all(), "value_head grad missing/non-finite"
    grad_norm = torch.sqrt(sum((p.grad.detach() ** 2).sum()
                               for p in net.parameters() if p.grad is not None)).item()
    n_with_grad = sum(1 for p in net.parameters() if p.grad is not None)
    opt.step()
    print(f"[4] backward OK: loss={float(loss):.4f} (policy={float(policy_loss):.4f} "
          f"value={float(value_loss):.4f}); grad_norm={grad_norm:.4f}; "
          f"params_with_grad={n_with_grad}; card_emb+value_head grads finite; opt.step() ran")

    print("\nVALIDATE_V2 PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
