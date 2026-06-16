"""Per-option actor-critic network for the cabt action space.

The action space is dynamic, so the actor does NOT have a fixed positional head.
Instead it scores each candidate option from that option's own features (type +
referenced-card features/embedding), conditioned on a global state vector, then
applies the legal-action mask. A separate scalar head scores the SUBMIT action.

Card identity is dual: static features (from the CSV) ++ a learned embedding of
the raw id (the "embedding hook"). The embedding table is shared across every
card surface (board slots, hand, options, stadium).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .encoding import MAX_OPTIONS, N_SELECT_TYPES, N_SELECT_CTX

# These keys are int64 id tensors; everything else is float32.
# Must match Encoder.int_keys.
_INT_KEYS = {"self_active_id", "opp_active_id", "self_bench_id", "opp_bench_id",
             "hand_id", "stadium_id", "opt_card_id", "select_type", "select_context"}


def obs_to_tensors(obs: dict, device) -> dict:
    """Stack-of-arrays obs dict -> torch tensors on device (adds no batch dim)."""
    out = {}
    for k, v in obs.items():
        if k in _INT_KEYS:
            out[k] = torch.as_tensor(np.asarray(v), dtype=torch.long, device=device)
        else:
            out[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
    return out


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


def _masked_mean(x, mask):
    """x: [B,N,D], mask: [B,N] -> [B,D] mean over valid rows."""
    m = mask.unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1.0)
    return (x * m).sum(dim=1) / denom


class ActorCritic(nn.Module):
    def __init__(self, card_feat_dim: int, vocab_size: int,
                 emb_dim: int = 32, card_h: int = 64, trunk_h: int = 256,
                 opt_h: int = 96):
        super().__init__()
        self.cf = card_feat_dim
        self.card_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.sel_type_emb = nn.Embedding(N_SELECT_TYPES, 8)
        self.sel_ctx_emb = nn.Embedding(N_SELECT_CTX, 8)

        # shared card encoder: static features ++ id embedding -> card_h
        self.card_enc = _mlp([card_feat_dim + emb_dim, card_h, card_h])

        self.slot_enc = _mlp([card_h + 18, card_h, card_h])   # +dyn(18)
        self.hand_enc = _mlp([card_h, card_h])
        self.disc_enc = _mlp([card_feat_dim, 64])
        self.stad_enc = _mlp([card_h, 64])
        self.opt_enc = _mlp([card_h + 28, opt_h, opt_h])      # +opt_dyn(28)

        # global state vector -> trunk
        g_dim = (14 + 8 + 8           # scalars + select type/ctx emb
                 + 10 + 10            # players
                 + card_h * 4         # self/opp ACTIVE (own vec) + BENCH (pooled)
                 + card_h             # hand pooled
                 + 64 + 64            # self/opp discard
                 + 64)                # stadium
        self.trunk = _mlp([g_dim, trunk_h, trunk_h])

        self.opt_score = _mlp([trunk_h + opt_h, opt_h, 1])    # per-option logit
        self.submit_score = _mlp([trunk_h, 64, 1])            # submit logit
        self.value = _mlp([trunk_h, trunk_h, 1])

    # -- card representation shared everywhere -----------------------------
    def _cards(self, static, ids):
        """static [.,K,cf], ids [.,K] -> [.,K,card_h]."""
        e = self.card_emb(ids)
        return self.card_enc(torch.cat([static, e], dim=-1))

    def _encode(self, o: dict):
        B = o["scalars"].shape[0]

        # board: ACTIVE as its own vector (zeroed when absent), BENCH pooled.
        def board(side):
            a_c = self._cards(o[f"{side}_active_static"], o[f"{side}_active_id"])    # [B,1,ch]
            a_x = self.slot_enc(torch.cat([a_c, o[f"{side}_active_dyn"]], dim=-1))   # [B,1,ch]
            a = (a_x * o[f"{side}_active_dyn"][..., 0:1]).squeeze(1)                 # [B,ch] 0 if no active
            b_c = self._cards(o[f"{side}_bench_static"], o[f"{side}_bench_id"])      # [B,5,ch]
            b_x = self.slot_enc(torch.cat([b_c, o[f"{side}_bench_dyn"]], dim=-1))
            b = _masked_mean(b_x, o[f"{side}_bench_dyn"][..., 0])                    # [B,ch]
            return a, b
        self_a, self_b = board("self")
        opp_a, opp_b = board("opp")

        hand_c = self._cards(o["hand_static"], o["hand_id"])
        hand = _masked_mean(self.hand_enc(hand_c), o["hand_mask"])

        self_disc = self.disc_enc(o["self_discard_agg"])
        opp_disc = self.disc_enc(o["opp_discard_agg"])
        stad_c = self._cards(o["stadium_static"].unsqueeze(1), o["stadium_id"]).squeeze(1)
        stad = self.stad_enc(stad_c)

        st = self.sel_type_emb(o["select_type"].squeeze(-1))
        sc = self.sel_ctx_emb(o["select_context"].squeeze(-1))

        g = torch.cat([
            o["scalars"], st, sc, o["self_player"], o["opp_player"],
            self_a, self_b, opp_a, opp_b, hand, self_disc, opp_disc, stad,
        ], dim=-1)
        h = self.trunk(g)

        # option representations (kept per-slot for scoring)
        opt_c = self._cards(o["opt_card_static"], o["opt_card_id"])              # [B,64,card_h]
        opt = self.opt_enc(torch.cat([opt_c, o["opt_dyn"]], dim=-1))            # [B,64,opt_h]
        return h, opt

    def logits_value(self, o: dict):
        h, opt = self._encode(o)
        B = h.shape[0]
        h_exp = h.unsqueeze(1).expand(B, MAX_OPTIONS, h.shape[-1])
        opt_logits = self.opt_score(torch.cat([h_exp, opt], dim=-1)).squeeze(-1)  # [B,64]
        submit_logit = self.submit_score(h)                                       # [B,1]
        logits = torch.cat([opt_logits, submit_logit], dim=-1)                    # [B,65]
        logits = logits.masked_fill(o["action_mask"] < 0.5, -1e9)
        return logits, self.value(h).squeeze(-1)

    def get_value(self, o: dict):
        h, _ = self._encode(o)
        return self.value(h).squeeze(-1)

    def get_action_and_value(self, o: dict, action=None):
        logits, value = self.logits_value(o)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


@torch.no_grad()
def greedy_action(net: ActorCritic, obs_single: dict, device) -> int:
    """Pick the argmax legal action for a single (unbatched) numpy obs."""
    o = {k: v[None] for k, v in obs_to_tensors(obs_single, device).items()}
    logits, _ = net.logits_value(o)
    return int(logits.argmax(dim=-1).item())
