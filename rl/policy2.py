"""Token-based Transformer actor-critic for the cabt action space (v2).

Additive sibling of ``rl/policy.py``. Consumes the token streams emitted by
``rl/encoding2.TokenEncoder`` and exposes the SAME interface as the existing nets
(``logits_value`` / ``get_value`` / ``get_action_and_value``) so it is drop-in.

Design (matches encoding2's streams):
  * one shared card embedding ``nn.Embedding(vocab_size+1, emb_dim, padding_idx=0)``
    -- index 0 = EMPTY/pad, index ``vocab_size`` = UNK (hidden card);
  * a per-token-type embedding (CLS, the card-list streams, the unit stream, the
    option stream) added to every token;
  * each card-list token = proj(card_emb(id)) + type_emb;
  * each in-play unit token = card_emb(top) + sum card_emb(preevo) + sum
    card_emb(tool) + unit_attr_proj(attr[23]) + type_emb;
  * each option token = card_emb(src) + card_emb(tgt) + opt_attr_proj(attr) +
    type_emb;
  * the CLS token = cls_param + scalar_proj(cls_scalars) + select-type/ctx emb;
  * one ``nn.TransformerEncoder`` over the whole sequence with a
    ``src_key_padding_mask`` built from each stream's pad mask (options use the
    legal-action mask). Value reads the CLS output; the policy pointer-scores the
    option-token outputs and concatenates a CLS-derived SUBMIT logit, then masks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .card_features import CardTable
from .encoding2 import (
    MAX_OPTIONS, N_ACTIONS, OPT_DYN,
    N_SELECT_TYPES, N_SELECT_CTX,
    UNIT_ATTR, OPT_UNIT_DIM, G,
)

# int64 id tensors (everything else float32). Must match TokenEncoder.int_keys.
_INT_KEYS = {
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

# Token-type ids for the type/zone embedding. Each card-list STREAM gets its own
# type so the net can tell a hand card from a discard card etc.
(_T_CLS,
 _T_SELF_DECK, _T_OPP_DECK,
 _T_SELF_PRIZE, _T_OPP_PRIZE,
 _T_SELF_HAND, _T_OPP_HAND,
 _T_SELF_DISC, _T_OPP_DISC,
 _T_STADIUM,
 _T_SELF_ACTIVE, _T_SELF_BENCH,
 _T_OPP_ACTIVE, _T_OPP_BENCH,
 _T_OPT, _T_EFFECT) = range(16)
_N_TTYPES = 16

# (key, type-id) for the flat card-list streams (id + matching *_mask).
_CARD_STREAMS = [
    ("self_deck", _T_SELF_DECK), ("opp_deck", _T_OPP_DECK),
    ("self_prize", _T_SELF_PRIZE), ("opp_prize", _T_OPP_PRIZE),
    ("self_hand", _T_SELF_HAND), ("opp_hand", _T_OPP_HAND),
    ("self_discard", _T_SELF_DISC), ("opp_discard", _T_OPP_DISC),
    ("stadium", _T_STADIUM),
    ("effect", _T_EFFECT),       # source-effect tokens: the card driving this select + contextCard
]


def load_compatible(net, state_dict):
    """Warm-start: load every parameter whose shape matches; skip the rest."""
    own = net.state_dict()
    keep = {k: v for k, v in state_dict.items() if k in own and v.shape == own[k].shape}
    net.load_state_dict(keep, strict=False)
    return sorted(set(own) - set(keep))


def obs_to_tensors2(obs: dict, device) -> dict:
    """encoding2 stack-of-arrays obs -> torch tensors on device (no batch dim)."""
    out = {}
    for k, v in obs.items():
        if k in _INT_KEYS:
            out[k] = torch.as_tensor(np.asarray(v), dtype=torch.long, device=device)
        else:
            out[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
    return out


class TokenTransformer(nn.Module):
    """Token-set state -> Transformer encoder -> pointer policy + CLS value.

    Same interface as policy.ActorCritic / TransformerActorCritic.
    """

    def __init__(self, vocab_size: int, emb_dim: int = 48, d_model: int = 128,
                 nhead: int = 4, nlayers: int = 2, ff: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d = d_model
        self.UNK = vocab_size                                  # hidden-card id index
        # +1 so index `vocab_size` (UNK) is a real, learnable row; 0 stays pad.
        self.card_emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        self.type_emb = nn.Embedding(_N_TTYPES, d_model)
        self.sel_type_emb = nn.Embedding(N_SELECT_TYPES, d_model)
        self.sel_ctx_emb = nn.Embedding(N_SELECT_CTX, d_model)
        self.cls = nn.Parameter(torch.zeros(d_model))

        # per-stream projections -> d_model
        self.card_proj = nn.Linear(emb_dim, d_model)           # any card-list token
        self.unit_id_proj = nn.Linear(emb_dim, d_model)        # unit id-bag (top+preevo+tool)
        self.unit_attr_proj = nn.Linear(UNIT_ATTR, d_model)    # 23-dim unit attr
        self.opt_id_proj = nn.Linear(emb_dim, d_model)         # option src+tgt id-bag
        self.opt_attr_proj = nn.Linear(OPT_DYN, d_model)       # option structural feats
        self.opt_src_unit_proj = nn.Linear(OPT_UNIT_DIM, d_model)  # acting-Pokemon live state
        self.opt_tgt_unit_proj = nn.Linear(OPT_UNIT_DIM, d_model)  # acted-on-Pokemon live state
        self.scalar_proj = nn.Linear(G, d_model)               # CLS scalars

        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, nlayers)

        self.opt_head = nn.Linear(d_model, 1)
        self.submit_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)

    # -- token builders -----------------------------------------------------
    def _type(self, B, K, t, device):
        return self.type_emb(torch.full((B, K), t, dtype=torch.long, device=device))

    def _card_stream(self, ids, t, device):
        """ids [B,K] -> token [B,K,d] = proj(emb(id)) + type_emb."""
        B, K = ids.shape
        return self.card_proj(self.card_emb(ids)) + self._type(B, K, t, device)

    def _unit_stream(self, top_id, preevo_id, tool_id, attr, active_t, bench_t, device):
        """unit tokens [B,U,d] = proj(emb(top)+sum emb(preevo)+sum emb(tool))
        + unit_attr_proj(attr) + type_emb. Row 0 = ACTIVE (own type), rows 1.. = BENCH
        (so the net distinguishes the battling Pokemon from the bench; bench stays symmetric)."""
        B, U = top_id.shape
        idbag = (self.card_emb(top_id)
                 + self.card_emb(preevo_id).sum(dim=2)
                 + self.card_emb(tool_id).sum(dim=2))          # [B,U,emb]
        types = torch.full((B, U), bench_t, dtype=torch.long, device=device)
        types[:, 0] = active_t                                  # slot 0 is the Active Pokemon
        return self.unit_id_proj(idbag) + self.unit_attr_proj(attr) + self.type_emb(types)

    def _opt_stream(self, src_id, tgt_id, attr, src_unit, tgt_unit, device):
        B, K = src_id.shape
        idbag = self.card_emb(src_id) + self.card_emb(tgt_id)  # [B,K,emb]
        return (self.opt_id_proj(idbag) + self.opt_attr_proj(attr)
                + self.opt_src_unit_proj(src_unit) + self.opt_tgt_unit_proj(tgt_unit)
                + self._type(B, K, _T_OPT, device))

    # -- core ---------------------------------------------------------------
    def _encode(self, o: dict):
        dev = o["cls_scalars"].device
        B = o["cls_scalars"].shape[0]
        toks, pads = [], []

        # CLS token (global scalars + select context). Never padded.
        cls_tok = (self.cls.expand(B, 1, self.d)
                   + self.scalar_proj(o["cls_scalars"]).unsqueeze(1)
                   + self.sel_type_emb(o["select_type"].squeeze(-1)).unsqueeze(1)
                   + self.sel_ctx_emb(o["select_context"].squeeze(-1)).unsqueeze(1)
                   + self._type(B, 1, _T_CLS, dev))
        toks.append(cls_tok)
        pads.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))

        # flat card-list streams (pad mask from each stream's *_mask)
        for name, t in _CARD_STREAMS:
            toks.append(self._card_stream(o[f"{name}_id"], t, dev))
            pads.append(o[f"{name}_mask"] < 0.5)

        # in-play unit streams (active + bench), both sides
        for side, (at, bt) in (("self", (_T_SELF_ACTIVE, _T_SELF_BENCH)),
                               ("opp", (_T_OPP_ACTIVE, _T_OPP_BENCH))):
            toks.append(self._unit_stream(
                o[f"{side}_unit_top_id"], o[f"{side}_unit_preevo_id"],
                o[f"{side}_unit_tool_id"], o[f"{side}_unit_attr"], at, bt, dev))
            pads.append(o[f"{side}_unit_mask"] < 0.5)

        # option tokens (pad = illegal options, from the action mask)
        opt_tok = self._opt_stream(o["opt_src_id"], o["opt_tgt_id"], o["opt_attr"],
                                   o["opt_src_unit"], o["opt_tgt_unit"], dev)
        opt_present = o["action_mask"][..., :MAX_OPTIONS] > 0.5
        n_opt = opt_tok.shape[1]
        toks.append(opt_tok)
        pads.append(~opt_present)

        seq = torch.cat(toks, dim=1)
        pad = torch.cat(pads, dim=1)
        # A fully-padded row would NaN the softmax inside attention. The CLS token
        # is never padded, so no row is all-pad -> safe. (Belt-and-braces: encoder
        # rows still attend to CLS.)
        enc = self.encoder(seq, src_key_padding_mask=pad)
        cls_out = enc[:, 0]                    # CLS
        opt_out = enc[:, -n_opt:]              # the trailing option tokens
        return cls_out, opt_out

    def logits_value(self, o: dict):
        cls_out, opt_out = self._encode(o)
        opt_logits = self.opt_head(opt_out).squeeze(-1)        # [B, MAX_OPTIONS]
        submit_logit = self.submit_head(cls_out)               # [B, 1]
        logits = torch.cat([opt_logits, submit_logit], dim=-1)  # [B, N_ACTIONS]
        logits = logits.masked_fill(o["action_mask"] < 0.5, -1e9)
        return logits, self.value_head(cls_out).squeeze(-1)

    def get_value(self, o: dict):
        cls_out, _ = self._encode(o)
        return self.value_head(cls_out).squeeze(-1)

    def get_action_and_value(self, o: dict, action=None):
        logits, value = self.logits_value(o)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def build_token_net(card_table: CardTable, net_config: dict | None = None) -> TokenTransformer:
    """Construct a TokenTransformer sized to ``card_table`` (vocab_size).

    ``net_config`` keys are constructor kwargs (emb_dim, d_model, nhead, nlayers,
    ff, dropout); unknown keys (e.g. an 'arch' tag) are ignored.
    """
    cfg = dict(net_config or {})
    cfg.pop("arch", None)
    return TokenTransformer(card_table.vocab_size, **cfg)


if __name__ == "__main__":
    # quick smoke: build the net, encode + batch 3 real obs, forward.
    import os
    import pickle

    from .encoding2 import TokenEncoder

    enc = TokenEncoder()
    net = build_token_net(enc.cards, {})
    pool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "notes", "vcalib_pool.pkl")
    pool = pickle.load(open(pool_path, "rb"))
    outs = [enc.encode(pool[i]["root"]) for i in range(3)]
    batch = {k: torch.stack([obs_to_tensors2(e, "cpu")[k] for e in outs]) for k in outs[0]}
    logits, value = net.logits_value(batch)
    a, lp, ent, v = net.get_action_and_value(batch)
    gv = net.get_value(batch)
    print("logits", tuple(logits.shape), "value", tuple(value.shape),
          "action", tuple(a.shape), "get_value", tuple(gv.shape))
    # every sampled action must be legal
    legal = batch["action_mask"].gather(1, a[:, None]).squeeze(1)
    assert (legal > 0.5).all(), "sampled an illegal action"
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params={n_params:,}  N_ACTIONS={N_ACTIONS}  OK")
