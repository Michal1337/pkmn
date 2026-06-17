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
             "hand_id", "self_discard_id", "opp_discard_id",
             "stadium_id", "opt_card_id", "select_type", "select_context"}


def load_compatible(net, state_dict):
    """Warm-start: load every parameter whose shape matches; skip the rest (e.g. a
    layer whose input dim changed across an encoding revision). Returns skipped keys."""
    own = net.state_dict()
    keep = {k: v for k, v in state_dict.items() if k in own and v.shape == own[k].shape}
    net.load_state_dict(keep, strict=False)
    return sorted(set(own) - set(keep))


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
        self.disc_enc = _mlp([card_h, 64])                    # per-card discard, pooled
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

        sd_c = self._cards(o["self_discard_static"], o["self_discard_id"])
        self_disc = _masked_mean(self.disc_enc(sd_c), o["self_discard_mask"])
        od_c = self._cards(o["opp_discard_static"], o["opp_discard_id"])
        opp_disc = _masked_mean(self.disc_enc(od_c), o["opp_discard_mask"])
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


def jit_wrap(net, enc):
    """Freeze a TorchScript adapter for fast CPU inference (~1.7x: the forward is
    op-dispatch-bound, ~40 tiny Linears). Same logits_value/get_value interface,
    byte-identical outputs. Use for opponents / search / eval, not training."""
    net.eval()
    keys = list(enc.shapes)
    ex = tuple(torch.zeros((1,) + enc.shapes[k],
                           dtype=(torch.long if k in enc.int_keys else torch.float32))
               for k in keys)

    class _Head(nn.Module):
        def __init__(self, fn):
            super().__init__(); self.net = net; self.fn = fn
        def forward(self, *a):
            return getattr(self.net, self.fn)({k: v for k, v in zip(keys, a)})

    with torch.no_grad():
        lv = torch.jit.freeze(torch.jit.trace(_Head("logits_value").eval(), ex, check_trace=False))
        gv = torch.jit.freeze(torch.jit.trace(_Head("get_value").eval(), ex, check_trace=False))

    class _Jit:
        def logits_value(self, o):
            return lv(*[o[k] for k in keys])
        def get_value(self, o):
            return gv(*[o[k] for k in keys])
    return _Jit()


@torch.no_grad()
def greedy_action(net, obs_single: dict, device) -> int:
    """Pick the argmax legal action for a single (unbatched) numpy obs."""
    o = {k: v[None] for k, v in obs_to_tensors(obs_single, device).items()}
    logits, _ = net.logits_value(o)
    return int(logits.argmax(dim=-1).item())


# ---- Transformer policy: state as a token set, attention + pointer scoring ----
# Token type ids (for the type/zone embedding).
_T_GLOBAL, _T_SACT, _T_SBEN, _T_OACT, _T_OBEN, _T_HAND, _T_STAD, _T_DISC, _T_OPT = range(9)
_N_TTYPES = 9


class TransformerActorCritic(nn.Module):
    """Encodes the board+hand+options as tokens, runs a Transformer encoder, then
    scores each OPTION token (pointer-style) and reads value/submit off a CLS token.
    Same interface as ActorCritic (logits_value/get_value/get_action_and_value)."""

    def __init__(self, card_feat_dim: int, vocab_size: int, emb_dim: int = 32,
                 d_model: int = 128, nhead: int = 4, nlayers: int = 3, ff: int = 256):
        super().__init__()
        self.cf = card_feat_dim
        self.d = d_model
        self.card_emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.type_emb = nn.Embedding(_N_TTYPES, d_model)
        self.cls = nn.Parameter(torch.zeros(d_model))
        # projections -> d_model
        self.card_proj = nn.Linear(card_feat_dim + emb_dim, d_model)  # any card token
        self.poke_dyn = nn.Linear(18, d_model)                        # board dynamic feats
        self.opt_dyn = nn.Linear(28, d_model)                         # option dynamic feats
        self.disc_proj = nn.Linear(card_feat_dim, d_model)            # discard aggregate
        self.scalar_proj = nn.Linear(14, d_model)
        self.sel_type_emb = nn.Embedding(N_SELECT_TYPES, d_model)
        self.sel_ctx_emb = nn.Embedding(N_SELECT_CTX, d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True, dropout=0.0)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.opt_head = nn.Linear(d_model, 1)
        self.submit_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)

    def _card_tok(self, static, ids):
        return self.card_proj(torch.cat([static, self.card_emb(ids)], dim=-1))

    def _type(self, B, K, t, device):
        return self.type_emb(torch.full((B, K), t, dtype=torch.long, device=device))

    def _encode(self, o: dict):
        dev = o["scalars"].device
        B = o["scalars"].shape[0]
        toks, pads = [], []

        # global / CLS token (carries scalars + select context)
        g = (self.cls.expand(B, 1, self.d)
             + self.scalar_proj(o["scalars"]).unsqueeze(1)
             + self.sel_type_emb(o["select_type"].squeeze(-1)).unsqueeze(1)
             + self.sel_ctx_emb(o["select_context"].squeeze(-1)).unsqueeze(1)
             + self._type(B, 1, _T_GLOBAL, dev))
        toks.append(g); pads.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))

        def add_pokes(side, ttype):
            for grp, n, tt in [("active", 1, ttype[0]), ("bench", o[f"{side}_bench_dyn"].shape[1], ttype[1])]:
                c = self._card_tok(o[f"{side}_{grp}_static"], o[f"{side}_{grp}_id"])
                t = c + self.poke_dyn(o[f"{side}_{grp}_dyn"]) + self._type(B, c.shape[1], tt, dev)
                present = o[f"{side}_{grp}_dyn"][..., 0] > 0.5      # present flag
                toks.append(t); pads.append(~present)
        add_pokes("self", (_T_SACT, _T_SBEN))
        add_pokes("opp", (_T_OACT, _T_OBEN))

        # hand
        h = self._card_tok(o["hand_static"], o["hand_id"]) + self._type(B, o["hand_id"].shape[1], _T_HAND, dev)
        toks.append(h); pads.append(o["hand_mask"] < 0.5)

        # stadium (single)
        st = self._card_tok(o["stadium_static"].unsqueeze(1), o["stadium_id"]) + self._type(B, 1, _T_STAD, dev)
        toks.append(st); pads.append((o["stadium_id"] == 0))

        # discard: per-card tokens (self/opp), identity-aware
        for side in ("self", "opp"):
            d = self._card_tok(o[f"{side}_discard_static"], o[f"{side}_discard_id"]) \
                + self._type(B, o[f"{side}_discard_id"].shape[1], _T_DISC, dev)
            toks.append(d); pads.append(o[f"{side}_discard_mask"] < 0.5)

        # options (the action candidates)
        oc = self._card_tok(o["opt_card_static"], o["opt_card_id"])
        ot = oc + self.opt_dyn(o["opt_dyn"]) + self._type(B, oc.shape[1], _T_OPT, dev)
        opt_present = o["opt_dyn"][..., :16].sum(-1) > 0            # a real option has a type one-hot
        n_opt = ot.shape[1]
        toks.append(ot); pads.append(~opt_present)

        seq = torch.cat(toks, dim=1)
        pad = torch.cat(pads, dim=1)
        enc = self.encoder(seq, src_key_padding_mask=pad)
        cls = enc[:, 0]                       # global token
        opt_enc = enc[:, -n_opt:]             # last n_opt tokens are the options
        return cls, opt_enc

    def logits_value(self, o: dict):
        cls, opt_enc = self._encode(o)
        opt_logits = self.opt_head(opt_enc).squeeze(-1)            # [B, MAX_OPTIONS]
        submit_logit = self.submit_head(cls)                      # [B,1]
        logits = torch.cat([opt_logits, submit_logit], dim=-1)
        logits = logits.masked_fill(o["action_mask"] < 0.5, -1e9)
        return logits, self.value_head(cls).squeeze(-1)

    def get_value(self, o: dict):
        cls, _ = self._encode(o)
        return self.value_head(cls).squeeze(-1)

    def get_action_and_value(self, o: dict, action=None):
        logits, value = self.logits_value(o)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


def build_net(card_feat_dim: int, vocab_size: int, net_config: dict):
    """Dispatch on net_config['arch'] ('mlp' default | 'transformer'). The
    remaining keys are constructor kwargs for the chosen class."""
    cfg = dict(net_config or {})
    arch = cfg.pop("arch", "mlp")
    if arch == "transformer":
        return TransformerActorCritic(card_feat_dim, vocab_size, **cfg)
    return ActorCritic(card_feat_dim, vocab_size, **cfg)
