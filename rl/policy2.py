"""Token-based Transformer actor-critic for the cabt action space (v2).

Additive sibling of ``rl/policy.py``. Consumes the token streams emitted by
``rl/encoding.TokenEncoder`` and exposes the SAME interface as the existing nets
(``logits_value`` / ``get_value`` / ``get_action_and_value``) so it is drop-in.

Design (matches encoding's streams):
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

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# b=1 padded-token truncation (exact; see _encode). Escape hatch for A/B benchmarking only.
_TRUNC_B1 = os.environ.get("V2_NO_TRUNC") != "1"

from .card_features import CardTable
from .encoding import (
    MAX_OPTIONS, N_ACTIONS, OPT_STRUCT, N_OPT_TYPES, MAX_ATTACK,
    N_SELECT_TYPES, N_SELECT_CTX,
    UNIT_ATTR, G,
)

# int64 id tensors (everything else float32). Must match TokenEncoder.int_keys.
_INT_KEYS = {
    "select_type", "select_context", "effect_id",
    "self_deck_id", "opp_deck_id",
    "self_prize_id", "opp_prize_id",
    "self_hand_id", "opp_hand_id",
    "self_discard_id", "opp_discard_id",
    "stadium_id",
    "self_unit_top_id", "self_unit_preevo_id", "self_unit_tool_id", "self_unit_energy_id",
    "opp_unit_top_id", "opp_unit_preevo_id", "opp_unit_tool_id", "opp_unit_energy_id",
    "opt_src_pos", "opt_tgt_pos",
    "opt_src_card", "opt_tgt_card",
    "opt_verb", "opt_attack_id",
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
 _T_OPT, _T_EFFECT,
 _T_SEL_TYPE, _T_SEL_CTX) = range(18)   # select-type / select-context now their own tokens (off CLS)
_T_CARD_SYNTH = 18   # synthesized card token (attached energy/tool, deck-search pick) -- APPENDED so
                     # ids 0..17 are unchanged; gives synth tokens the zone marker real cards have
_N_TTYPES = 19

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
    """encoding stack-of-arrays obs -> torch tensors on device (no batch dim)."""
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
                 nhead: int = 4, nlayers: int = 2, dropout: float = 0.0,
                 card_feat=None, structured: bool = False):
        super().__init__()
        self.d = d_model
        self.UNK = vocab_size                                  # hidden-card id index
        # +1 so index `vocab_size` (UNK) is a real, learnable row; 0 stays pad. (BC: emb_dim 48 > 128,
        # but kept tunable.)
        self.card_emb = nn.Embedding(vocab_size + 1, emb_dim, padding_idx=0)
        # frozen STATIC per-card features (printed/constant: type/weakness/stage/base-maxHp/cost/...),
        # gathered by id. Complements the LEARNED card_emb AND the engine's DYNAMIC per-instance attrs
        # (effective in-play maxHp/energy/conditions live in unit_attr; here it is the BASE printed maxHp).
        # UNK row (index vocab_size) = zeros. card_feat=None -> disabled (old emb-only behavior).
        if card_feat is not None:
            _f = np.zeros((vocab_size + 1, card_feat.shape[1]), dtype=np.float32)
            _f[:card_feat.shape[0]] = card_feat
            self.register_buffer("card_feat", torch.from_numpy(_f), persistent=False)
            self.static_proj = nn.Linear(card_feat.shape[1], d_model)
        else:
            self.card_feat = None
            self.static_proj = None
        # STRUCTURED action head: verb-conditioned option scoring. The option's OptionType
        # (verb: PLAY/ATTACH/EVOLVE/ATTACK/ABILITY/...) is the first 16 dims of opt_attr; give
        # each verb its own scoring query+bias so ATTACH options share a scoring direction
        # (the action-semantics analogue of static card features). Shared opt_head kept as a
        # fallback for rare verbs. None -> plain pointer (old behavior).
        self.structured = structured
        if structured:
            self.type_query = nn.Embedding(N_OPT_TYPES, d_model)  # OptionType (0..16) -> scoring query
            self.type_bias = nn.Embedding(N_OPT_TYPES, 1)
        self.type_emb = nn.Embedding(_N_TTYPES, d_model)
        self.sel_type_emb = nn.Embedding(N_SELECT_TYPES, d_model)
        self.sel_ctx_emb = nn.Embedding(N_SELECT_CTX, d_model)
        self.cls = nn.Parameter(torch.zeros(d_model))

        # per-stream id projections emb_dim -> d_model (a shared bottleneck across all cards).
        # Identity when emb_dim == d_model; otherwise Linear (the BC-preferred path).
        _id_proj = (lambda: nn.Identity()) if emb_dim == d_model else (lambda: nn.Linear(emb_dim, d_model))
        self.card_proj = _id_proj()                            # any card-list token
        self.unit_id_proj = _id_proj()                         # unit id-bag (top+preevo+tool)
        self.unit_attr_proj = nn.Linear(UNIT_ATTR, d_model)    # unit attr
        # An option's source/target each = the EXACT pre-encoder token of the card/unit it points at
        # (gathered by position in _encode), through its own d->d projection. Distinct maps so src vs
        # tgt stay separable. That gathered token already carries the card's identity/features and,
        # for board refs, the full unit state; opt_attr carries the per-option structural distinctions.
        self.opt_src_proj = nn.Linear(d_model, d_model)
        self.opt_tgt_proj = nn.Linear(d_model, d_model)
        self.opt_attr_proj = nn.Linear(OPT_STRUCT, d_model)    # option structural feats (no verb one-hot)
        self.opt_verb_emb = nn.Embedding(N_OPT_TYPES, d_model)  # OptionType (verb) -> its own learned vector
        self.attack_emb = nn.Embedding(MAX_ATTACK, d_model, padding_idx=0)  # attackId -> learned identity (0=no-attack, zero)
        # (the acting/acted-on Pokemon's state now enters each option as that unit's FULL token,
        #  gathered by slot in _encode -- no dedicated projection needed.)
        self.scalar_proj = nn.Linear(G, d_model)               # CLS scalars

        ff = 4 * d_model                                       # FFN width = 4x d_model (conventional)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True, dropout=dropout)
        self.encoder = nn.TransformerEncoder(layer, nlayers)

        self.opt_head = nn.Linear(d_model, 1)
        self.submit_head = nn.Linear(d_model, 1)
        # value reads CLS ++ mean-pool over all non-pad tokens (not CLS alone) -- a single token
        # under-summarizes the board, and value is the demonstrated LB ceiling.
        self.value_head = nn.Linear(2 * d_model, 1)

    # -- token builders -----------------------------------------------------
    def _type(self, B, K, t, device):
        return self.type_emb(torch.full((B, K), t, dtype=torch.long, device=device))

    def _static(self, ids):
        """static_proj(card_feat[ids]) per card token; 0 when static features are disabled."""
        if self.static_proj is None:
            return 0
        return self.static_proj(self.card_feat[ids])

    def _card_stream(self, ids, t, device):
        """ids [B,K] -> token [B,K,d] = proj(emb(id)) + type_emb + static."""
        B, K = ids.shape
        return self.card_proj(self.card_emb(ids)) + self._type(B, K, t, device) + self._static(ids)

    def _unit_stream(self, top_id, preevo_id, tool_id, energy_id, attr, active_t, bench_t, device):
        """unit tokens [B,U,d] = proj(emb(top)+sum emb(preevo)+sum emb(tool)+sum emb(energy))
        + unit_attr_proj(attr) + type_emb. Row 0 = ACTIVE (own type), rows 1.. = BENCH
        (so the net distinguishes the battling Pokemon from the bench; bench stays symmetric).
        The energy id-bag carries SPECIAL-energy identity (Double Turbo/Team Rocket/...),
        complementing the color histogram + count already in attr."""
        B, U = top_id.shape
        idbag = (self.card_emb(top_id)
                 + self.card_emb(preevo_id).sum(dim=2)
                 + self.card_emb(tool_id).sum(dim=2)
                 + self.card_emb(energy_id).sum(dim=2))         # [B,U,emb]
        types = torch.full((B, U), bench_t, dtype=torch.long, device=device)
        types[:, 0] = active_t                                  # slot 0 is the Active Pokemon
        return (self.unit_id_proj(idbag) + self.unit_attr_proj(attr)
                + self.type_emb(types) + self._static(top_id))

    def _opt_stream(self, src_tok, tgt_tok, attr, verb, attack_id, device):
        """Option tokens [B,K,d]. src_tok/tgt_tok ARE the gathered pre-encoder tokens of the
        card/unit each option references as source/target (its view IS that exact token), each
        through its own projection; summed with the per-option structural attr, a learned
        per-OptionType (verb) embedding, a learned per-ATTACK embedding (attack_emb; 0=no-attack ->
        zero via padding_idx, so it distinguishes multiple attacks of the same Pokemon), and the
        option-type-token marker. Card identity/features/unit-state all live in the gathered tokens."""
        B, K = attr.shape[0], attr.shape[1]
        return (self.opt_src_proj(src_tok) + self.opt_tgt_proj(tgt_tok)
                + self.opt_attr_proj(attr) + self.opt_verb_emb(verb)
                + self.attack_emb(attack_id)
                + self._type(B, K, _T_OPT, device))

    # -- core ---------------------------------------------------------------
    def _encode(self, o: dict):
        dev = o["cls_scalars"].device
        B = o["cls_scalars"].shape[0]
        toks, pads = [], []

        # CLS token = global scalars only (the decision's framing now lives in its own tokens
        # below, so CLS isn't overloaded). Never padded; its output feeds value + submit heads.
        cls_tok = (self.cls.expand(B, 1, self.d)
                   + self.scalar_proj(o["cls_scalars"]).unsqueeze(1)
                   + self._type(B, 1, _T_CLS, dev))
        toks.append(cls_tok)
        pads.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))
        # select-TYPE (kind of decision) and select-CONTEXT (its purpose) as SEPARATE tokens:
        # each = its categorical embedding + a token-type marker. The transformer broadcasts this
        # framing to every option via attention. Never padded.
        _nopad = lambda: pads.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))
        toks.append(self.sel_type_emb(o["select_type"].squeeze(-1)).unsqueeze(1)
                    + self._type(B, 1, _T_SEL_TYPE, dev)); _nopad()
        toks.append(self.sel_ctx_emb(o["select_context"].squeeze(-1)).unsqueeze(1)
                    + self._type(B, 1, _T_SEL_CTX, dev)); _nopad()

        # flat card-list streams (pad mask from each stream's *_mask)
        for name, t in _CARD_STREAMS:
            toks.append(self._card_stream(o[f"{name}_id"], t, dev))
            pads.append(o[f"{name}_mask"] < 0.5)

        # in-play unit streams (active + bench), both sides
        for side, (at, bt) in (("self", (_T_SELF_ACTIVE, _T_SELF_BENCH)),
                               ("opp", (_T_OPP_ACTIVE, _T_OPP_BENCH))):
            toks.append(self._unit_stream(
                o[f"{side}_unit_top_id"], o[f"{side}_unit_preevo_id"],
                o[f"{side}_unit_tool_id"], o[f"{side}_unit_energy_id"],
                o[f"{side}_unit_attr"], at, bt, dev))
            pads.append(o[f"{side}_unit_mask"] < 0.5)

        # all STATE tokens so far (order matches encoding._TOKEN_LAYOUT); options index into these.
        state_seq = torch.cat(toks, dim=1)                     # [B, N_STATE, d]

        # option tokens: each gathers the EXACT pre-encoder token of the card/unit it references as
        # source/target (global position from encode; -1 -> zero), so the option's view IS that token.
        # A referenced card with NO token (an attached energy/tool) instead carries its id, from which
        # we SYNTHESIZE that card's token (same form as a card-list token: emb + static).
        def _resolve(pos, card):
            safe = pos.clamp(min=0)
            tok = torch.gather(state_seq, 1, safe.unsqueeze(-1).expand(-1, -1, self.d))
            tok = tok * (pos >= 0).unsqueeze(-1).to(tok.dtype)
            need = ((pos < 0) & (card > 0)).unsqueeze(-1).to(tok.dtype)
            synth = (self.card_proj(self.card_emb(card)) + self._static(card)
                     + self._type(card.shape[0], card.shape[1], _T_CARD_SYNTH, card.device))
            return tok + need * synth
        opt_tok = self._opt_stream(_resolve(o["opt_src_pos"], o["opt_src_card"]),
                                   _resolve(o["opt_tgt_pos"], o["opt_tgt_card"]),
                                   o["opt_attr"], o["opt_verb"], o["opt_attack_id"], dev)
        opt_present = o["action_mask"][..., :MAX_OPTIONS] > 0.5
        n_opt = opt_tok.shape[1]

        seq = torch.cat([state_seq, opt_tok], dim=1)
        pad = torch.cat(pads + [~opt_present], dim=1)
        # b=1 (the per-worker opponent + any single-obs inference) is dominated by the
        # O(seq^2) attention over a mostly-PADDED sequence (128 option slots, usually
        # <20 legal). The encoder has no positional encoding (only type embeddings), so
        # attention is permutation-invariant in the keys -> dropping padded tokens before
        # the encoder is EXACT, not an approximation. Cuts the CPU forward several-fold.
        if seq.shape[0] == 1 and _TRUNC_B1:
            keep = (~pad[0]).nonzero(as_tuple=False).squeeze(1)   # present token indices
            enc_t = self.encoder(seq[:, keep])                    # no pad mask: all present
            enc = seq.new_zeros(seq.shape)
            enc[:, keep] = enc_t                                  # scatter back; padded slots
            #                                                       stay 0 but get action-masked
            return enc[:, 0], enc[:, -n_opt:], enc_t.mean(dim=1)  # pooled = mean over PRESENT tokens
        # A fully-padded row would NaN the softmax inside attention. The CLS token
        # is never padded, so no row is all-pad -> safe. (Belt-and-braces: encoder
        # rows still attend to CLS.)
        enc = self.encoder(seq, src_key_padding_mask=pad)
        cls_out = enc[:, 0]                    # CLS
        opt_out = enc[:, -n_opt:]              # the trailing option tokens
        present = (~pad).unsqueeze(-1).to(enc.dtype)              # [B,K,1] mean-pool over non-pad tokens
        pooled = (enc * present).sum(dim=1) / present.sum(dim=1).clamp(min=1.0)
        return cls_out, opt_out, pooled

    def logits_value(self, o: dict):
        cls_out, opt_out, pooled = self._encode(o)
        if self.structured:                                    # verb-conditioned scoring + shared fallback
            verb = o["opt_verb"]                                # [B, MAX_OPTIONS] OptionType per option
            opt_logits = ((opt_out * self.type_query(verb)).sum(-1)
                          + self.type_bias(verb).squeeze(-1)
                          + self.opt_head(opt_out).squeeze(-1))
        else:
            opt_logits = self.opt_head(opt_out).squeeze(-1)    # [B, MAX_OPTIONS]
        submit_logit = self.submit_head(cls_out)               # [B, 1]
        logits = torch.cat([opt_logits, submit_logit], dim=-1)  # [B, N_ACTIONS]
        logits = logits.masked_fill(o["action_mask"] < 0.5, -1e9)
        return logits, self.value_head(torch.cat([cls_out, pooled], dim=-1)).squeeze(-1)

    def get_value(self, o: dict):
        cls_out, _, pooled = self._encode(o)
        return self.value_head(torch.cat([cls_out, pooled], dim=-1)).squeeze(-1)

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
    cfg.pop("ff", None)                                    # FFN width derived as 4*d_model
    use_static = cfg.pop("static", False)                  # opt-in static card features; old ckpts have no key -> OFF (load-compatible)
    use_structured = cfg.pop("structured", False)          # opt-in verb-conditioned action head
    cfg.pop("would_ko", None)                              # metadata flag (env/inference annotate); not a net arg
    feat = card_table.matrix if use_static else None
    return TokenTransformer(card_table.vocab_size, card_feat=feat, structured=use_structured, **cfg)


if __name__ == "__main__":
    # quick smoke: build the net, encode + batch 3 real obs, forward.
    import os
    import pickle

    from .encoding import TokenEncoder

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
