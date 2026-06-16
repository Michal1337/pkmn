"""Pure-numpy forward pass for the trained ActorCritic (inference only).

The Kaggle cabt runtime may not have torch, and bundling torch is huge. The net
is tiny and our encoder is already numpy, so we replay the exact forward pass
with numpy from weights saved as an .npz. This mirrors ActorCritic.logits_value
in rl/policy.py — keep the two in sync if the architecture changes.

Self-contained: no torch, no rl imports (MAX_OPTIONS is read from obs shapes).
"""

from __future__ import annotations

import numpy as np


def _relu(x):
    return np.maximum(x, 0.0)


def _linear(x, W, b):
    # torch Linear: y = x @ W.T + b, W is (out, in)
    return x @ W.T + b


class NumpyPolicy:
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.w = {k: d[k].astype(np.float32) for k in d.files}

    def _mlp(self, x, prefix):
        """Apply a Sequential MLP (Linear/ReLU/Linear/...) by its weight keys."""
        idxs = sorted({int(k.split(".")[1]) for k in self.w
                       if k.startswith(prefix + ".") and k.endswith(".weight")})
        for i, idx in enumerate(idxs):
            x = _linear(x, self.w[f"{prefix}.{idx}.weight"], self.w[f"{prefix}.{idx}.bias"])
            if i < len(idxs) - 1:
                x = _relu(x)
        return x

    def _cards(self, static, ids):
        """static [...,K,cf], ids [...,K] -> [...,K,card_h] (card_enc(static ++ emb))."""
        emb = self.w["card_emb.weight"][ids]            # fancy index
        return self._mlp(np.concatenate([static, emb], axis=-1), "card_enc")

    def logits(self, o: dict) -> np.ndarray:
        """o: single (unbatched) encoded obs dict -> [N_ACTIONS] masked logits."""
        max_opt = o["opt_dyn"].shape[0]

        def board(side):
            a_c = self._cards(o[f"{side}_active_static"], o[f"{side}_active_id"])   # [1,ch]
            a_x = self._mlp(np.concatenate([a_c, o[f"{side}_active_dyn"]], -1), "slot_enc")
            a = (a_x * o[f"{side}_active_dyn"][:, 0:1])[0]                          # [ch] 0 if no active
            b_c = self._cards(o[f"{side}_bench_static"], o[f"{side}_bench_id"])     # [5,ch]
            b_x = self._mlp(np.concatenate([b_c, o[f"{side}_bench_dyn"]], -1), "slot_enc")
            m = o[f"{side}_bench_dyn"][:, 0:1]
            b = (b_x * m).sum(0) / max(m.sum(), 1.0)
            return a, b
        self_a, self_b = board("self")
        opp_a, opp_b = board("opp")

        hand_c = self._cards(o["hand_static"], o["hand_id"])                   # [15,ch]
        hand_x = self._mlp(hand_c, "hand_enc")
        hm = o["hand_mask"][:, None]
        hand = (hand_x * hm).sum(0) / max(hm.sum(), 1.0)

        self_disc = self._mlp(o["self_discard_agg"], "disc_enc")
        opp_disc = self._mlp(o["opp_discard_agg"], "disc_enc")
        stad_c = self._cards(o["stadium_static"][None, :], o["stadium_id"])[0]  # [ch]
        stad = self._mlp(stad_c, "stad_enc")

        st = self.w["sel_type_emb.weight"][int(o["select_type"][0])]
        sc = self.w["sel_ctx_emb.weight"][int(o["select_context"][0])]

        g = np.concatenate([
            o["scalars"], st, sc, o["self_player"], o["opp_player"],
            self_a, self_b, opp_a, opp_b, hand, self_disc, opp_disc, stad,
        ])
        h = self._mlp(g, "trunk")                                              # [256]

        opt_c = self._cards(o["opt_card_static"], o["opt_card_id"])            # [64,ch]
        opt = self._mlp(np.concatenate([opt_c, o["opt_dyn"]], -1), "opt_enc")  # [64,oh]
        h_exp = np.broadcast_to(h, (max_opt, h.shape[0]))
        opt_logits = self._mlp(np.concatenate([h_exp, opt], -1), "opt_score")[:, 0]  # [64]
        submit_logit = self._mlp(h, "submit_score")                           # [1]
        logits = np.concatenate([opt_logits, submit_logit])                   # [65]
        return np.where(o["action_mask"] < 0.5, -1e9, logits)

    def select(self, o: dict) -> int:
        """Greedy legal action index."""
        return int(np.argmax(self.logits(o)))
