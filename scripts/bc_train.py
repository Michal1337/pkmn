"""Behavioral-cloning trainer for a given v2 architecture -> reports val top-1 accuracy.

Cheap ARCHITECTURE SELECTOR: supervised CE on expert (winning-side) decisions, no RL. Run with
different --emb-dim/--static/... and compare best_val_acc to pick the architecture.

  python scripts/bc_train.py <data.npz> --emb-dim 48            # baseline
  python scripts/bc_train.py <data.npz> --emb-dim 48 --static   # static features
  python scripts/bc_train.py <data.npz> --emb-dim 128           # bigger embedding
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from rl.card_features import get_card_table
from rl.encoding import TokenEncoder
from rl.policy2 import build_token_net


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data")
    p.add_argument("--emb-dim", type=int, default=48)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--static", action="store_true")
    p.add_argument("--structured", action="store_true", help="verb-conditioned action head")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ct = get_card_table(); enc = TokenEncoder(ct)
    int_keys = set(enc.int_keys)
    dev = torch.device(a.device)

    d = np.load(a.data)
    labels = d["__labels__"]
    N = len(labels)
    keys = [k for k in d.files if k != "__labels__"]
    obs = {k: torch.as_tensor(d[k], dtype=(torch.long if k in int_keys else torch.float32)) for k in keys}
    y = torch.as_tensor(labels, dtype=torch.long)

    g = torch.Generator().manual_seed(a.seed)
    perm = torch.randperm(N, generator=g)
    nval = max(1, int(N * a.val_frac))
    vi, ti = perm[:nval], perm[nval:]

    cfg = {"arch": "transformer2", "emb_dim": a.emb_dim, "d_model": a.d_model,
           "nhead": a.nhead, "nlayers": a.nlayers, "ff": a.ff, "static": a.static,
           "structured": a.structured}
    net = build_token_net(ct, cfg).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss()
    nparams = sum(pp.numel() for pp in net.parameters())
    tag = f"e{a.emb_dim}d{a.d_model}h{a.nhead}{'+static' if a.static else ''}{'+struct' if a.structured else ''}"
    print(f"[bc-train] {tag} params={nparams:,} N={N} train={len(ti)} val={len(vi)} dev={dev}", flush=True)

    def batches(idx, bs, shuffle):
        order = idx[torch.randperm(len(idx))] if shuffle else idx
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            yield {k: obs[k][b].to(dev) for k in keys}, y[b].to(dev)

    best = best3 = 0.0
    for ep in range(a.epochs):
        net.train()
        for ob, yb in batches(ti, a.batch, True):
            loss = lossf(net.logits_value(ob)[0], yb)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); correct = top3 = tot = 0; vloss = 0.0
        with torch.no_grad():
            for ob, yb in batches(vi, a.batch, False):
                lg = net.logits_value(ob)[0]
                vloss += float(lossf(lg, yb)) * len(yb)
                correct += int((lg.argmax(1) == yb).sum())
                top3 += int((lg.topk(3, 1).indices == yb[:, None]).any(1).sum())
                tot += len(yb)
        acc = correct / max(tot, 1); t3 = top3 / max(tot, 1)
        best = max(best, acc); best3 = max(best3, t3)
        if ep % 5 == 0 or ep == a.epochs - 1:
            print(f"[bc-train] {tag} ep{ep} val_acc={acc:.4f} top3={t3:.4f} val_loss={vloss / max(tot, 1):.4f}", flush=True)
    print(f"[bc-train] RESULT {tag}: best_val_acc={best:.4f} best_top3={best3:.4f} params={nparams:,}", flush=True)
    if a.out:
        torch.save({"net": net.state_dict(), "net_config": cfg, "bc_val_acc": best}, a.out)


if __name__ == "__main__":
    main()
