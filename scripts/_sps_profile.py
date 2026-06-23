"""H100 forward-throughput profile from pre-encoded bc_new.npz: net forward at various batch
sizes + the deck-token-cut benefit on the b=1 (inference/truncation) path. (encode is CPU ~0.56ms.)"""
import time, torch, numpy as np
from rl.encoding import TokenEncoder
from rl.policy import build_token_net
from rl.card_features import get_card_table

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("DEV", torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", flush=True)
ct = get_card_table(); ik = set(TokenEncoder(ct).int_keys)
net = build_token_net(ct, {"arch": "transformer2", "emb_dim": 48, "d_model": 128,
                           "nhead": 4, "nlayers": 2, "static": True}).to(dev).eval()
d = np.load("_kaggle_scout/bc_new.npz"); keys = [k for k in d.files if k != "__labels__"]


def make_ob(B):
    return {k: torch.as_tensor(d[k][:B], dtype=(torch.long if k in ik else torch.float32), device=dev)
            for k in keys}


def bench(ob, tag, it=300):
    for _ in range(20):
        with torch.no_grad():
            net.logits_value(ob)
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(it):
        with torch.no_grad():
            net.logits_value(ob)
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / it * 1000
    B = ob["cls_scalars"].shape[0]
    print("%-30s %.3f ms/call  %.4f ms/sample" % (tag, dt, dt / B), flush=True)


for B in (1, 8, 16, 64, 256, 512, 1024):
    bench(make_ob(B), "fwd b=%d" % B)

ob1 = make_ob(1)
ob1nd = {k: v.clone() for k, v in ob1.items()}
ob1nd["self_deck_mask"] = torch.zeros_like(ob1nd["self_deck_mask"])
ob1nd["opp_deck_mask"] = torch.zeros_like(ob1nd["opp_deck_mask"])
bench(ob1, "b=1 decks PRESENT (~165 tok)")
bench(ob1nd, "b=1 decks CUT (~45 tok)")
print("SPS_DONE", flush=True)
