"""Verify + benchmark dynamic sequence truncation for the b=1 opponent forward.
Run once to save a reference (identical weights + identical encoded obs), edit
policy2._encode to truncate, run again -> compares logits to ref and times the fwd.
"""
import os
import time
import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(1)

from rl.card_features import get_card_table
from rl.encoding import TokenEncoder
from rl.policy2 import build_token_net

REF = "/tmp/trunc_ref.pt"
OBS = "/tmp/trunc_obs.pt"

enc = TokenEncoder(get_card_table())
ck = torch.load("_tourney/v2_baseline/latest.pt", map_location="cpu")
net = build_token_net(enc.cards, ck.get("net_config", {})); net.load_state_dict(ck["net"]); net.eval()

if os.path.exists(OBS):
    t = torch.load(OBS)
else:
    import rl.decks_generated as dg
    from sdk_cg.game import battle_start, battle_select, battle_finish
    decks = list(dg.GENERATED.values())
    try: battle_finish()
    except Exception: pass
    obs = battle_start(decks[0], decks[1])[0]
    while obs.get("select") is None and obs["current"]["result"] < 0:
        obs = battle_select([int(c) for c in decks[0]])
    try: battle_finish()
    except Exception: pass
    o = enc.encode(obs, set(), self_deck=decks[0])
    int_keys = enc.int_keys
    t = {k: torch.as_tensor(np.asarray(v)[None], dtype=(torch.long if k in int_keys else torch.float32))
         for k, v in o.items()}
    torch.save(t, OBS)

logits, value = net.logits_value(t)

def time_ms(fn, n=200):
    for _ in range(5): fn()
    t0 = time.time()
    for _ in range(n): fn()
    return (time.time() - t0) / n * 1000

fwd = time_ms(lambda: net.logits_value(t))

if os.path.exists(REF):
    ref = torch.load(REF)
    # compare only LEGAL option logits + value (padded logits are -1e9 either way)
    mask = t["action_mask"][0] > 0.5
    dl = (logits[0][mask] - ref["logits"][0][mask]).abs().max().item()
    dv = (value - ref["value"]).abs().max().item()
    print(f"TRUNCATED: fwd={fwd:.2f} ms   max|Δlogit(legal)|={dl:.2e}   max|Δvalue|={dv:.2e}")
    print("  -> IDENTICAL" if dl < 1e-3 and dv < 1e-3 else "  -> *** MISMATCH ***")
else:
    torch.save({"logits": logits, "value": value}, REF)
    n_legal = int((t["action_mask"][0] > 0.5).sum())
    seq_present = int(sum(int((t[f"{nm}_mask"][0] > 0.5).sum()) for nm in
                          ("hand", "self_discard", "opp_discard") if f"{nm}_mask" in t))
    print(f"BASELINE (full seq): fwd={fwd:.2f} ms   legal_options={n_legal}/128")
    print("  saved reference; now edit policy2._encode and re-run.")
