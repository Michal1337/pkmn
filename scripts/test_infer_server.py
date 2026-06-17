"""Standalone test for rl.infer_server: correctness vs a local net + multi-client
throughput (the batching win). Must be run as a guarded script (Windows spawn)."""
import logging; logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time


def _hammer(client, encoded, secs, retq):
    t0 = time.time(); n = 0
    while time.time() - t0 < secs:
        client.logits_value(encoded); n += 1
    retq.put(n)


def main():
    import numpy as np, torch
    from rl.card_features import get_card_table
    from rl.encoding import Encoder
    from rl.policy import build_net
    from rl.env import load_deck
    from rl import search_agent as SA
    from rl.infer_server import InferenceServer
    from sdk_cg.game import battle_start, battle_select, battle_finish
    torch.set_num_threads(1)
    enc = Encoder(get_card_table())
    net = build_net(enc.cf, enc.cards.vocab_size, {"emb_dim": 32}); net.eval()
    deck = load_deck()
    try: battle_finish()
    except Exception: pass
    obs = battle_start(deck, deck)[0]
    for _ in range(16):
        if obs.get("select") is None:
            obs = battle_select([int(c) for c in deck]); continue
        obs = battle_select(SA._net_greedy_select(obs, net, enc, "cpu"))
        if obs["current"]["result"] >= 0:
            break
    arr = enc.encode(obs)

    srv = InferenceServer({"arch": "mlp", "emb_dim": 32}, n_clients=8, device="cpu", max_batch=64)
    srv.set_weights({k: v.cpu() for k, v in net.state_dict().items()})

    # --- correctness ---
    c = srv.client(0)
    srv_lg = np.asarray(c.logits_value(arr))
    o = {k: torch.as_tensor(arr[k][None], dtype=(torch.long if k in enc.int_keys else torch.float32)) for k in arr}
    with torch.no_grad():
        loc_lg = net.logits_value(o)[0].numpy(); loc_v = float(net.get_value(o)[0])
    srv_v = c.get_value(arr)
    print(f"[correctness] logits max diff {np.abs(srv_lg-loc_lg).max():.2e}  "
          f"argmax match {int(srv_lg.argmax()==loc_lg.argmax())}  value diff {abs(srv_v-loc_v):.2e}")

    # --- throughput: 8 concurrent clients vs single-process local baseline ---
    ctx = mp.get_context("spawn"); retq = ctx.Queue(); secs = 4.0
    procs = [ctx.Process(target=_hammer, args=(srv.client(i), arr, secs, retq)) for i in range(8)]
    [p.start() for p in procs]
    total = sum(retq.get() for _ in procs)
    [p.join() for p in procs]
    server_rps = total / secs

    t0 = time.time(); n = 0
    with torch.no_grad():
        while time.time() - t0 < secs:
            net.logits_value(o); n += 1
    local_rps = n / secs
    print(f"[throughput] server (8 clients): {server_rps:6.0f} forwards/s   "
          f"local single-proc: {local_rps:6.0f}/s   speedup {server_rps/local_rps:.1f}x")
    srv.close(); print("server closed OK")


if __name__ == "__main__":
    mp.freeze_support()
    main()
