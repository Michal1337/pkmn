"""Central inference server: batches net forwards across env workers.

Profiling: self-play SPS is bottlenecked by the per-worker CPU opponent/MCTS forward
(~3 ms, op-dispatch-bound). This routes every worker's forward to ONE process holding
the net (on GPU on the cluster), which dynamically batches concurrent requests into a
single forward -> ~B workers' forwards become one batched forward (CPU ~5.5x/B=32, GPU
much more).

Protocol (all via a spawn-context mp.Queue + per-client Pipe):
  worker -> server:  (client_id, kind, payload)
      kind 'lv'/'gv' : payload = encoded obs (numpy dict, no batch dim) -> response on pipe
      kind 'weights' : payload = cpu state_dict   -> ack on ack_q (no pipe response)
      kind 'stop'    : shut down
  server -> worker:  the per-row result (numpy) on that client's response Pipe.

A ServerNet exposes the usual logits_value/get_value interface so it drops into the
opponent / search code unchanged (it takes the *encoded numpy* obs, not tensors).
"""
from __future__ import annotations

import multiprocessing as mp
import queue as _q

import numpy as np


def _serve(req_q, conns, ack_q, net_config, device, int_keys, max_batch):
    import logging; logging.disable(logging.CRITICAL)
    import torch
    from .card_features import get_card_table
    from .encoding import Encoder
    from .policy import build_net
    torch.set_num_threads(1)
    dev = torch.device(device)
    enc = Encoder(get_card_table())
    net = build_net(enc.cf, enc.cards.vocab_size, net_config).to(dev).eval()

    def run_batch(grp, method):
        keys = list(grp[0][2].keys())
        batch = {k: torch.as_tensor(np.stack([it[2][k] for it in grp]),
                                    dtype=(torch.long if k in int_keys else torch.float32),
                                    device=dev) for k in keys}
        with torch.no_grad():
            if method == "lv":
                out = net.logits_value(batch)[0]
            else:
                out = net.get_value(batch)
        out = out.cpu().numpy()
        for j, it in enumerate(grp):
            conns[it[0]].send(out[j])

    while True:
        items = [req_q.get()]
        while len(items) < max_batch:
            try:
                items.append(req_q.get_nowait())
            except _q.Empty:
                break
        infer = []
        for it in items:
            cid, kind, payload = it
            if kind == "weights":
                net.load_state_dict(payload); net.eval(); ack_q.put(cid)
            elif kind == "stop":
                ack_q.put(cid); return
            else:
                infer.append(it)
        for method in ("lv", "gv"):
            grp = [it for it in infer if it[1] == method]
            if grp:
                run_batch(grp, method)


class ServerNet:
    """Worker-side proxy with the net interface; forwards run on the server, batched.
    Takes the *encoded numpy* obs dict (enc.encode output), not tensors."""

    def __init__(self, cid, req_q, conn):
        self.cid, self.req_q, self.conn = cid, req_q, conn

    def logits_value(self, encoded):
        self.req_q.put((self.cid, "lv", encoded))
        return self.conn.recv()              # numpy [N_ACTIONS]

    def get_value(self, encoded):
        self.req_q.put((self.cid, "gv", encoded))
        return float(self.conn.recv())


class InferenceServer:
    def __init__(self, net_config, n_clients, device="cpu", max_batch=64, ctx=None):
        self.ctx = ctx or mp.get_context("spawn")
        from .encoding import Encoder
        from .card_features import get_card_table
        int_keys = Encoder(get_card_table()).int_keys
        self.req_q = self.ctx.Queue()
        self.ack_q = self.ctx.Queue()
        pipes = [self.ctx.Pipe() for _ in range(n_clients)]
        self._srv_conns = [a for a, _ in pipes]
        self.client_conns = [b for _, b in pipes]   # hand conn i to worker i
        self.proc = self.ctx.Process(
            target=_serve,
            args=(self.req_q, self._srv_conns, self.ack_q, net_config, device, int_keys, max_batch),
            daemon=True)
        self.proc.start()

    def set_weights(self, state_dict):
        self.req_q.put((-1, "weights", state_dict))
        self.ack_q.get()                      # wait until applied

    def client(self, cid):
        return ServerNet(cid, self.req_q, self.client_conns[cid])

    def close(self):
        try:
            self.req_q.put((-1, "stop", None)); self.ack_q.get(timeout=5)
        except Exception:
            pass
        self.proc.join(timeout=5)
