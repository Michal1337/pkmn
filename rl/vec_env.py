"""Subprocess vector env for cabt.

The native engine holds ONE global battle pointer, so each env must live in its
own process. This runs ``num_envs`` workers, each owning a single ``CabtEnv``,
with auto-reset on episode end and a channel to broadcast opponent-policy weights
for self-play.

Two opponent execution modes (``opponent_mode``):

* ``"local"`` (default): the opponent net runs INSIDE each worker on CPU. A central
  server was once tried for the MLP and measured ~2-3x SLOWER -- one process
  serializing IPC round-trips beat per-worker local forwards because the MLP forward
  is sub-ms (round-trip dominates).

* ``"server"``: the opponent ENCODES locally (cheap) but routes the forward to a
  batched inference server -- a THREAD in the main process holding the snapshot net
  on the GPU. While the main thread blocks in ``step`` waiting on workers, the server
  thread batches the workers' concurrent forward requests into ONE GPU forward. This
  WINS for the v2 transformer (CPU forward is expensive, so per-worker forwards thrash
  the cores), the opposite regime to the cheap MLP. No second CUDA context (shares the
  main process's), no weight pickling, and workers hold no net.

  The encoded obs (fixed shapes) crosses to the server via SHARED MEMORY, not the pipe:
  each worker writes its obs into row ``i`` of a shared batch buffer and signals the
  server with just its index (a few bytes); the server reads the ready rows as a
  zero-copy numpy view, forwards on GPU, writes logits into a shared response buffer,
  and acks. This removes the ~40 KB pickle/unpickle per request that otherwise caps the
  single server thread.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from multiprocessing import resource_tracker, shared_memory
from multiprocessing.connection import wait

import numpy as np


def _policy_opponent_factory(net_config):
    """Worker-local opponent playing a frozen snapshot via a jit_wrap'd net (~1.7x CPU,
    independent per worker -> scales with cores). Returns (opponent_fn, set_weights)."""
    import torch
    from rl.card_features import get_card_table
    torch.set_num_threads(1)

    v2 = net_config.get("arch") == "transformer2"
    if v2:    # token transformer: the env threads the opponent's true deck + shared tracker; no jit (trace is fragile)
        from rl.encoding import TokenEncoder
        from rl.encoding import SUBMIT_ACTION
        from rl.policy2 import build_token_net
        enc = TokenEncoder(get_card_table())
    else:
        from rl.encoding import Encoder, SUBMIT_ACTION
        from rl.policy import build_net, jit_wrap
        enc = Encoder(get_card_table())
    state = {"net": None}

    def set_weights(sd):
        if sd is None:
            state["net"] = None
            return
        sd = {k: torch.as_tensor(v) for k, v in sd.items()}   # numpy (from the pipe) -> tensors
        if v2:
            net = build_token_net(enc.cards, net_config)
            net.load_state_dict(sd); net.eval()
            state["net"] = net                       # raw net (transformer doesn't jit-freeze cleanly)
        else:
            net = build_net(enc.cf, enc.cards.vocab_size, net_config)
            net.load_state_dict(sd); net.eval()
            state["net"] = jit_wrap(net, enc)        # frozen TorchScript, ~1.7x

    @torch.no_grad()
    def opponent_fn(raw_obs, rng, deck=None, tracker=None, ability_slots=None):
        net = state["net"]
        sel = raw_obs["select"]
        if net is None:                              # random legal (warmup / no snapshot)
            n, k = len(sel["option"]), sel["maxCount"]
            return rng.sample(range(n), min(k, n)) if n else []
        picked: list[int] = []
        while True:
            enc_obs = (enc.encode(raw_obs, set(picked), self_deck=deck, tracker=tracker,
                                  ability_slots=ability_slots)
                       if v2 else enc.encode(raw_obs, set(picked)))
            o = {k: torch.as_tensor(v[None], dtype=(torch.long if k in enc.int_keys else torch.float32))
                 for k, v in enc_obs.items()}
            logits, _ = net.logits_value(o)
            a = int(logits.argmax(-1).item())
            if a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                break
        return sorted(set(picked))

    return opponent_fn, set_weights


def _attach_shm(name):
    """Attach to a main-owned shared-memory block. Unregister from THIS process's
    resource_tracker so worker exit doesn't try to unlink a block the main owns."""
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return shm


def _server_opponent_factory(net_config, opp_remote, srv):
    """Opponent that ENCODES locally (cheap) and routes the FORWARD to the main-process
    server: writes its obs into shared-memory row ``srv['idx']``, signals the server with
    its index, and reads logits back from the shared response row. No per-worker net.
    ``set_mode(bool)`` toggles server use (False = random legal, the warmup curriculum)."""
    import torch
    from rl.card_features import get_card_table
    torch.set_num_threads(1)

    v2 = net_config.get("arch") == "transformer2"
    if v2:
        from rl.encoding import TokenEncoder
        from rl.encoding import SUBMIT_ACTION
        enc = TokenEncoder(get_card_table())
    else:
        from rl.encoding import Encoder, SUBMIT_ACTION
        enc = Encoder(get_card_table())

    idx, n = srv["idx"], srv["n"]
    shms = {k: _attach_shm(name) for k, name in srv["names"].items()}
    bufs = {k: np.ndarray((n, *shp), dtype=dt, buffer=shms[k].buf)
            for k, (shp, dt) in srv["spec"].items()}
    resp_shm = _attach_shm(srv["resp"])
    resp = np.ndarray((n, srv["resp_w"]), dtype=np.float32, buffer=resp_shm.buf)
    # CRITICAL: keep the SharedMemory objects alive for the closure's lifetime. The numpy
    # views (bufs/resp) hold only the mmap memoryview; if the SharedMemory objects are GC'd
    # their __del__ closes the mmap -> the views point to freed memory -> C-level SEGFAULT
    # on the next access (silent: no catchable Python exception). Stash them in `state`.
    state = {"use_server": False, "_keep": (shms, resp_shm)}

    def set_mode(flag):
        state["use_server"] = bool(flag)

    def _random(sel, rng):
        m, k = len(sel["option"]), sel["maxCount"]
        return rng.sample(range(m), min(k, m)) if m else []

    def opponent_fn(raw_obs, rng, deck=None, tracker=None, ability_slots=None):
        sel = raw_obs["select"]
        if not state["use_server"]:                  # warmup: random legal, no round-trip
            return _random(sel, rng)
        picked: list[int] = []
        while True:
            enc_obs = (enc.encode(raw_obs, set(picked), self_deck=deck, tracker=tracker,
                                  ability_slots=ability_slots)
                       if v2 else enc.encode(raw_obs, set(picked)))
            for k, v in enc_obs.items():
                bufs[k][idx] = v                     # write my row in shared memory
            opp_remote.send(idx)                     # tiny signal (few bytes)
            if not opp_remote.poll(10.0):            # server unresponsive -> degrade, never hang
                return _random(sel, rng)
            ok = opp_remote.recv()                   # ack (True), or None on server error
            if ok is None:
                return _random(sel, rng)
            a = int(np.argmax(resp[idx]))            # logits already action-masked by the net
            if a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                break
        return sorted(set(picked))

    return opponent_fn, set_mode


def _worker(remote, parent_remote, env_kwargs, net_config, seed, srv=None):
    parent_remote.close()
    import logging; logging.disable(logging.CRITICAL)
    _worker_main(remote, env_kwargs, net_config, seed, srv)


def _worker_main(remote, env_kwargs, net_config, seed, srv=None):
    from rl.env import CabtEnv, prize_diff_shaping

    env_kwargs = dict(env_kwargs)
    shaping = env_kwargs.pop("shaping", None)
    reward_shaping = prize_diff_shaping(0.1) if shaping == "prize_diff" else None

    if srv is not None:                            # server mode: forward routed to main proc
        opponent_fn, set_opp = _server_opponent_factory(net_config, srv["opp"], srv)
    else:                                          # local mode: net runs in-worker
        opponent_fn, set_opp = _policy_opponent_factory(net_config)
    encoder = None
    if net_config.get("arch") == "transformer2":          # v2 token encoder for the learner side
        from rl.encoding import TokenEncoder
        from rl.card_features import get_card_table
        encoder = TokenEncoder(get_card_table())
    env = CabtEnv(seed=seed, opponent_fn=opponent_fn, reward_shaping=reward_shaping,
                  encoder=encoder, v2=encoder is not None, **env_kwargs)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == "step":
                try:
                    obs, r, term, trunc, info = env.step(data)
                    done = term or trunc
                    if done:
                        info = {**info, "truncated": trunc}
                        if term:                       # only a real game end has a terminal reward;
                            info["terminal_reward"] = r  # truncation (max_steps) is not terminal
                        obs, _ = env.reset()  # auto-reset
                except Exception:
                    # a bad game/engine state must NOT kill the worker -> it would cascade
                    # (ConnectionReset in the main) and fail the whole job. Recover: reset and
                    # report a neutral episode boundary; one lost transition >> a dead run.
                    try:
                        obs, _ = env.reset()
                    except Exception:
                        obs = env._encode()
                    r, done, info = 0.0, True, {"recovered": True}
                remote.send((obs, r, done, info))
            elif cmd == "set_opponent":            # local: data=state_dict|None ; server: data=use_server bool
                set_opp(data)
                remote.send(True)
            elif cmd == "close":
                env.close()
                remote.send(True)
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()


def _ref_spec(net_config, env_kwargs):
    """Per-key encoded-obs (shape, dtype) from a throwaway CabtEnv reset -- shapes are
    fixed (the encoder pads to constants), so one sample defines the shared buffers."""
    from rl.env import CabtEnv
    from rl.card_features import get_card_table
    ek = dict(env_kwargs); ek.pop("shaping", None)
    v2 = net_config.get("arch") == "transformer2"
    if v2:
        from rl.encoding import TokenEncoder
        enc = TokenEncoder(get_card_table())
    else:
        from rl.encoding import Encoder
        enc = Encoder(get_card_table())
    env = CabtEnv(seed=0, encoder=enc, v2=v2, **ek)
    try:
        obs, _ = env.reset()
    finally:
        env.close()
    return {k: (tuple(v.shape), v.dtype) for k, v in obs.items()}


class SubprocVecEnv:
    def __init__(self, num_envs: int, env_kwargs: dict, net_config: dict,
                 base_seed: int = 0, start_method: str | None = None,
                 server_device: str = "cpu", opponent_mode: str = "local"):
        self.num_envs = num_envs
        self.opponent_mode = opponent_mode
        self.net_config = net_config
        ctx = mp.get_context(start_method or ("spawn"))
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])

        srvs = [None] * num_envs
        self._shm = []
        if opponent_mode == "server":
            spec = _ref_spec(net_config, env_kwargs)               # {k: (shape, dtype)}
            resp_w = int(np.prod(spec["action_mask"][0]))          # N_ACTIONS
            names = {}
            self._obs_bufs = {}
            for k, (shp, dt) in spec.items():
                nbytes = max(num_envs * int(np.prod(shp)) * np.dtype(dt).itemsize, 1)
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self._shm.append(shm); names[k] = shm.name
                self._obs_bufs[k] = np.ndarray((num_envs, *shp), dtype=dt, buffer=shm.buf)
            resp_shm = shared_memory.SharedMemory(create=True, size=num_envs * resp_w * 4)
            self._shm.append(resp_shm)
            self._resp = np.ndarray((num_envs, resp_w), dtype=np.float32, buffer=resp_shm.buf)

            opp_pairs = [ctx.Pipe() for _ in range(num_envs)]
            self.opp_conns = [a for a, _ in opp_pairs]             # main-side ends (server reads)
            for i in range(num_envs):
                srvs[i] = {"idx": i, "n": num_envs, "names": names, "spec": spec,
                           "resp": resp_shm.name, "resp_w": resp_w, "opp": opp_pairs[i][1]}
        else:
            self.opp_conns = None

        self.procs = []
        for i, (wr, r) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=_worker,
                            args=(wr, r, env_kwargs, net_config, base_seed + i, srvs[i]),
                            daemon=True)
            p.start()
            self.procs.append(p)
        for wr in self.work_remotes:
            wr.close()
        if opponent_mode == "server":
            for i in range(num_envs):
                srvs[i]["opp"].close()                             # workers hold their own copies
            self._start_server(net_config, server_device)

    # ---- batched inference server (thread in THIS process) --------------------
    def _start_server(self, net_config, device):
        import torch
        from rl.card_features import get_card_table
        dev = torch.device(device)
        if net_config.get("arch") == "transformer2":
            from rl.encoding import TokenEncoder
            from rl.policy2 import build_token_net
            enc = TokenEncoder(get_card_table())
            net = build_token_net(enc.cards, net_config)
        else:
            from rl.encoding import Encoder
            from rl.policy import build_net
            enc = Encoder(get_card_table())
            net = build_net(enc.cf, enc.cards.vocab_size, net_config)
        self._srv_net = net.to(dev).eval()
        self._srv_dev = dev
        self._srv_lock = threading.Lock()
        self._srv_stop = threading.Event()
        import os
        # Dynamic-batching window. Default OFF: each opponent decision is on a worker's
        # critical path, so waiting to coalesce bigger batches adds latency that hurts more
        # than it helps (measured: 0ms best, monotonically worse to 4ms). Tunable for other regimes.
        self._srv_coalesce = float(os.environ.get("SRV_COALESCE_MS", "0")) / 1000.0
        self._srv_thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._srv_thread.start()

    def _serve_loop(self):
        import torch
        conns = list(self.opp_conns)
        keys = list(self._obs_bufs.keys())
        coalesce = self._srv_coalesce                          # dynamic-batching window (s)
        while not self._srv_stop.is_set():
            ready = wait(conns, timeout=0.25)
            if not ready:
                continue
            # Dynamic batching: wait() returns the instant ONE worker is ready, but workers
            # are staggered by between-decision engine work -> tiny GPU batches. Briefly pull
            # in stragglers so one GPU forward serves many workers (fewer host<->GPU syncs).
            ready = list(ready)
            if coalesce > 0:
                remaining = [c for c in conns if c not in ready]
                if remaining:
                    ready += list(wait(remaining, timeout=coalesce))
            pend = []                                              # (conn, worker_idx)
            for c in ready:
                try:
                    pend.append((c, c.recv()))
                except EOFError:
                    conns.remove(c)
            if not pend:
                continue
            rows = [idx for _, idx in pend]
            try:
                batch = {k: torch.as_tensor(self._obs_bufs[k][rows], device=self._srv_dev)
                         for k in keys}
                with torch.no_grad(), self._srv_lock:
                    logits = self._srv_net.logits_value(batch)[0]
                self._resp[rows] = logits.cpu().numpy()            # write shared response rows
                for c, _ in pend:
                    c.send(True)                                   # ack: response ready
            except Exception:
                for c, _ in pend:                                  # never deadlock the workers
                    try:
                        c.send(None)
                    except Exception:
                        pass

    def _stack(self, obs_list):
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        obs, infos = zip(*[r.recv() for r in self.remotes])
        return self._stack(obs), list(infos)

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", int(a)))
        results = [r.recv() for r in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return (self._stack(obs),
                np.asarray(rews, dtype=np.float32),
                np.asarray(dones, dtype=np.bool_),
                list(infos))

    def set_opponent(self, state_dict):
        """Set the snapshot opponent (or None = random).

        server mode: load the snapshot into the server net (one place) and tell workers
        to route forwards to it (None -> workers fall back to random legal). local mode:
        broadcast the weights to each worker as NUMPY -- sending torch tensors over mp
        uses fd-based storage sharing, and many workers x many tensors exhausts the open
        file limit ('Too many open files') at the broadcast; numpy pickles by value."""
        if self.opponent_mode == "server":
            if state_dict is not None:
                import torch
                sd = {k: torch.as_tensor(v) for k, v in state_dict.items()}
                with self._srv_lock:
                    self._srv_net.load_state_dict(sd)
                    self._srv_net.eval()
            flag = state_dict is not None
            for r in self.remotes:
                r.send(("set_opponent", flag))
            return [r.recv() for r in self.remotes]

        payload = (None if state_dict is None
                   else {k: v.detach().cpu().numpy() for k, v in state_dict.items()})
        for r in self.remotes:
            r.send(("set_opponent", payload))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None)); r.recv()
            except Exception:
                pass
        if self.opponent_mode == "server":
            self._srv_stop.set()
            if self._srv_thread.is_alive():
                self._srv_thread.join(timeout=5)
            for shm in self._shm:
                try:
                    shm.close(); shm.unlink()
                except Exception:
                    pass
        for p in self.procs:
            p.join(timeout=5)
