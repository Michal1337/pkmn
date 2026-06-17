"""Subprocess vector env for cabt.

The native engine holds ONE global battle pointer, so each env must live in its
own process. This runs ``num_envs`` workers, each owning a single ``CabtEnv``,
with auto-reset on episode end and a channel to broadcast opponent-policy weights
for self-play (the opponent runs *inside* the worker on CPU).
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _policy_opponent_factory(client):
    """Build a worker opponent that runs the snapshot net via the central inference
    SERVER (batched across workers). `client` is a rl.infer_server.ServerNet.
    Returns (opponent_fn, set_use) where set_use(bool) toggles server vs random."""
    from rl.encoding import Encoder, SUBMIT_ACTION
    from rl.card_features import get_card_table

    enc = Encoder(get_card_table())
    state = {"use": False}

    def set_use(flag):
        state["use"] = bool(flag)

    def opponent_fn(raw_obs, rng):
        sel = raw_obs["select"]
        if not state["use"]:                  # random legal (warmup / no snapshot yet)
            n, k = len(sel["option"]), sel["maxCount"]
            return rng.sample(range(n), min(k, n)) if n else []
        picked: list[int] = []
        while True:
            logits = client.logits_value(enc.encode(raw_obs, set(picked)))   # batched on the server
            if logits is None:                # server hit an error -> random legal, never hang
                n, k = len(sel["option"]), sel["maxCount"]
                return rng.sample(range(n), min(k, n)) if n else []
            a = int(logits.argmax())
            if a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                break
        return sorted(set(picked))

    return opponent_fn, set_use


def _worker(remote, parent_remote, env_kwargs, client, seed):
    parent_remote.close()
    import logging; logging.disable(logging.CRITICAL)
    from rl.env import CabtEnv, prize_diff_shaping

    env_kwargs = dict(env_kwargs)
    shaping = env_kwargs.pop("shaping", None)
    reward_shaping = prize_diff_shaping(0.1) if shaping == "prize_diff" else None

    opponent_fn, set_use = _policy_opponent_factory(client)
    env = CabtEnv(seed=seed, opponent_fn=opponent_fn,
                  reward_shaping=reward_shaping, **env_kwargs)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                obs, info = env.reset()
                remote.send((obs, info))
            elif cmd == "step":
                obs, r, term, trunc, info = env.step(data)
                done = term or trunc
                if done:
                    info = {**info, "terminal_reward": r, "truncated": trunc}
                    obs, _ = env.reset()  # auto-reset
                remote.send((obs, r, done, info))
            elif cmd == "set_opponent":            # data = use-server flag (bool)
                set_use(data)
                remote.send(True)
            elif cmd == "close":
                env.close()
                remote.send(True)
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()


class SubprocVecEnv:
    def __init__(self, num_envs: int, env_kwargs: dict, net_config: dict,
                 base_seed: int = 0, start_method: str | None = None,
                 server_device: str = "cpu"):
        from rl.infer_server import InferenceServer
        self.num_envs = num_envs
        ctx = mp.get_context(start_method or ("spawn"))
        # central inference server: batches the snapshot-opponent forwards across workers
        self.server = InferenceServer(net_config, num_envs, device=server_device,
                                      max_batch=max(num_envs * 2, 256), ctx=ctx)  # amortize scattered arrivals
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.procs = []
        for i, (wr, r) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=_worker,
                            args=(wr, r, env_kwargs, self.server.client(i), base_seed + i),
                            daemon=True)
            p.start()
            self.procs.append(p)
        for wr in self.work_remotes:
            wr.close()

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
        """Set the snapshot opponent: weights go to the inference server (once), and
        workers are toggled to use it. state_dict=None -> random opponent."""
        use = state_dict is not None
        if use:
            self.server.set_weights(state_dict)        # central, batched
        for r in self.remotes:
            r.send(("set_opponent", use))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None)); r.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
        try:
            self.server.close()
        except Exception:
            pass
