"""Subprocess vector env for cabt.

The native engine holds ONE global battle pointer, so each env must live in its
own process. This runs ``num_envs`` workers, each owning a single ``CabtEnv``,
with auto-reset on episode end and a channel to broadcast opponent-policy weights
for self-play (the opponent runs *inside* the worker on CPU).
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np


def _policy_opponent_factory(net_config):
    """Build a worker-local opponent that plays via a frozen policy snapshot.

    Returns (opponent_fn, set_weights) where opponent_fn(raw_obs, rng)->list[int]
    assembles a full engine selection by running the net with internal buffering.
    """
    import torch
    from rl.encoding import Encoder, SUBMIT_ACTION, build_mask
    from rl.card_features import get_card_table
    from rl.policy import build_net, jit_wrap, obs_to_tensors

    torch.set_num_threads(1)
    enc = Encoder(get_card_table())
    state = {"net": None}

    def set_weights(sd):
        if sd is None:
            state["net"] = None
            return
        net = build_net(enc.cf, enc.cards.vocab_size, net_config)
        net.load_state_dict(sd)
        net.eval()
        state["net"] = jit_wrap(net, enc)    # ~1.7x faster CPU inference (opponent runs here)

    @torch.no_grad()
    def opponent_fn(raw_obs, rng):
        net = state["net"]
        if net is None:  # fall back to random legal
            sel = raw_obs["select"]
            n, k = len(sel["option"]), sel["maxCount"]
            return rng.sample(range(n), min(k, n)) if n else []
        sel = raw_obs["select"]
        picked: list[int] = []
        while True:
            o = enc.encode(raw_obs, set(picked))
            ot = {k: torch.as_tensor(v[None]) for k, v in
                  {kk: (vv.astype("int64") if kk in enc.int_keys else vv.astype("float32"))
                   for kk, vv in o.items()}.items()}
            logits, _ = net.logits_value(ot)
            a = int(logits.argmax(-1).item())
            if a == SUBMIT_ACTION:
                break
            picked.append(a)
            if len(picked) >= sel["maxCount"]:
                break
        return sorted(set(picked))

    return opponent_fn, set_weights


def _worker(remote, parent_remote, env_kwargs, net_config, seed):
    parent_remote.close()
    import logging; logging.disable(logging.CRITICAL)
    from rl.env import CabtEnv, prize_diff_shaping

    env_kwargs = dict(env_kwargs)
    shaping = env_kwargs.pop("shaping", None)
    reward_shaping = prize_diff_shaping(0.1) if shaping == "prize_diff" else None

    opponent_fn, set_weights = _policy_opponent_factory(net_config)
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
            elif cmd == "set_opponent":
                set_weights(data)
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
                 base_seed: int = 0, start_method: str | None = None):
        self.num_envs = num_envs
        ctx = mp.get_context(start_method or ("spawn"))
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(num_envs)])
        self.procs = []
        for i, (wr, r) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=_worker,
                            args=(wr, r, env_kwargs, net_config, base_seed + i),
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
        """Broadcast opponent weights (CPU state_dict) or None (=random)."""
        for r in self.remotes:
            r.send(("set_opponent", state_dict))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None)); r.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
