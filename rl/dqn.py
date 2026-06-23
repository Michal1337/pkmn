"""Off-policy n-step Double *Dueling* DQN self-play for the cabt v2 pointer net — the
off-policy counterpart of the on-policy PPO in train.py.

v2 (2026-06-20), after an adversarial 4-lens audit of v1 found v1 collapsed (Q->0 everywhere,
flat tiny loss, greedy-Q at only early-PPO strength). v1's confirmed root causes + the fixes here:

  1. SPARSE terminal reward + 1-step bootstrap -> the Bellman target was ~0 on every non-terminal
     step, so the net trivially learned Q~=0. FIX: (a) dense `prize_diff` reward shaping, and
     (b) N-STEP returns (default n=8) so reward propagates n steps per update instead of 1.
  2. Reusing the softmax/pointer LOGITS as Q is scale-free (no anchor to the +/-1 reward scale),
     and the value head was unused. FIX: a DUELING Q-head Q(s,a)=tanh(V(s)) + A(s,a) - mean_legal A,
     where V=value_head (now trained) carries the scale and A=option logits the preferences.
     NOTE argmax_a Q == argmax_a logits (V, mean are per-state constants), so vec_env's opponent
     (which plays argmax(logits)) already plays correct greedy-Q -> no vec_env change needed.
  3. Degenerate single frozen self-play opponent. FIX: a snapshot POOL (like PPO) sampled at refresh.
  4. Too-low update rate + uninformative metric. FIX: grad_steps>1 (replay reuse), faster eps anneal,
     log Q magnitude/spread + the eps-greedy winrate (true greedy strength = checkpoint h2h).
  + truncation (max_steps) is NO LONGER conflated with termination (info["truncated"] => no bootstrap-zero).

    python -m rl.dqn --decks mega_abomasnow --num-envs 96 --total-timesteps 20000000 \
        --n-step 8 --shaping prize_diff --out $HOME/pkmn_runs/dqn
"""
from __future__ import annotations

import argparse
import collections
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .vec_env import SubprocVecEnv
from .card_features import get_card_table
from .encoding import TokenEncoder
from .policy import build_token_net, obs_to_tensors
from .train import resolve_deck_pool


class Replay:
    """CPU replay of n-step transitions; obs kept as dict-of-numpy, sampled onto the GPU.

    A transition is (s, a, R, s2, done, n): R is the n-step discounted reward sum, s2 the
    bootstrap state n steps later, done=1 iff that window ended in a real TERMINATION (no
    bootstrap), n the actual step count (for gamma**n)."""

    def __init__(self, cap, int_keys):
        self.cap = cap
        self.int_keys = set(int_keys)
        self.size = 0
        self.ptr = 0
        self.s = [None] * cap
        self.s2 = [None] * cap
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.done = np.zeros(cap, np.float32)
        self.n = np.ones(cap, np.float32)

    def push(self, s, a, r, s2, d, n):
        i = self.ptr
        self.s[i] = s; self.s2[i] = s2; self.a[i] = a; self.r[i] = r; self.done[i] = d; self.n[i] = n
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def _batch(self, buf, idx, device):
        keys = buf[idx[0]].keys()
        return {k: torch.as_tensor(np.stack([buf[i][k] for i in idx]),
                                   dtype=(torch.long if k in self.int_keys else torch.float32),
                                   device=device) for k in keys}

    def sample(self, n, device):
        idx = np.random.randint(0, self.size, n)
        s = self._batch(self.s, idx, device)
        s2 = self._batch(self.s2, idx, device)
        a = torch.as_tensor(self.a[idx], dtype=torch.long, device=device)
        r = torch.as_tensor(self.r[idx], dtype=torch.float32, device=device)
        d = torch.as_tensor(self.done[idx], dtype=torch.float32, device=device)
        nn_ = torch.as_tensor(self.n[idx], dtype=torch.float32, device=device)
        return s, a, r, s2, d, nn_


def _split(obs_np, e):
    return {k: obs_np[k][e].copy() for k in obs_np}


def dueling_q(net, obs):
    """Q(s,a) = tanh(V(s)) + A(s,a) - mean_{legal} A,  illegal -> -1e9.

    Built from logits_value (logits already masked to -1e9 for illegal). For legal options the
    masked logits ARE the raw advantage A; V=value_head anchors the +/-1 scale. argmax is unchanged."""
    logits, value = net.logits_value(obs)            # logits [B,A] (illegal=-1e9), value [B]
    legal = logits > -1e8
    adv = logits.masked_fill(~legal, 0.0)
    mean_adv = adv.sum(1) / legal.sum(1).clamp(min=1)
    q = torch.tanh(value).unsqueeze(1) + logits - mean_adv.unsqueeze(1)
    return q.masked_fill(~legal, -1e9)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--decks", default="mega_abomasnow")
    p.add_argument("--num-envs", type=int, default=96)
    p.add_argument("--total-timesteps", type=int, default=20_000_000)
    p.add_argument("--n-step", type=int, default=8, help="n-step return horizon")
    p.add_argument("--shaping", default="prize_diff", help="dense reward shaping ('none' to disable)")
    p.add_argument("--buffer", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--grad-steps", type=int, default=4, help="grad steps per env-step batch (replay reuse)")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--learn-start", type=int, default=20000)
    p.add_argument("--target-update", type=int, default=1500, help="GRAD-steps between target copies")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-frac", type=float, default=0.15, help="anneal eps over this frac of training")
    p.add_argument("--snapshot-every", type=int, default=100000)
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--emb-dim", type=int, default=48)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=os.path.join(os.environ.get("HOME", "."), "pkmn_runs", "dqn"))
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(f"[cfg] {vars(a)}", flush=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    device = torch.device(a.device)
    g = a.gamma

    ct = get_card_table(); enc = TokenEncoder(ct)
    net_config = {"arch": "transformer2", "emb_dim": a.emb_dim, "d_model": a.d_model,
                  "nhead": a.nhead, "nlayers": a.nlayers, "ff": a.ff}
    pool = resolve_deck_pool(a.decks)
    print(f"[decks] {a.decks} -> {len(pool)} deck(s)", flush=True)
    env_kwargs = {"agent_decks": pool, "opponent_decks": pool, "randomize_side": True,
                  "shaping": a.shaping}
    envs = SubprocVecEnv(a.num_envs, env_kwargs, net_config, base_seed=a.seed * 1000,
                         server_device=device, opponent_mode="server")

    net = build_token_net(ct, net_config).to(device)
    target = build_token_net(ct, net_config).to(device)
    target.load_state_dict(net.state_dict()); target.eval()
    opt = optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    print(f"[net] params={sum(p.numel() for p in net.parameters()):,}", flush=True)

    buf = Replay(a.buffer, enc.int_keys)
    snap_pool = collections.deque(maxlen=a.pool_size)
    snap_pool.append({k: v.cpu() for k, v in net.state_dict().items()})
    envs.set_opponent(snap_pool[-1])

    nbuf = [collections.deque() for _ in range(a.num_envs)]   # per-env pending (s,a,r) for n-step
    next_obs_np, _ = envs.reset()
    global_step = 0; grad_step = 0; start = time.time(); last_target = 0; last_snap = 0
    losses = collections.deque(maxlen=500); ep_ret = collections.deque(maxlen=400)
    qmag = collections.deque(maxlen=500)
    anneal = max(1, int(a.total_timesteps * a.eps_frac))

    def flush_or_emit(e, next_s_e, done_e, trunc_e):
        dq = nbuf[e]
        if done_e:                       # episode ended -> emit ALL pending as terminal (no bootstrap)
            items = list(dq); dq.clear()
            m = len(items)
            for i in range(m):
                R = 0.0
                for k in range(i, m):
                    R += (g ** (k - i)) * items[k][2]
                buf.push(items[i][0], items[i][1], R, next_s_e, 1.0, m - i)
        elif len(dq) == a.n_step:        # window full -> emit oldest with bootstrap from next_s_e
            R = 0.0
            for k in range(a.n_step):
                R += (g ** k) * dq[k][2]
            buf.push(dq[0][0], dq[0][1], R, next_s_e, 0.0, a.n_step)
            dq.popleft()

    while global_step < a.total_timesteps:
        eps = max(a.eps_end, a.eps_start - (a.eps_start - a.eps_end) * global_step / anneal)

        with torch.no_grad():
            q = dueling_q(net, obs_to_tensors(next_obs_np, device))
        greedy = q.argmax(1).cpu().numpy()
        qmag.append(float(q.masked_fill(q < -1e8, float("nan")).abs().nanmean()))
        mask = next_obs_np["action_mask"]
        acts = np.empty(a.num_envs, np.int64)
        for e in range(a.num_envs):
            if random.random() < eps:
                legal = np.flatnonzero(mask[e] > 0.5)
                acts[e] = np.random.choice(legal) if legal.size else greedy[e]
            else:
                acts[e] = greedy[e]

        s_list = [_split(next_obs_np, e) for e in range(a.num_envs)]
        next_obs_np, reward, done, infos = envs.step(acts)
        next_s_list = [_split(next_obs_np, e) for e in range(a.num_envs)]
        for e in range(a.num_envs):
            trunc_e = bool(infos[e].get("truncated", False))
            nbuf[e].append((s_list[e], int(acts[e]), float(reward[e])))
            flush_or_emit(e, next_s_list[e], bool(done[e]), trunc_e)
            if done[e] and "terminal_reward" in infos[e]:
                ep_ret.append(infos[e]["terminal_reward"])
        global_step += a.num_envs

        # --- learn (n-step Double Dueling DQN) ---
        if buf.size >= a.learn_start:
            for _ in range(a.grad_steps):
                s, act, r, s2, dn, nn_ = buf.sample(a.batch_size, device)
                with torch.no_grad():
                    a2 = dueling_q(net, s2).argmax(1)                       # online selects a'
                    q2 = dueling_q(target, s2).gather(1, a2[:, None]).squeeze(1)  # target evaluates
                    tgt = r + torch.pow(torch.tensor(g, device=device), nn_) * (1.0 - dn) * q2
                qa = dueling_q(net, s).gather(1, act[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(qa, tgt)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
                losses.append(float(loss)); grad_step += 1
                if grad_step - last_target >= a.target_update:
                    target.load_state_dict(net.state_dict()); last_target = grad_step

        if global_step - last_snap >= a.snapshot_every:
            snap_pool.append({k: v.cpu() for k, v in net.state_dict().items()})
            envs.set_opponent(snap_pool[np.random.randint(len(snap_pool))]); last_snap = global_step

        if global_step % (a.num_envs * 200) < a.num_envs:
            sps = global_step / max(time.time() - start, 1e-9)
            wr = np.mean([1.0 if z > 0 else 0.0 for z in ep_ret]) if ep_ret else 0.0
            print(f"step={global_step} eps={eps:.3f} loss={np.mean(losses) if losses else 0:.4f} "
                  f"|Q|={np.mean(qmag) if qmag else 0:.3f} winrate={wr:.3f} sps={sps:.0f} "
                  f"buf={buf.size} grad={grad_step}", flush=True)
        if global_step % a.save_every < a.num_envs:
            torch.save({"net": net.state_dict(), "net_config": net_config, "global_step": global_step},
                       os.path.join(a.out, "latest.pt"))

    torch.save({"net": net.state_dict(), "net_config": net_config, "global_step": global_step},
               os.path.join(a.out, "latest.pt"))
    print("done.", flush=True)


if __name__ == "__main__":
    main()
