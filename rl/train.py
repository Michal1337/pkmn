"""CleanRL-style PPO for the cabt env, with action masking and self-play.

Curriculum: train vs the random opponent until ``--selfplay-start`` env steps,
then switch the in-worker opponent to frozen snapshots of the learner (a small
pool, refreshed every ``--snapshot-every`` steps).

Run:
    python -m rl.train --total-timesteps 5000000 --num-envs 16 --device cuda
The engine is CPU-bound and one-battle-per-process, so throughput scales with
--num-envs (subprocess workers); the net is tiny and barely uses the GPU.
"""

from __future__ import annotations

import argparse
import collections
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .card_features import get_card_table
from .decks import DECKS
try:
    from .decks_generated import GENERATED      # auto-generated archetype decks
except Exception:
    GENERATED = {}
from .encoding import Encoder
from .env import load_deck
from .policy import build_net, load_compatible, obs_to_tensors
from .vec_env import SubprocVecEnv


def resolve_deck_pool(name: str) -> list[list[int]]:
    """Map a --decks value to a list of decks (each side is sampled from this)."""
    sample = load_deck()  # engine sample deck (agent/deck.csv)
    if name == "all":
        return list(DECKS.values()) + [sample]
    if name == "gen":                                  # 50 generated archetypes
        return list(GENERATED.values()) or [sample]
    if name == "all+gen":                              # official + sample + generated
        return list(DECKS.values()) + [sample] + list(GENERATED.values())
    if name == "official":
        return list(DECKS.values())
    if name == "sample":
        return [sample]
    if name in DECKS:
        return [DECKS[name]]
    if name in GENERATED:
        return [GENERATED[name]]
    raise SystemExit(f"unknown --decks '{name}' (all|gen|all+gen|official|sample|<name>)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=5_000_000)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--norm-adv", action="store_true", default=True)
    # self-play
    p.add_argument("--selfplay-start", type=int, default=500_000,
                   help="env steps of random-opponent warmup before self-play")
    p.add_argument("--snapshot-every", type=int, default=100_000)
    p.add_argument("--pool-size", type=int, default=5)
    # env / data
    p.add_argument("--shaping", choices=["none", "prize_diff"], default="none")
    p.add_argument("--decks", type=str, default="all",
                   help="deck pool: 'all' (4 official + sample), 'official' (4), 'sample', "
                        "or a named deck from rl.decks (e.g. mega_abomasnow)")
    p.add_argument("--no-randomize-side", action="store_true")
    # net (architecture)
    p.add_argument("--arch", choices=["mlp", "transformer"], default="mlp")
    p.add_argument("--emb-dim", type=int, default=32)
    # mlp dims
    p.add_argument("--card-h", type=int, default=64)
    p.add_argument("--trunk-h", type=int, default=256)
    p.add_argument("--opt-h", type=int, default=96)
    # transformer dims
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--init-from", type=str, default=None,
                   help="checkpoint .pt to load net weights from (resume / start self-play from)")
    # infra
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=str, default=os.path.join(os.environ.get("HOME", "."), "pkmn_runs", "ppo"))
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    os.makedirs(args.out, exist_ok=True)
    print(f"[cfg] {vars(args)}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ct = get_card_table()
    enc = Encoder(ct)
    if args.arch == "transformer":
        net_config = {"arch": "transformer", "emb_dim": args.emb_dim, "d_model": args.d_model,
                      "nhead": args.nhead, "nlayers": args.nlayers, "ff": args.ff}
    else:
        net_config = {"arch": "mlp", "emb_dim": args.emb_dim, "card_h": args.card_h,
                      "trunk_h": args.trunk_h, "opt_h": args.opt_h}

    pool = resolve_deck_pool(args.decks)
    print(f"[decks] pool='{args.decks}' -> {len(pool)} deck(s); both sides sampled per episode", flush=True)
    env_kwargs = {
        "agent_decks": pool,
        "opponent_decks": pool,
        "randomize_side": not args.no_randomize_side,
        "shaping": args.shaping,
    }
    envs = SubprocVecEnv(args.num_envs, env_kwargs, net_config, base_seed=args.seed * 1000)

    net = build_net(enc.cf, ct.vocab_size, net_config).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    print(f"[net] params={sum(p.numel() for p in net.parameters()):,}", flush=True)

    if args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        skipped = load_compatible(net, ck["net"])
        msg = f" (reinit {len(skipped)} params: {skipped})" if skipped else ""
        print(f"[init] loaded {args.init_from} (was trained to step {ck.get('global_step')}){msg}", flush=True)

    shapes = enc.shapes
    # rollout storage (obs kept on device)
    obs_buf = {}
    for k, sh in shapes.items():
        dt = torch.long if k in enc.int_keys else torch.float32
        obs_buf[k] = torch.zeros((args.num_steps, args.num_envs, *sh), dtype=dt, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start = time.time()
    next_obs_np, _ = envs.reset()
    next_obs = obs_to_tensors(next_obs_np, device)
    next_done = torch.zeros(args.num_envs, device=device)

    snapshot_pool = collections.deque(maxlen=args.pool_size)
    selfplay_on = False
    last_snapshot = 0
    ep_returns = collections.deque(maxlen=200)  # terminal rewards (~win signal)

    # start self-play immediately (e.g. resuming a model that already beats random)
    if args.selfplay_start <= 0:
        selfplay_on = True
        snapshot_pool.append({k: v.cpu() for k, v in net.state_dict().items()})
        envs.set_opponent(snapshot_pool[-1])
        print("[selfplay] enabled at start (selfplay_start<=0)", flush=True)

    num_iters = args.total_timesteps // args.batch_size
    for it in range(1, num_iters + 1):
        if args.anneal_lr:
            frac = 1.0 - (it - 1.0) / num_iters
            opt.param_groups[0]["lr"] = frac * args.lr

        for step in range(args.num_steps):
            global_step += args.num_envs
            for k in shapes:
                obs_buf[k][step] = next_obs[k]
            dones[step] = next_done
            with torch.no_grad():
                action, logp, _, value = net.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logp
            values[step] = value

            next_obs_np, reward, done, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.as_tensor(reward, device=device)
            next_obs = obs_to_tensors(next_obs_np, device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
            for d, info in zip(done, infos):
                if d and "terminal_reward" in info:
                    ep_returns.append(info["terminal_reward"])

        # ---- self-play scheduling ----
        if not selfplay_on and global_step >= args.selfplay_start:
            selfplay_on = True
            snapshot_pool.append({k: v.cpu() for k, v in net.state_dict().items()})
            envs.set_opponent(snapshot_pool[-1])
            last_snapshot = global_step
            print(f"[selfplay] enabled at step {global_step}", flush=True)
        if selfplay_on and global_step - last_snapshot >= args.snapshot_every:
            snapshot_pool.append({k: v.cpu() for k, v in net.state_dict().items()})
            pick = snapshot_pool[np.random.randint(len(snapshot_pool))]
            envs.set_opponent(pick)
            last_snapshot = global_step

        # ---- GAE ----
        with torch.no_grad():
            next_value = net.get_value(next_obs)
            advantages = torch.zeros_like(rewards)
            lastgae = 0
            for t in reversed(range(args.num_steps)):
                nonterminal = 1.0 - (next_done if t == args.num_steps - 1 else dones[t + 1])
                nextval = next_value if t == args.num_steps - 1 else values[t + 1]
                delta = rewards[t] + args.gamma * nextval * nonterminal - values[t]
                advantages[t] = lastgae = delta + args.gamma * args.gae_lambda * nonterminal * lastgae
            returns = advantages + values

        # ---- flatten ----
        b_obs = {k: obs_buf[k].reshape((-1, *shapes[k])) for k in shapes}
        b_logp = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            for s in range(0, args.batch_size, args.minibatch_size):
                mb = inds[s:s + args.minibatch_size]
                mb_obs = {k: b_obs[k][mb] for k in shapes}
                _, newlogp, entropy, newval = net.get_action_and_value(mb_obs, b_actions[mb])
                ratio = (newlogp - b_logp[mb]).exp()
                with torch.no_grad():
                    clipfracs.append(((ratio - 1.0).abs() > args.clip_coef).float().mean().item())
                adv = b_adv[mb]
                if args.norm_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_loss = 0.5 * ((newval - b_returns[mb]) ** 2).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * ent_loss + args.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm)
                opt.step()

        # ---- logging ----
        if it % args.log_every == 0:
            sps = int(global_step / (time.time() - start))
            wr = float(np.mean([r > 0 for r in ep_returns])) if ep_returns else float("nan")
            mret = float(np.mean(ep_returns)) if ep_returns else float("nan")
            print(f"step={global_step} it={it}/{num_iters} sps={sps} "
                  f"winrate={wr:.3f} ep_ret={mret:+.3f} "
                  f"pg={pg_loss.item():.3f} v={v_loss.item():.3f} ent={ent_loss.item():.3f} "
                  f"clip={np.mean(clipfracs):.3f} selfplay={selfplay_on}", flush=True)

        # ---- checkpoint ----
        if global_step % args.save_every < args.batch_size or it == num_iters:
            path = os.path.join(args.out, f"ckpt_{global_step}.pt")
            torch.save({"net": net.state_dict(), "args": vars(args),
                        "net_config": net_config, "global_step": global_step}, path)
            torch.save({"net": net.state_dict(), "args": vars(args),
                        "net_config": net_config, "global_step": global_step},
                       os.path.join(args.out, "latest.pt"))
            print(f"[ckpt] saved {path}", flush=True)

    envs.close()
    print("done.", flush=True)


if __name__ == "__main__":
    main()
