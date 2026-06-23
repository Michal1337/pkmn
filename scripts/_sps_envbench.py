"""Collection-SPS bench: real snapshot opponent, sweep (opponent_mode, num_envs) at fixed CPUs.
Isolates env-step + encode + OPPONENT forward (local b=1/worker vs server batched-GPU). The PPO
learner is separate (batched, mode-independent). Usage: python _sps_envbench.py <local|server> <n>"""
import sys, time
import numpy as np


def main():
    from rl.vec_env import SubprocVecEnv
    from rl.policy2 import build_token_net
    from rl.card_features import get_card_table
    import rl.decks_generated as dg
    mode = sys.argv[1]
    n = int(sys.argv[2])
    nc = {"arch": "transformer2", "emb_dim": 48, "d_model": 128, "nhead": 4, "nlayers": 2, "static": True}
    decks = list(dg.GENERATED.values())
    ek = dict(agent_decks=decks, opponent_decks=decks, randomize_side=True)
    ve = SubprocVecEnv(n, ek, nc, base_seed=0, server_device="cuda", opponent_mode=mode)
    try:
        net = build_token_net(get_card_table(), dict(nc))
        ve.set_opponent(net.state_dict())             # real opponent -> forwards run
        obs, _ = ve.reset()
        for _ in range(100):                              # warmup
            obs, _, _, _ = ve.step([int(np.argmax(r)) for r in obs["action_mask"]])
        N = 10000                                         # long window -> self-averaging, stable SPS
        t = time.time()
        for i in range(N):
            obs, _, _, _ = ve.step([int(np.argmax(r)) for r in obs["action_mask"]])
            if i == N // 2:                               # mid-point split -> sanity on drift/variance
                tm = time.time()
        dt = time.time() - t
        h1 = (N // 2) * n / (tm - t); h2 = (N - N // 2) * n / (dt - (tm - t))
        print("RESULT mode=%-6s n_envs=%-4d %8.0f env-steps/s  %6.2f ms/vec-step  (halves %.0f/%.0f)"
              % (mode, n, N * n / dt, dt / N * 1000, h1, h2), flush=True)
    finally:
        ve.close()


if __name__ == "__main__":
    main()
