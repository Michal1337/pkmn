"""Integration test: SubprocVecEnv with the central inference server (opponent).
Confirms games run + complete via the server, no hang/deadlock. Guarded for spawn."""
import logging; logging.disable(logging.CRITICAL)
import multiprocessing as mp
import time


def main():
    import numpy as np
    from rl.card_features import get_card_table
    from rl.encoding import Encoder
    from rl.policy import build_net
    from rl.env import load_deck
    from rl.vec_env import SubprocVecEnv
    enc = Encoder(get_card_table())
    net = build_net(enc.cf, enc.cards.vocab_size, {"emb_dim": 32}); net.eval()
    deck = load_deck()
    n = 4
    env_kwargs = {"agent_decks": [deck], "opponent_decks": [deck]}
    envs = SubprocVecEnv(n, env_kwargs, {"arch": "mlp", "emb_dim": 32},
                         base_seed=0, server_device="cpu")
    obs, info = envs.reset()
    envs.set_opponent({k: v.cpu() for k, v in net.state_dict().items()})   # opponent via server
    t0 = time.time(); dones = 0
    for _ in range(300):
        actions = [int(np.argmax(obs["action_mask"][i])) for i in range(n)]   # first legal action
        obs, rews, done, infos = envs.step(actions)
        dones += int(done.sum())
    dt = time.time() - t0
    print(f"[vec+server] 300 steps x{n} envs in {dt:.1f}s ({300*n/dt:.0f} env-steps/s), "
          f"episodes completed={dones} -- opponent ran via server, no hang")
    envs.close(); print("closed OK")


if __name__ == "__main__":
    mp.freeze_support()
    main()
