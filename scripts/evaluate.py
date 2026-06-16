"""Play many cabt games between two agents and report the win rate.

Plays each pairing on both sides (P0 and P1) to cancel out first-player
advantage. Use this to check whether a policy change actually helps.

Usage:
    python scripts/evaluate.py -n 50                          # main.py vs random
    python scripts/evaluate.py agent/main.py random -n 100
"""

import argparse
import os

from _common import AGENT_DIR, load_agent, make_env


def resolve(name: str):
    if name in ("random", "first"):
        from kaggle_environments.envs.cabt.cabt import agents

        return agents[name]
    return load_agent(name)


def play(env, a, b) -> int:
    """Run one game; return +1 if `a` (player 0) wins, -1 if loses, 0 draw."""
    env.reset()
    env.run([a, b])
    r0, r1 = (s.reward for s in env.steps[-1])
    return 1 if r0 > r1 else (-1 if r0 < r1 else 0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("p0", nargs="?", default=os.path.join(AGENT_DIR, "main.py"))
    p.add_argument("p1", nargs="?", default="random")
    p.add_argument("-n", "--games", type=int, default=20)
    args = p.parse_args()

    a, b = resolve(args.p0), resolve(args.p1)
    env = make_env()

    wins = losses = draws = 0
    for i in range(args.games):
        # Alternate sides each game.
        if i % 2 == 0:
            r = play(env, a, b)
        else:
            r = -play(env, b, a)
        wins += r == 1
        losses += r == -1
        draws += r == 0
        print(f"  game {i + 1}/{args.games}: W={wins} L={losses} D={draws}", end="\r")

    total = max(args.games, 1)
    print()
    print(f"{args.p0} vs {args.p1} over {args.games} games:")
    print(f"  wins={wins}  losses={losses}  draws={draws}  win_rate={wins / total:.1%}")


if __name__ == "__main__":
    main()
