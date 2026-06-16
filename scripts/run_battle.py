"""Run a single cabt battle and write an HTML replay.

Usage:
    python scripts/run_battle.py                       # our agent vs built-in random
    python scripts/run_battle.py agent/main.py random  # explicit
    python scripts/run_battle.py agent/main.py agent/main.py -o result.html
"""

import argparse
import os

from _common import AGENT_DIR, load_agent, make_env


def resolve(name: str):
    """Map a CLI arg to an agent: a file path, or a built-in name."""
    if name in ("random", "first"):
        from kaggle_environments.envs.cabt.cabt import agents

        return agents[name]
    return load_agent(name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("p0", nargs="?", default=os.path.join(AGENT_DIR, "main.py"))
    p.add_argument("p1", nargs="?", default="random")
    p.add_argument("-o", "--out", default="result.html")
    args = p.parse_args()

    env = make_env()
    env.run([resolve(args.p0), resolve(args.p1)])

    last = env.steps[-1]
    rewards = [s.reward for s in last]
    print(f"Result: P0={rewards[0]}  P1={rewards[1]}  (steps={len(env.steps)})")
    if rewards[0] == rewards[1]:
        print("Draw.")
    else:
        print(f"Winner: P{0 if rewards[0] > rewards[1] else 1}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(env.render(mode="html"))
    print(f"Replay written to {args.out}")


if __name__ == "__main__":
    main()
