"""cProfile the TRAINING MAIN LOOP (collection vs learner-forward vs PPO update).
cProfile only sees the main process (SubprocVecEnv workers are separate) -- which is
exactly the suspected bottleneck. Run a handful of iterations with a huge selfplay-start
(random opponent = warmup regime) and no checkpointing.

    python scripts/prof_train.py --arch mlp --num-envs 64 --total-timesteps 25000 ...
"""
import cProfile
import io
import pstats
import sys

from rl import train

if __name__ == "__main__":
    sys.argv = ["train"] + sys.argv[1:]
    pr = cProfile.Profile()
    pr.enable()
    try:
        train.main()
    except SystemExit:
        pass
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22)
    print("\n==== PROFILE (top by self-time) ====")
    print(s.getvalue())
