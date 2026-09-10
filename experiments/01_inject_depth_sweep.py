"""Experiment 1 - recurrent depth on the inject task (one-fact lookup).

Question: if you take a 4-block Mamba and instead apply *one* block four times
(same forward-pass depth, a quarter of the parameters), how much slower is it to
learn? And does applying a block extra times ever buy anything on its own?

Five arms, described in recurrent_depth/model.py:ARM_SPECS. The headline compares
baseline_4 (4 distinct blocks) against rd_1x4 (1 block, applied 4x).

    python experiments/01_inject_depth_sweep.py --smoke   # tiny, runs on CPU
    python experiments/01_inject_depth_sweep.py --full    # the real sweep (GPU)
    python experiments/01_inject_depth_sweep.py --arm rd_1x4 --seed 0 --steps 300
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recurrent_depth.model import ARM_SPECS
from recurrent_depth.sweep import run_arm

TAG = "01_inject_depth"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_SPECS), default=None)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--smoke", action="store_true", help="tiny 2-arm CPU check")
    ap.add_argument("--full", action="store_true", help="all 5 arms x 3 seeds")
    args = ap.parse_args()

    if args.smoke:
        run_arm(arm="baseline_4", task="inject", tag=TAG, steps=40, k=args.k, batch=args.batch)
        run_arm(arm="rd_1x4", task="inject", tag=TAG, steps=40, k=args.k, batch=args.batch)
        return
    if args.full:
        for arm in ARM_SPECS:
            for seed in (0, 1, 2):
                run_arm(arm=arm, task="inject", tag=TAG, steps=args.steps,
                        k=args.k, seed=seed, batch=args.batch)
        return
    if args.arm is None:
        ap.error("pass --arm, --smoke, or --full")
    run_arm(arm=args.arm, task="inject", tag=TAG, steps=args.steps,
            k=args.k, seed=args.seed, batch=args.batch)


if __name__ == "__main__":
    main()
