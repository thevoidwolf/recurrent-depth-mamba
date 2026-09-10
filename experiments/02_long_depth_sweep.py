"""Experiment 2 - recurrent depth on the long task (associative recall).

The inject task (experiment 1) is easy. This is the harder one: K facts are laid
out inline and the model has to stream them, then answer a query about one of
them. Baseline needs thousands of steps here, so it is a fairer test of whether
weight-shared depth holds up when the task is not trivial.

Three arms only: baseline_4 (reference), rd_1x4 (max sharing), rd_2x2 (the
interpolant: 2 blocks, each applied twice, ~half the parameters). Five seeds,
because this is where the seed-to-seed spread matters and where an honest read
needs more than two runs.

    python experiments/02_long_depth_sweep.py --smoke   # tiny, runs on CPU
    python experiments/02_long_depth_sweep.py --full    # the real sweep (GPU)
    python experiments/02_long_depth_sweep.py --arm rd_2x2 --seed 0 --steps 8000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recurrent_depth.model import ARM_SPECS
from recurrent_depth.sweep import run_arm

TAG = "02_long_depth"
ARMS = ("baseline_4", "rd_1x4", "rd_2x2")   # the three worth the long-arm budget


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_SPECS), default=None)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--smoke", action="store_true", help="tiny 2-arm CPU check")
    ap.add_argument("--full", action="store_true", help="3 arms x 5 seeds")
    args = ap.parse_args()

    if args.smoke:
        run_arm(arm="baseline_4", task="long", tag=TAG, steps=40, k=args.k, batch=args.batch)
        run_arm(arm="rd_2x2", task="long", tag=TAG, steps=40, k=args.k, batch=args.batch)
        return
    if args.full:
        for arm in ARMS:
            for seed in (0, 1, 2, 3, 4):
                run_arm(arm=arm, task="long", tag=TAG, steps=args.steps,
                        k=args.k, seed=seed, batch=args.batch)
        return
    if args.arm is None:
        ap.error("pass --arm, --smoke, or --full")
    run_arm(arm=args.arm, task="long", tag=TAG, steps=args.steps,
            k=args.k, seed=args.seed, batch=args.batch)


if __name__ == "__main__":
    main()
