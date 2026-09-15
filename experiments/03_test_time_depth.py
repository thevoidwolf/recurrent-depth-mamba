"""Experiment 3 - test-time depth scaling and loop stability.

Experiments 1 and 2 measured *training* convergence speed (step_to_95). They
never showed the axis the recurrent-depth literature actually cares about: does
accuracy scale as you add loop iterations at *test* time, and does the residual
stream stay bounded while it does?

This script trains matched pre-norm (the repo's original) and post-norm+inject
(stabilised) looped models and compares three things:

  - depth_curve     accuracy vs test-time apply count r  (rise vs rise-then-collapse)
  - residual trace  residual-stream RMS across the unrolled loop (bounded vs growing)
  - step_to_95      training convergence (unchanged bookkeeping)

The hypothesis (2602.12078 / STARS 2605.26733): the original pre-norm loop grows
the residual stream and collapses at higher r; post-norm + input feedback
keeps it bounded and lets accuracy hold (or rise) as r increases.

    python experiments/03_test_time_depth.py --smoke                     # tiny fallback A/B
    python experiments/03_test_time_depth.py --ab --arm rd_1x4 --task long --steps 8000
    python experiments/03_test_time_depth.py --arm rd_1x4 --task inject --steps 400 --post
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recurrent_depth.model import ARM_SPECS
from recurrent_depth.sweep import run_arm

TAG = "03_test_time_depth"


def _summary(p) -> str:
    dc = "  ".join(f"r={d['applies']}:{d['acc']:.2f}" for d in p["depth_curve"])
    nt = p["residual_norm_trace"]
    verdict = "bounded" if nt[-1] <= 3 * nt[0] else "GROWING"
    return (f"    step_to_95={p['step_to_95']}  "
            f"peak@r={p['depth_peak_applies']} acc={p['depth_peak_acc']:.3f}\n"
            f"    depth:     {dc}\n"
            f"    resid-RMS: {nt[0]:.2f} -> {nt[-1]:.2f} ({verdict})")


def run_one(*, arm, task, steps, seed, batch, block, post, k, depth_eval,
            random_depth=False, random_depth_range=None):
    kw = dict(arm=arm, task=task, tag=TAG, steps=steps, seed=seed, batch=batch,
              k=k, block=block, quiet=True, depth_eval=depth_eval,
              random_depth=random_depth, random_depth_range=random_depth_range)
    if post:
        kw.update(norm_position="post", inject_input=True)
    return run_arm(**kw)


def _print_aggregate(rows, labels=("pre", "post")) -> None:
    """The statistic the `long`-task question turns on: how many seeds each arm
    actually drove to 95% (step_to_95 is None => never got there)."""
    a, b = labels
    n = len(rows)
    print(f"\n=== aggregate over {n} seeds ===")
    print(f"  seed | {a:>6} step95   peak | {b:>6} step95   peak")
    a_ok = b_ok = 0
    for sd, ra, rb in rows:
        a_ok += ra["step_to_95"] is not None
        b_ok += rb["step_to_95"] is not None
        print("  %4d | %11s  %.3f | %11s  %.3f" % (
            sd, ra["step_to_95"], ra["peak_acc"],
            rb["step_to_95"], rb["peak_acc"]))
    print(f"  reached 95%:  {a} {a_ok}/{n}   {b} {b_ok}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_SPECS), default="rd_1x4")
    ap.add_argument("--task", choices=["inject", "long"], default="inject")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="run several seeds (overrides --seed); prints an aggregate")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--post", action="store_true", help="post-norm + input feedback (re-add the embedded input each loop)")
    ap.add_argument("--ab", action="store_true", help="run pre-norm vs post-norm and compare")
    ap.add_argument("--rd-ablation", action="store_true",
                    help="isolate randomized-depth training: post+inject FIXED depth vs "
                         "post+inject RANDOM depth (the test-time-scaling experiment)")
    ap.add_argument("--random-depth", action="store_true",
                    help="sample the per-block apply count each training step")
    ap.add_argument("--rd-range", type=int, nargs=2, default=[1, 8],
                    metavar=("LO", "HI"), help="range for --random-depth / --rd-ablation")
    ap.add_argument("--smoke", action="store_true", help="tiny fallback A/B, no GPU needed")
    ap.add_argument("--depth-eval", type=int, nargs="*", default=None,
                    help="test-time apply counts to sweep (default 1..32)")
    args = ap.parse_args()

    if args.smoke:
        args.ab, args.steps, args.arm, block = True, 40, "rd_1x4", "fallback"
        depth_eval = args.depth_eval or [1, 2, 4, 8]
    else:
        block = "auto"
        depth_eval = args.depth_eval or [1, 2, 4, 8, 16, 32]

    seeds = args.seeds if args.seeds else [args.seed]
    rd_range = tuple(args.rd_range)
    common = dict(arm=args.arm, task=args.task, steps=args.steps,
                  batch=args.batch, k=args.k, block=block, depth_eval=depth_eval)

    if args.rd_ablation:
        # Isolate the training-depth schedule: both arms are post-norm+inject;
        # one trained at fixed depth, one at randomized depth over rd_range. This
        # is the test-time-scaling experiment -- does variable-depth training let
        # accuracy hold as test-time r grows past the nominal depth?
        rows = []
        for sd in seeds:
            print(f"[rd-ablation] arm={args.arm} task={args.task} steps={args.steps} "
                  f"seed={sd} rd_range={rd_range} block={block}", flush=True)
            fixed = run_one(post=True, seed=sd, **common)
            print("  post+inject, FIXED depth:")
            print(_summary(fixed), flush=True)
            rand = run_one(post=True, seed=sd, random_depth=True,
                           random_depth_range=rd_range, **common)
            print(f"  post+inject, RANDOM depth {rd_range}:")
            print(_summary(rand), flush=True)
            rows.append((sd, fixed, rand))
        if len(rows) > 1:
            _print_aggregate(rows, labels=("fixed", "random"))
        return

    if args.ab:
        rows = []
        for sd in seeds:
            print(f"[A/B] arm={args.arm} task={args.task} steps={args.steps} "
                  f"seed={sd} block={block}", flush=True)
            pre = run_one(post=False, seed=sd, **common)
            print("  pre-norm (original):")
            print(_summary(pre), flush=True)
            post = run_one(post=True, seed=sd, random_depth=args.random_depth,
                           random_depth_range=rd_range, **common)
            label = "post-norm + inject"
            if args.random_depth:
                label += f" + random-depth {rd_range}"
            print(f"  {label} (stabilised):")
            print(_summary(post), flush=True)
            rows.append((sd, pre, post))
        if len(rows) > 1:
            _print_aggregate(rows)
        return

    for sd in seeds:
        p = run_one(post=args.post, seed=sd, random_depth=args.random_depth,
                    random_depth_range=rd_range, **common)
        print(f"[{TAG}] arm={args.arm} task={args.task} seed={sd} "
              f"post={args.post} random_depth={args.random_depth}")
        print(_summary(p), flush=True)


if __name__ == "__main__":
    main()
