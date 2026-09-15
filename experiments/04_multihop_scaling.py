"""Experiment 4 - the true test-time-scaling test on multi-hop chains.

Experiment 3 showed the stabilised looped model is depth-*robust*: randomized-depth
training holds accuracy flat at 1.00 for every test-time loop count r on K-way
associative recall. But that task is solvable in a single pass, so it structurally
cannot show depth-*scaling* -- "adding loops buys reasoning depth." A flat line at
1.00 is robustness, not scaling.

This script runs the experiment that can. It trains the same stabilised looped
model (post-norm + input re-injection + randomized-depth, the recipe from exp 3)
on multi-hop retrieval chains, then sweeps test-time depth r and looks for
accuracy that *rises* with r.

The lever is the fact-bank layout (see recurrent_depth/tasks.py):

  favorable    banks ordered so one pass down the sequence composes every link
               (bankN..bank1 · Q). Control: should solve at low r.
  adversarial  banks ordered so a pass can chain at most one link before it needs
               a binding the next bank hasn't supplied yet (bank1..bankN · Q).
               The counting rule predicts ~n_hops passes -> accuracy should be low
               at r=1 and climb as r crosses the hop count.

So the signature of genuine test-time scaling is: favorable flat-and-high across r,
adversarial low-at-r=1 and rising -- and 3-hop needing more loops than 2-hop.

    python experiments/04_multihop_scaling.py --smoke                 # CPU wiring check
    python experiments/04_multihop_scaling.py --hops 2 --steps 6000   # 2-hop fav vs adv (GPU)
    python experiments/04_multihop_scaling.py --hops 3 --steps 12000  # 3-hop fav vs adv (GPU)
    python experiments/04_multihop_scaling.py --task long_3hop_adversarial --steps 12000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recurrent_depth.model import ARM_SPECS
from recurrent_depth.sweep import MULTIHOP_ARMS, run_arm

TAG = "04_multihop_scaling"

# favorable (control) vs adversarial (the scaling test) for each hop count.
HOP_PAIRS = {
    2: ("long_2hop", "long_2hop_swap"),
    3: ("long_3hop_favorable", "long_3hop_adversarial"),
}


def _threshold_r(depth_curve, thr=0.5):
    """Smallest test-time r whose accuracy first reaches `thr` -- the readout for
    "how many loops does this layout need." None if it never gets there."""
    for d in depth_curve:
        if d["acc"] >= thr:
            return d["applies"]
    return None


def _summary(p) -> str:
    dc = "  ".join(f"r={d['applies']}:{d['acc']:.2f}" for d in p["depth_curve"])
    nt = p["residual_norm_trace"]
    verdict = "bounded" if nt[-1] <= 3 * nt[0] else "GROWING"
    r50 = _threshold_r(p["depth_curve"], 0.5)
    r90 = _threshold_r(p["depth_curve"], 0.9)
    return (f"    step_to_95={p['step_to_95']}  peak_train_acc={p['peak_acc']:.3f}\n"
            f"    depth-sweep: {dc}\n"
            f"    loops-to-acc: r@0.5={r50}  r@0.9={r90}  "
            f"peak@r={p['depth_peak_applies']}({p['depth_peak_acc']:.2f})\n"
            f"    resid-RMS: {nt[0]:.2f} -> {nt[-1]:.2f} ({verdict})")


def run_one(*, task, arm, steps, seed, batch, k, block, depth_eval,
            post, random_depth, rd_range, quiet=True):
    kw = dict(arm=arm, task=task, tag=TAG, steps=steps, seed=seed, batch=batch,
              k=k, block=block, quiet=quiet, depth_eval=depth_eval,
              random_depth=random_depth,
              random_depth_range=rd_range if random_depth else None)
    if post:
        kw.update(norm_position="post", inject_input=True)
    return run_arm(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARM_SPECS), default="rd_1x8",
                    help="looped arm to train (default rd_1x8: one block, up to 8 loops)")
    ap.add_argument("--hops", type=int, choices=[2, 3], default=2,
                    help="run the favorable-vs-adversarial pair for this hop count")
    ap.add_argument("--task", choices=sorted(MULTIHOP_ARMS), default=None,
                    help="run a single named layout instead of the --hops pair")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--k", type=int, default=16,
                    help="facts per bank (smaller keeps the harder chains learnable)")
    ap.add_argument("--depth-eval", type=int, nargs="*", default=None,
                    help="test-time loop counts to sweep (default fine at low r)")
    # Stabilisers: default to the exp-3 depth-robust recipe. Flags turn them off.
    ap.add_argument("--no-post", action="store_true",
                    help="disable post-norm+inject (use the original pre-norm loop)")
    ap.add_argument("--no-random-depth", action="store_true",
                    help="train at fixed depth instead of randomized-depth")
    ap.add_argument("--rd-range", type=int, nargs=2, default=[1, 8],
                    metavar=("LO", "HI"))
    ap.add_argument("--smoke", action="store_true", help="tiny CPU wiring check")
    ap.add_argument("--verbose", action="store_true",
                    help="stream the per-eval training curve (for monitoring long runs)")
    args = ap.parse_args()

    if args.smoke:
        block, steps = "fallback", 40
        depth_eval = args.depth_eval or [1, 2, 3, 4]
        k = min(args.k, 6)
    else:
        block, steps = "auto", args.steps
        depth_eval = args.depth_eval or [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
        k = args.k

    if args.task is not None:
        tasks_to_run = [args.task]
    else:
        tasks_to_run = list(HOP_PAIRS[args.hops])

    seeds = args.seeds if args.seeds else [args.seed]
    rd_range = tuple(args.rd_range)
    post = not args.no_post
    random_depth = not args.no_random_depth

    recipe = (f"arm={args.arm} k={k} steps={steps} "
              f"post+inject={post} random_depth={random_depth}"
              + (f"{rd_range}" if random_depth else "") + f" block={block}")
    print(f"[{TAG}] {recipe}", flush=True)
    print(f"  layouts: {tasks_to_run}   test-time r sweep: {depth_eval}\n", flush=True)

    results = {}
    for task in tasks_to_run:
        for sd in seeds:
            print(f"--- {task}  seed={sd} ---", flush=True)
            p = run_one(task=task, arm=args.arm, steps=steps, seed=sd,
                        batch=args.batch, k=k, block=block, depth_eval=depth_eval,
                        post=post, random_depth=random_depth, rd_range=rd_range,
                        quiet=not args.verbose)
            print(_summary(p), flush=True)
            results.setdefault(task, []).append(p)

    # Contrast readout: favorable should solve at low r; adversarial should need
    # more loops (and 3-hop more than 2-hop). This is the scaling fingerprint.
    if args.task is None and len(tasks_to_run) == 2:
        fav, adv = tasks_to_run
        print(f"\n=== scaling contrast ({args.hops}-hop) ===")
        print(f"  {'layout':<22} r@0.5  r@0.9  acc(r=1)  peak_acc")
        for task in (fav, adv):
            for p in results[task]:
                r1 = next((d["acc"] for d in p["depth_curve"] if d["applies"] == 1), float("nan"))
                print("  %-22s %5s  %5s  %8.2f  %7.2f" % (
                    task, _threshold_r(p["depth_curve"], 0.5),
                    _threshold_r(p["depth_curve"], 0.9), r1, p["depth_peak_acc"]))
        print("\n  scaling signal = adversarial rises with r while favorable is "
              "flat-high;\n  a genuine hop->loop law shows r@0.9(adv) > r@0.9(fav).")


if __name__ == "__main__":
    main()
