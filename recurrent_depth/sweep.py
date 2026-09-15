"""One training run of one recurrent-depth arm, shared by both sweeps.

The only difference between the two experiments is the task the model is trained
on ("inject", a one-fact lookup, or "long", associative recall over K inline
facts) and how often we stop to evaluate. Everything else, the model, the
optimiser, the accuracy-threshold bookkeeping, is identical, so it lives here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .diagnostics import depth_sweep, eval_diagnostics, residual_norm_trace
from .model import ARM_SPECS, CoreConfig, make_recurrent_model
from . import tasks
from .util import cosine_lr, count_params, pick_device, seed_all, timer, write_result


# Task/layout names routed to the multi-hop samplers. Anything not listed here
# falls through to tasks.sample_batch ('inject' / 'long').
_TWO_HOP_ARMS = frozenset({
    "long_2hop", "long_2hop_swap", "long_2hop_interleave",
    "inject_2hop", "inject_collapsed",
})
_THREE_HOP_ARMS = frozenset({"long_3hop_favorable", "long_3hop_adversarial"})
MULTIHOP_ARMS = _TWO_HOP_ARMS | _THREE_HOP_ARMS


def fine_schedule(total: int):
    """Fine early, coarse late: inject converges in tens of steps."""
    steps = set(range(5, min(101, total + 1), 5))
    steps.update(range(125, total + 1, 25))
    steps.add(total)
    return sorted(s for s in steps if 1 <= s <= total)


def coarse_schedule(total: int, every: int = 100):
    steps = set(range(every, total + 1, every))
    steps.add(total)
    return sorted(s for s in steps if 1 <= s <= total)


def run_arm(*, arm: str, task: str, tag: str, steps: int, k: int = 32,
            seed: int = 0, batch: int = 64, warmup: int = 50,
            schedule=None, target_acc: float = 0.95, quiet: bool = False,
            norm_position: str = "pre", inject_input: bool = False,
            bptt_window: int | None = None, block: str = "auto",
            random_depth: bool = False, random_depth_range: tuple | None = None,
            depth_eval: list | None = None) -> dict:
    """Train one arm on one task and return (and save) a result dict.

    arm    one of ARM_SPECS ('baseline_4', 'rd_1x4', 'rd_2x2', 'rd_4x2', 'rd_1x8')
    task   'inject' (one-fact lookup) or 'long' (associative recall over K facts)

    Recurrent-depth stabilisers (all default to the repo's original behaviour):
      norm_position       'pre' (original) or 'post' (2602.12078 stabiliser)
      inject_input        re-add the embedded input before every apply (Huginn)
      bptt_window         backprop only through the last k applies (truncated BPTT)
      random_depth        sample the per-block apply count each step (Geiping/STARS)
      random_depth_range  (lo, hi) for random_depth; default (1, applies_per_block)
      depth_eval          list of test-time apply counts to sweep after training
                          (the accuracy-vs-r curve); default derived from the arm
    """
    device = pick_device()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_all(seed)

    spec = ARM_SPECS[arm]
    task_cfg = tasks.TaskCfg(k_facts_long=k, seed=seed)
    # n_layers only sets the mixer's layer_idx range; we set it to the number of
    # distinct blocks so each distinct block gets its own index.
    cfg = CoreConfig(d_model=256, n_layers=spec["n_distinct"], d_state=64,
                     d_conv=4, expand=2, headdim=64, block=block,
                     norm_position=norm_position, inject_input=inject_input,
                     bptt_window=bptt_window)
    model = make_recurrent_model(task_cfg.vocab_size, device, cfg,
                                 spec["n_distinct"], spec["applies_per_block"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                            betas=(0.9, 0.95))

    train_gen = torch.Generator(device="cpu").manual_seed(seed)
    eval_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)
    depth_gen = torch.Generator(device="cpu").manual_seed(seed + 20_000)

    # Resolve randomized-depth range: default 1..applies_per_block.
    rd_lo, rd_hi = random_depth_range or (1, spec["applies_per_block"])

    def sample(arm, batch, gen, device):          # `arm` carries the task/layout name
        # Dispatch on the task string so the same run_arm drives 1-hop
        # (inject/long), 2-hop, and 3-hop chains. eval/depth-sweep pass the
        # task name back in as `arm`, so a fixed `task` closure is enough.
        if arm in _TWO_HOP_ARMS:
            return tasks.sample_2hop_batch(task_cfg, arm, batch, gen, device=device)
        if arm in _THREE_HOP_ARMS:
            return tasks.sample_3hop_batch(task_cfg, arm, batch, gen, device=device)
        return tasks.sample_batch(task_cfg, arm, batch, gen, device=device)

    seq0, _, _ = sample(task, 2, train_gen, device)
    seq_len = seq0.shape[1]
    n_params = count_params(model)
    total_applies = spec["n_distinct"] * spec["applies_per_block"]

    if schedule is None:
        # 'inject' is the only trivial (tens-of-steps) task; long and every
        # multi-hop chain want the coarse, thousands-of-steps schedule.
        schedule = fine_schedule(steps) if task == "inject" else coarse_schedule(steps)
    schedule = set(schedule)

    if not quiet:
        print(f"[{tag}:{arm} task={task} seed={seed}] "
              f"n_distinct={spec['n_distinct']} applies={spec['applies_per_block']} "
              f"total_applies={total_applies} params={n_params:,} "
              f"seq_len={seq_len} steps={steps} device={device}")

    curve = []
    hit = {95: None, 99: None, 100: None}
    tokens_processed = 0

    with timer() as t:
        for step in range(steps):
            lr = cosine_lr(step, warmup=warmup, total=steps, base=3e-4, floor=3e-5)
            for pg in opt.param_groups:
                pg["lr"] = lr

            seq, ans_pos, target = sample(task, batch, train_gen, device)
            r = None
            if random_depth:
                r = int(torch.randint(rd_lo, rd_hi + 1, (1,), generator=depth_gen).item())
            logits = model(seq, applies=r)
            loss = F.cross_entropy(logits[:, ans_pos - 1], target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tokens_processed += batch * seq_len

            if (step + 1) in schedule:
                d = eval_diagnostics(model, sample, task, eval_gen, device=device)
                d["step"] = step + 1
                d["tokens"] = tokens_processed
                d["train_loss"] = float(loss.item())
                d["lr"] = lr
                curve.append(d)
                acc = d["acc"]
                for thr, frac in ((95, 0.95), (99, 0.99), (100, 1.0 - 1e-9)):
                    if hit[thr] is None and acc >= frac:
                        hit[thr] = step + 1
                if not quiet:
                    print(f"  step {step+1:>5d}  loss {loss.item():.3f}  "
                          f"acc {acc:.3f}  margin {d['logit_margin']:+.2f}")

    # ---- the two axes step_to_95 never showed: accuracy-vs-test-time-depth and
    #      residual-stream norm growth across the unrolled loop. ----
    if depth_eval is None:
        base = spec["applies_per_block"]
        depth_eval = sorted({1, 2, 4, 8, base, base * 2, base * 4})
    depth_curve = depth_sweep(model, sample, task, eval_gen, depth_eval,
                              batch=256, n_batches=2, device=device)
    norm_trace = residual_norm_trace(model, sample, task, eval_gen,
                                     applies=max(depth_eval), batch=64, device=device)
    depth_peak = max(depth_curve, key=lambda d: d["acc"], default=None)
    if not quiet:
        pts = "  ".join(f"r={d['applies']}:{d['acc']:.2f}" for d in depth_curve)
        print(f"  depth-sweep  {pts}")
        print(f"  resid-RMS    first={norm_trace[0]:.2f} last={norm_trace[-1]:.2f} "
              f"({'bounded' if norm_trace[-1] <= 3 * norm_trace[0] else 'GROWING'})")

    payload = {
        "tag": tag, "arm": arm, "task": task, "k_facts": k, "seed": seed,
        "batch": batch, "seq_len": seq_len, "steps": steps, "params": n_params,
        "vocab_size": task_cfg.vocab_size,
        "n_distinct": spec["n_distinct"], "applies_per_block": spec["applies_per_block"],
        "total_applies": total_applies,
        "core": {"d_model": cfg.d_model, "d_state": cfg.d_state,
                 "expand": cfg.expand, "headdim": cfg.headdim, "block": cfg.block},
        "config": {"norm_position": norm_position, "inject_input": inject_input,
                   "bptt_window": bptt_window, "random_depth": random_depth,
                   "random_depth_range": [rd_lo, rd_hi] if random_depth else None},
        "device": device,
        "tokens_processed": tokens_processed,
        "step_to_95": hit[95], "step_to_99": hit[99], "step_to_100": hit[100],
        "tokens_to_95": None if hit[95] is None else hit[95] * batch * seq_len,
        "target_acc": target_acc,
        "final_acc": curve[-1]["acc"] if curve else 0.0,
        "peak_acc": max((d["acc"] for d in curve), default=0.0),
        "depth_eval": depth_eval,
        "depth_curve": depth_curve,
        "depth_peak_acc": depth_peak["acc"] if depth_peak else 0.0,
        "depth_peak_applies": depth_peak["applies"] if depth_peak else None,
        "residual_norm_trace": norm_trace,
        "wallclock_s": round(t["elapsed_s"], 3),
        "curve": curve,
    }
    name = f"{tag}__{arm}__s{seed}"
    path = write_result(name, payload)
    if not quiet:
        print(f"  -> {path.name}  step_to_95={hit[95]}  "
              f"final_acc={payload['final_acc']:.3f}  wall={t['elapsed_s']:.1f}s")
    return payload
