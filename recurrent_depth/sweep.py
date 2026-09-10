"""One training run of one recurrent-depth arm, shared by both sweeps.

The only difference between the two experiments is the task the model is trained
on ("inject", a one-fact lookup, or "long", associative recall over K inline
facts) and how often we stop to evaluate. Everything else, the model, the
optimiser, the accuracy-threshold bookkeeping, is identical, so it lives here.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .diagnostics import eval_diagnostics
from .model import ARM_SPECS, CoreConfig, make_recurrent_model
from . import tasks
from .util import cosine_lr, count_params, pick_device, seed_all, timer, write_result


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
            schedule=None, target_acc: float = 0.95, quiet: bool = False) -> dict:
    """Train one arm on one task and return (and save) a result dict.

    arm    one of ARM_SPECS ('baseline_4', 'rd_1x4', 'rd_2x2', 'rd_4x2', 'rd_1x8')
    task   'inject' (one-fact lookup) or 'long' (associative recall over K facts)
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
                     d_conv=4, expand=2, headdim=64, block="auto")
    model = make_recurrent_model(task_cfg.vocab_size, device, cfg,
                                 spec["n_distinct"], spec["applies_per_block"])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01,
                            betas=(0.9, 0.95))

    train_gen = torch.Generator(device="cpu").manual_seed(seed)
    eval_gen = torch.Generator(device="cpu").manual_seed(seed + 10_000)

    def sample(arm, batch, gen, device):          # arm is ignored; task is fixed
        return tasks.sample_batch(task_cfg, task, batch, gen, device=device)

    seq0, _, _ = sample(task, 2, train_gen, device)
    seq_len = seq0.shape[1]
    n_params = count_params(model)
    total_applies = spec["n_distinct"] * spec["applies_per_block"]

    if schedule is None:
        schedule = coarse_schedule(steps) if task == "long" else fine_schedule(steps)
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
            logits = model(seq)
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

    payload = {
        "tag": tag, "arm": arm, "task": task, "k_facts": k, "seed": seed,
        "batch": batch, "seq_len": seq_len, "steps": steps, "params": n_params,
        "vocab_size": task_cfg.vocab_size,
        "n_distinct": spec["n_distinct"], "applies_per_block": spec["applies_per_block"],
        "total_applies": total_applies,
        "core": {"d_model": cfg.d_model, "d_state": cfg.d_state,
                 "expand": cfg.expand, "headdim": cfg.headdim, "block": cfg.block},
        "device": device,
        "tokens_processed": tokens_processed,
        "step_to_95": hit[95], "step_to_99": hit[99], "step_to_100": hit[100],
        "tokens_to_95": None if hit[95] is None else hit[95] * batch * seq_len,
        "target_acc": target_acc,
        "final_acc": curve[-1]["acc"] if curve else 0.0,
        "peak_acc": max((d["acc"] for d in curve), default=0.0),
        "wallclock_s": round(t["elapsed_s"], 3),
        "curve": curve,
    }
    name = f"{tag}__{arm}__s{seed}"
    path = write_result(name, payload)
    if not quiet:
        print(f"  -> {path.name}  step_to_95={hit[95]}  "
              f"final_acc={payload['final_acc']:.3f}  wall={t['elapsed_s']:.1f}s")
    return payload
