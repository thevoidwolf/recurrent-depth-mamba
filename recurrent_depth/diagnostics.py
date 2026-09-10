"""Reusable eval diagnostics for the rigs.

Returns not just accuracy but the shape of the answer distribution — useful
for distinguishing "the circuit is installed" from "the loss is decreasing but
the answer is still guessed."
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_diagnostics(model, sample_fn, arm: str, gen, batch: int = 512,
                     n_batches: int = 4, device: str = "cuda") -> dict:
    """Run `n_batches` eval batches; return per-eval diagnostics dict.

    Fields:
      acc              — top-1 accuracy at answer position (fraction)
      loss             — mean CE at answer position (nats)
      logit_margin     — mean(top1_logit - top2_logit) at answer position
      correct_prob     — mean softmax probability on the correct token
      correct_prob_min — worst-case correct-token prob in the eval set (a
                         cheap tail indicator: if this is high the model is
                         answering confidently on every sample, not just on
                         average)
    """
    model.eval()
    n = 0
    correct = 0
    loss_sum = 0.0
    margin_sum = 0.0
    cprob_sum = 0.0
    cprob_min = 1.0

    for _ in range(n_batches):
        seq, ans_pos, target = sample_fn(arm=arm, batch=batch, gen=gen, device=device)
        logits = model(seq)                       # [B, L, V]
        step_logits = logits[:, ans_pos - 1]      # [B, V]

        pred = step_logits.argmax(-1)
        correct += (pred == target).sum().item()

        loss_sum += F.cross_entropy(step_logits, target, reduction="sum").item()

        top2 = step_logits.topk(2, dim=-1).values      # [B, 2]
        margin_sum += (top2[:, 0] - top2[:, 1]).sum().item()

        probs = step_logits.softmax(-1)
        cprob = probs.gather(1, target.unsqueeze(1)).squeeze(1)   # [B]
        cprob_sum += cprob.sum().item()
        cprob_min = min(cprob_min, cprob.min().item())

        n += batch

    model.train()
    return {
        "acc": correct / n,
        "loss": loss_sum / n,
        "logit_margin": margin_sum / n,
        "correct_prob": cprob_sum / n,
        "correct_prob_min": cprob_min,
    }
