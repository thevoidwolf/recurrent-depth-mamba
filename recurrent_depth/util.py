"""Small shared helpers for the sweeps: seeding, parameter counting, timing,
result writing, learning-rate schedule, and device selection.

These are deliberately tiny and dependency-light so the experiment scripts read
top to bottom without hunting through a framework.
"""
from __future__ import annotations

import json
import math
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


@contextmanager
def timer():
    t0 = time.perf_counter()
    obj = {"elapsed_s": None}
    yield obj
    obj["elapsed_s"] = time.perf_counter() - t0


def write_result(name: str, payload: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = RESULTS_DIR / f"{name}__{ts}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def cosine_lr(step, warmup, total, base, floor):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    if step >= total:
        return floor
    prog = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * prog))
