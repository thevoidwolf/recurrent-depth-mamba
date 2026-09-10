"""The model: a tiny language model whose backbone is a stack of Mamba-2 blocks.

Architecture:

    tokens -> embedding -> [ RMSNorm -> Mamba2 -> +residual ] x n_layers
           -> final RMSNorm -> linear head -> logits

On a CUDA GPU the backbone uses the real `mamba_ssm.Mamba2` kernels. If CUDA is
not available it falls back to a small pure-PyTorch selective scan so the
pipeline still runs on a laptop CPU for smoke tests. The fallback is NOT the
same kernel and will not reproduce the reported numbers; it exists only so the
code is runnable without a GPU. The headline results require the real Mamba-2 kernel.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CoreConfig:
    d_model: int = 256
    n_layers: int = 4
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    block: str = "auto"   # "auto" -> mamba2 on CUDA else fallback; or force "mamba2"/"fallback"


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * x / rms


class FallbackMixer(nn.Module):
    """A small pure-PyTorch gated diagonal selective scan.

    Same [B, L, D] -> [B, L, D] contract as Mamba2, runs on CPU. It is a real
    (slow) recurrent mixer, enough to exercise the training pipeline, but it is
    not the Mamba-2 kernel and does not reproduce the reported numbers.
    """
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        d_inner = cfg.d_model * cfg.expand
        self.in_proj = nn.Linear(cfg.d_model, 2 * d_inner, bias=False)
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.A_log = nn.Parameter(torch.zeros(d_inner))
        self.out_proj = nn.Linear(d_inner, cfg.d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape
        xi, z = self.in_proj(x).chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(xi))
        decay = torch.exp(-dt * torch.exp(self.A_log))
        h = torch.zeros(B, xi.shape[-1], device=x.device, dtype=x.dtype)
        ys = []
        for t in range(L):
            h = decay[:, t] * h + (1 - decay[:, t]) * xi[:, t]
            ys.append(h)
        y = torch.stack(ys, dim=1) * torch.sigmoid(z)
        return self.out_proj(y)


def _build_mixer(cfg: CoreConfig, layer_idx: int) -> nn.Module:
    block = cfg.block
    if block == "auto":
        block = "mamba2" if torch.cuda.is_available() else "fallback"
    if block == "mamba2":
        from mamba_ssm import Mamba2
        return Mamba2(d_model=cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv,
                      expand=cfg.expand, headdim=cfg.headdim, layer_idx=layer_idx)
    if block == "fallback":
        return FallbackMixer(cfg)
    raise ValueError(f"unknown block {block!r} (use 'auto', 'mamba2', or 'fallback')")


class CoreBlock(nn.Module):
    """Pre-norm residual wrapper around one mixer."""
    def __init__(self, cfg: CoreConfig, layer_idx: int):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = _build_mixer(cfg, layer_idx)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class CoreStack(nn.Module):
    """A stack of pre-norm Mamba-2 blocks plus a final norm."""
    def __init__(self, cfg: CoreConfig):
        super().__init__()
        self.cfg = cfg
        self.blocks = nn.ModuleList([CoreBlock(cfg, i) for i in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm_f(x)


class TinyLM(nn.Module):
    """Untied token embedding, Mamba-2 core, linear head. Nothing else."""
    def __init__(self, vocab_size: int, cfg: CoreConfig):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.core = CoreStack(cfg)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, tokens):
        return self.head(self.core(self.embed(tokens)))


def make_model(vocab_size: int, device, cfg: CoreConfig | None = None):
    cfg = cfg or CoreConfig()
    return TinyLM(vocab_size, cfg).to(device), cfg


# --------------------------------------------------------------------------
# Recurrent-depth variant: instead of N distinct stacked blocks, apply a
# smaller set of blocks several times each (weight sharing across depth).
# --------------------------------------------------------------------------

# Each arm is (number of distinct blocks) x (times each block is applied).
# "total applies" = n_distinct * applies_per_block is the forward-pass depth.
ARM_SPECS = {
    "baseline_4": {"n_distinct": 4, "applies_per_block": 1},  # 4 distinct blocks, each once (the reference)
    "rd_1x4":     {"n_distinct": 1, "applies_per_block": 4},  # 1 block applied 4x (max sharing)
    "rd_2x2":     {"n_distinct": 2, "applies_per_block": 2},  # 2 blocks, each applied 2x (interpolant)
    "rd_4x2":     {"n_distinct": 4, "applies_per_block": 2},  # 4 blocks, each 2x (over-compute, same params as baseline)
    "rd_1x8":     {"n_distinct": 1, "applies_per_block": 8},  # 1 block applied 8x (max compute, min params)
}


class RecurrentDepthStack(nn.Module):
    """`n_distinct` pre-norm Mamba-2 blocks, each applied `applies_per_block`
    times in sequence. Total forward-pass depth = n_distinct * applies_per_block.

    With n_distinct=1 this is a single block looped, the weight-shared case:
    the same parameters do the work of a deeper stack at a fraction of the
    parameter count. It is the SSM analogue of a Universal / Looped Transformer.
    """
    def __init__(self, cfg: CoreConfig, n_distinct: int, applies_per_block: int):
        super().__init__()
        self.n_distinct = n_distinct
        self.applies = applies_per_block
        self.blocks = nn.ModuleList([CoreBlock(cfg, i) for i in range(n_distinct)])
        self.norm_f = RMSNorm(cfg.d_model)

    def forward(self, x):
        for blk in self.blocks:
            for _ in range(self.applies):
                x = blk(x)
        return self.norm_f(x)


class RecurrentDepthLM(nn.Module):
    """Untied token embedding, recurrent-depth Mamba-2 core, linear head."""
    def __init__(self, vocab_size: int, cfg: CoreConfig, n_distinct: int, applies: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.core = RecurrentDepthStack(cfg, n_distinct, applies)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, tokens):
        return self.head(self.core(self.embed(tokens)))


def make_recurrent_model(vocab_size: int, device, cfg: CoreConfig,
                         n_distinct: int, applies: int):
    return RecurrentDepthLM(vocab_size, cfg, n_distinct, applies).to(device)
