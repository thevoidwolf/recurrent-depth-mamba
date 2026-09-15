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
    block: str = "auto"   # "auto" -> mamba2 if importable else fallback; or force "mamba2"/"fallback"
    # --- recurrent-depth stabilizers (all default to the original behaviour) ---
    norm_position: str = "pre"     # "pre": x + mixer(norm(x));  "post": norm(x + mixer(x))
    inject_input: bool = False     # re-add the stack input embedding before every apply (Huginn)
    bptt_window: int | None = None # if set, only backprop through the last k applies (truncated BPTT)


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
        # Prefer the real kernel, but only if it actually imports. On a box with a
        # GPU visible but no built mamba_ssm (e.g. stock ROCm) the old code crashed
        # here; now "auto" degrades to the pure-PyTorch scan with a warning instead.
        block = "fallback"
        if torch.cuda.is_available():
            try:
                from mamba_ssm import Mamba2  # noqa: F401
                block = "mamba2"
            except Exception as e:
                import warnings
                warnings.warn(
                    f"mamba_ssm unavailable ({type(e).__name__}: {e}); falling back "
                    f"to the pure-PyTorch scan. Headline numbers need the real kernel."
                )
    if block == "mamba2":
        from mamba_ssm import Mamba2
        return Mamba2(d_model=cfg.d_model, d_state=cfg.d_state, d_conv=cfg.d_conv,
                      expand=cfg.expand, headdim=cfg.headdim, layer_idx=layer_idx)
    if block == "fallback":
        return FallbackMixer(cfg)
    raise ValueError(f"unknown block {block!r} (use 'auto', 'mamba2', or 'fallback')")


class CoreBlock(nn.Module):
    """Residual wrapper around one mixer, pre- or post-norm.

    pre-norm  (original):   x + mixer(norm(x))    -- when the block is unrolled many
                                                     times the residual stream can grow
                                                     ~sqrt(depth) and eventually diverge.
    post-norm (2602.12078): norm(x + mixer(x))    -- bounds the state magnitude across
                                                     unrolling; the make-or-break
                                                     stabiliser for looped SSM depth.
    """
    def __init__(self, cfg: CoreConfig, layer_idx: int):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model)
        self.mixer = _build_mixer(cfg, layer_idx)
        self.post = (cfg.norm_position == "post")

    def forward(self, x):
        if self.post:
            return self.norm(x + self.mixer(x))
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
    "rd_2x4":     {"n_distinct": 2, "applies_per_block": 4},  # 2 blocks, each 4x (8 total applies, one block per hop: composition vs reuse)
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
        self.inject_input = cfg.inject_input
        self.bptt_window = cfg.bptt_window
        self.blocks = nn.ModuleList([CoreBlock(cfg, i) for i in range(n_distinct)])
        self.norm_f = RMSNorm(cfg.d_model)

    def forward(self, x, applies: int | None = None, trace: bool = False):
        """Apply each distinct block `applies` times (default: the trained value).

        applies  override the per-block loop count at call time -- drives both
                 test-time depth sweeps and randomized-depth training.
        trace    also return the residual-stream RMS after every apply, for the
                 pre- vs post-norm norm-growth diagnostic.
        """
        applies = self.applies if applies is None else applies
        x0 = x                                    # embedded input, re-added each loop (input feedback)
        total = self.n_distinct * applies
        norms = [] if trace else None
        i = 0
        for blk in self.blocks:
            for _ in range(applies):
                x = blk(x + x0 if self.inject_input else x)
                i += 1
                if trace:
                    norms.append(float(x.detach().pow(2).mean(-1).sqrt().mean()))
                # Truncated BPTT: cut the graph so gradients reach only the last
                # `bptt_window` applies -- keeps memory/stability bounded at depth.
                if self.bptt_window is not None and (total - i) >= self.bptt_window:
                    x = x.detach()
        out = self.norm_f(x)
        return (out, norms) if trace else out


class RecurrentDepthLM(nn.Module):
    """Untied token embedding, recurrent-depth Mamba-2 core, linear head."""
    def __init__(self, vocab_size: int, cfg: CoreConfig, n_distinct: int, applies: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, cfg.d_model)
        self.core = RecurrentDepthStack(cfg, n_distinct, applies)
        self.head = nn.Linear(cfg.d_model, vocab_size, bias=False)

    def forward(self, tokens, applies: int | None = None, trace: bool = False):
        x = self.embed(tokens)
        if trace:
            h, norms = self.core(x, applies=applies, trace=True)
            return self.head(h), norms
        return self.head(self.core(x, applies=applies))


def make_recurrent_model(vocab_size: int, device, cfg: CoreConfig,
                         n_distinct: int, applies: int):
    return RecurrentDepthLM(vocab_size, cfg, n_distinct, applies).to(device)
