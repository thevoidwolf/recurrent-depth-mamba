"""Synthetic tasks for training-recipe microtests.

`hop1_retrieval` frames the same 1-hop lookup question two ways:

  arm 'long'   [FACT] q1 v1 [SEP] q2 v2 ... [SEP] qK vK [Q] q* [A] v*
  arm 'inject' [FACT]        q* v*                     [Q] q* [A] v*

Both arms score only the token at the [A]+1 position. Entity/value pairings are
random per sample, so held-out generalisation tests the copy-from-context skill,
not memorisation. Vocab layout is deliberately small so the rig runs in minutes
on a 5090.

  0 PAD  1 BOS  2 FACT  3 SEP  4 Q  5 A  6 EOS  7..7+E-1 entities  ..values
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

PAD, BOS, FACT, SEP, QTOK, ATOK, EOS = 0, 1, 2, 3, 4, 5, 6
N_CONTROL = 7


@dataclass
class TaskCfg:
    n_entities: int = 128
    n_values: int = 128
    k_facts_long: int = 32     # arm 'long': facts inline before the query
    seed: int = 0

    @property
    def vocab_size(self) -> int:
        return N_CONTROL + self.n_entities + self.n_values

    def entity_ids(self):
        return range(N_CONTROL, N_CONTROL + self.n_entities)

    def value_ids(self):
        return range(N_CONTROL + self.n_entities, N_CONTROL + self.n_entities + self.n_values)


def _sample_pairs(cfg: TaskCfg, batch: int, k: int, gen: torch.Generator):
    # sample k distinct entities per row, and k values (with replacement) per row.
    ents = torch.stack([
        torch.randperm(cfg.n_entities, generator=gen)[:k] + N_CONTROL for _ in range(batch)
    ])
    vals = torch.randint(cfg.n_values, (batch, k), generator=gen) + (N_CONTROL + cfg.n_entities)
    return ents, vals


def _pack_long(cfg: TaskCfg, ents, vals, target_idx):
    """arm 'long': fact bank inline. seq layout:
       FACT q1 v1 SEP q2 v2 ... SEP qK vK Q q* A v* EOS
    """
    B, K = ents.shape
    device = ents.device
    parts = [torch.full((B, 1), FACT, device=device, dtype=torch.long)]
    for i in range(K):
        parts.append(ents[:, i:i+1])
        parts.append(vals[:, i:i+1])
        if i < K - 1:
            parts.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))
    q = ents.gather(1, target_idx.unsqueeze(1))
    a = vals.gather(1, target_idx.unsqueeze(1))
    parts.append(torch.full((B, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q)
    parts.append(torch.full((B, 1), ATOK, device=device, dtype=torch.long))
    parts.append(a)
    parts.append(torch.full((B, 1), EOS, device=device, dtype=torch.long))
    seq = torch.cat(parts, dim=1)
    ans_pos = seq.shape[1] - 2   # position holding v* — its logits (i.e. from position ans_pos-1) are what we score
    return seq, ans_pos


def _pack_inject(cfg: TaskCfg, ents, vals, target_idx):
    """arm 'inject': only the retrieved fact is present.
       FACT q* v* Q q* A v* EOS
    """
    B = ents.shape[0]
    device = ents.device
    q = ents.gather(1, target_idx.unsqueeze(1))
    a = vals.gather(1, target_idx.unsqueeze(1))
    seq = torch.cat([
        torch.full((B, 1), FACT, device=device, dtype=torch.long),
        q, a,
        torch.full((B, 1), QTOK, device=device, dtype=torch.long),
        q,
        torch.full((B, 1), ATOK, device=device, dtype=torch.long),
        a,
        torch.full((B, 1), EOS, device=device, dtype=torch.long),
    ], dim=1)
    ans_pos = seq.shape[1] - 2
    return seq, ans_pos


def sample_batch(cfg: TaskCfg, arm: str, batch: int, gen: torch.Generator, device="cuda"):
    """Return (tokens[B,L], ans_pos:int, target_token[B]).
    The model is teacher-forced; loss is CE at position ans_pos-1 against target_token.
    """
    k = cfg.k_facts_long if arm == "long" else 1
    ents, vals = _sample_pairs(cfg, batch, k, gen)
    ents = ents.to(device); vals = vals.to(device)
    target_idx = torch.randint(k, (batch,), generator=gen).to(device)
    if arm == "long":
        seq, ans_pos = _pack_long(cfg, ents, vals, target_idx)
    elif arm == "inject":
        seq, ans_pos = _pack_inject(cfg, ents, vals, target_idx)
    else:
        raise ValueError(arm)
    target = seq[:, ans_pos]
    return seq, ans_pos, target


# --------------------------------------------------------------------------
# hop2_retrieval — chained retrieval, for rig 002
# --------------------------------------------------------------------------
#
# Graph per sample: K distinct entities e_1..e_K, a hop-1 permutation π on
# {1..K}, and a hop-2 mapping e_i → v_i (values with replacement). Answering
# for e_t means: follow (e_t, e_{π(t)}) then (e_{π(t)}, v_{π(t)}) → v_{π(t)}.
#
# Token type disambiguates the two hops without needing explicit R1/R2 tokens:
# a "fact" is a triple (source, dest, SEP); if dest is an entity-token it is a
# hop-1 fact, if dest is a value-token it is a hop-2 fact. The model must
# induce that structure.
#
# Layouts (writing entities as e#, values as v#):
#
#   long_2hop:
#       FACT e1 e2 SEP e3 e4 SEP ... [K hop-1 pairs, all interior SEPs]
#            SEP e1 v1 SEP e3 v3 SEP ...   [K hop-2 pairs]
#            Q e_t A v_ans EOS
#
#   inject_2hop:
#       FACT e_t e_mid SEP e_mid v_ans Q e_t A v_ans EOS
#
#   inject_collapsed (reference: the retriever already collapsed the chain):
#       FACT e_t v_ans Q e_t A v_ans EOS       ← identical to rig 001 inject


def _pack_long_2hop(cfg: TaskCfg, ents, vals, perm, target_idx):
    """K hop-1 pairs (e_i, e_{π(i)}) then K hop-2 pairs (e_i, v_i)."""
    B, K = ents.shape
    device = ents.device
    # hop-1 destinations: e[π(i)]
    ent_dest = ents.gather(1, perm)          # [B, K]

    parts = [torch.full((B, 1), FACT, device=device, dtype=torch.long)]
    for i in range(K):
        parts.append(ents[:, i:i+1])
        parts.append(ent_dest[:, i:i+1])
        if i < K - 1:
            parts.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))
    for i in range(K):
        parts.append(ents[:, i:i+1])
        parts.append(vals[:, i:i+1])
        if i < K - 1:
            parts.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))

    q = ents.gather(1, target_idx.unsqueeze(1))               # e_t
    mid_slot = perm.gather(1, target_idx.unsqueeze(1))        # π(t)
    a = vals.gather(1, mid_slot)                              # v_{π(t)}

    parts.append(torch.full((B, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q)
    parts.append(torch.full((B, 1), ATOK, device=device, dtype=torch.long))
    parts.append(a)
    parts.append(torch.full((B, 1), EOS, device=device, dtype=torch.long))
    seq = torch.cat(parts, dim=1)
    ans_pos = seq.shape[1] - 2
    return seq, ans_pos


def _pack_inject_2hop(cfg: TaskCfg, ents, vals, perm, target_idx):
    """Just the two facts on the chain."""
    B = ents.shape[0]
    device = ents.device
    q = ents.gather(1, target_idx.unsqueeze(1))               # e_t
    mid_slot = perm.gather(1, target_idx.unsqueeze(1))        # π(t)
    e_mid = ents.gather(1, mid_slot)                          # e_{π(t)}
    a = vals.gather(1, mid_slot)                              # v_{π(t)}
    seq = torch.cat([
        torch.full((B, 1), FACT, device=device, dtype=torch.long),
        q, e_mid,
        torch.full((B, 1), SEP, device=device, dtype=torch.long),
        e_mid, a,
        torch.full((B, 1), QTOK, device=device, dtype=torch.long),
        q,
        torch.full((B, 1), ATOK, device=device, dtype=torch.long),
        a,
        torch.full((B, 1), EOS, device=device, dtype=torch.long),
    ], dim=1)
    ans_pos = seq.shape[1] - 2
    return seq, ans_pos


def _pack_inject_collapsed(cfg: TaskCfg, ents, vals, perm, target_idx):
    """Retriever pre-collapsed the chain — identical shape to rig 001 inject."""
    B = ents.shape[0]
    device = ents.device
    q = ents.gather(1, target_idx.unsqueeze(1))
    mid_slot = perm.gather(1, target_idx.unsqueeze(1))
    a = vals.gather(1, mid_slot)
    seq = torch.cat([
        torch.full((B, 1), FACT, device=device, dtype=torch.long),
        q, a,
        torch.full((B, 1), QTOK, device=device, dtype=torch.long),
        q,
        torch.full((B, 1), ATOK, device=device, dtype=torch.long),
        a,
        torch.full((B, 1), EOS, device=device, dtype=torch.long),
    ], dim=1)
    ans_pos = seq.shape[1] - 2
    return seq, ans_pos


def _pack_long_2hop_swap(cfg: TaskCfg, ents, vals, perm, target_idx):
    """ADVERSARIAL bank swap: hop-2 bank BEFORE hop-1 bank.

    Same chain as _pack_long_2hop (e_t → e_π(t) → v_π(t)), same targets,
    but the two banks are emitted in reverse order. Per rig 041a's
    terminology this is "swapped" / adversarial-with-banks. Zero-shot
    accuracy on this vs `_pack_long_2hop` distinguishes layout-specific
    circuits from layout-general ones.
    """
    B, K = ents.shape
    device = ents.device
    ent_dest = ents.gather(1, perm)

    parts = [torch.full((B, 1), FACT, device=device, dtype=torch.long)]
    # hop-2 bank FIRST (was second in the favorable layout)
    for i in range(K):
        parts.append(ents[:, i:i+1])
        parts.append(vals[:, i:i+1])
        if i < K - 1:
            parts.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))
    parts.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))
    # hop-1 bank SECOND
    for i in range(K):
        parts.append(ents[:, i:i+1])
        parts.append(ent_dest[:, i:i+1])
        if i < K - 1:
            parts.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))

    q = ents.gather(1, target_idx.unsqueeze(1))
    mid_slot = perm.gather(1, target_idx.unsqueeze(1))
    a = vals.gather(1, mid_slot)

    parts.append(torch.full((B, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q)
    parts.append(torch.full((B, 1), ATOK, device=device, dtype=torch.long))
    parts.append(a)
    parts.append(torch.full((B, 1), EOS, device=device, dtype=torch.long))
    seq = torch.cat(parts, dim=1)
    ans_pos = seq.shape[1] - 2
    return seq, ans_pos


def _pack_long_2hop_interleave(cfg: TaskCfg, ents, vals, perm, target_idx, gen):
    """RANDOM INTERLEAVE: 2K facts (K hop-1 pairs + K hop-2 pairs) shuffled.

    Kills the two-bank structure entirely — each fact appears at a random
    absolute position with SEP separators. Tests robustness of the
    two-segment circuit to structure-free layouts (rig 041c).

    NOTE: `gen` is used for the per-batch shuffle (kept off the CUDA
    graph — CPU-side).
    """
    B, K = ents.shape
    device = ents.device
    ent_dest = ents.gather(1, perm)

    # Build fact-pair tensor [B, 2K, 2]: first K are (e_i, e_π(i)) hop-1,
    # last K are (e_i, v_i) hop-2. Shuffle along dim=1 per row.
    hop1 = torch.stack([ents, ent_dest], dim=-1)   # [B, K, 2]
    hop2 = torch.stack([ents, vals], dim=-1)       # [B, K, 2]
    all_facts = torch.cat([hop1, hop2], dim=1)     # [B, 2K, 2]

    # Per-row shuffle (CPU-side because torch.randperm isn't batched)
    shuffled = []
    for b in range(B):
        idx = torch.randperm(2 * K, generator=gen)
        shuffled.append(all_facts[b, idx])
    all_facts = torch.stack(shuffled, dim=0).to(device)

    parts = [torch.full((B, 1), FACT, device=device, dtype=torch.long)]
    for i in range(2 * K):
        parts.append(all_facts[:, i, 0:1])   # source
        parts.append(all_facts[:, i, 1:2])   # dest / value
        if i < 2 * K - 1:
            parts.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))

    q = ents.gather(1, target_idx.unsqueeze(1))
    mid_slot = perm.gather(1, target_idx.unsqueeze(1))
    a = vals.gather(1, mid_slot)

    parts.append(torch.full((B, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q)
    parts.append(torch.full((B, 1), ATOK, device=device, dtype=torch.long))
    parts.append(a)
    parts.append(torch.full((B, 1), EOS, device=device, dtype=torch.long))
    seq = torch.cat(parts, dim=1)
    ans_pos = seq.shape[1] - 2
    return seq, ans_pos


def sample_2hop_batch(cfg: TaskCfg, arm: str, batch: int, gen: torch.Generator, device="cuda"):
    """Arms: 'long_2hop', 'long_2hop_swap', 'long_2hop_interleave',
    'inject_2hop', 'inject_collapsed'. Uses cfg.k_facts_long as K.
    """
    K = cfg.k_facts_long
    ents, vals = _sample_pairs(cfg, batch, K, gen)
    perm = torch.stack([torch.randperm(K, generator=gen) for _ in range(batch)])
    ents = ents.to(device); vals = vals.to(device); perm = perm.to(device)
    target_idx = torch.randint(K, (batch,), generator=gen).to(device)

    if arm == "long_2hop":
        seq, ans_pos = _pack_long_2hop(cfg, ents, vals, perm, target_idx)
    elif arm == "long_2hop_swap":
        seq, ans_pos = _pack_long_2hop_swap(cfg, ents, vals, perm, target_idx)
    elif arm == "long_2hop_interleave":
        seq, ans_pos = _pack_long_2hop_interleave(cfg, ents, vals, perm, target_idx, gen)
    elif arm == "inject_2hop":
        seq, ans_pos = _pack_inject_2hop(cfg, ents, vals, perm, target_idx)
    elif arm == "inject_collapsed":
        seq, ans_pos = _pack_inject_collapsed(cfg, ents, vals, perm, target_idx)
    else:
        raise ValueError(arm)
    target = seq[:, ans_pos]
    return seq, ans_pos, target


# --------------------------------------------------------------------------
# hop3_retrieval — 3-chain, for rig 033 counting-rule staircase
# --------------------------------------------------------------------------
#
# Graph per sample: K distinct entities, two independent permutations
# π and ρ. Chain per entity e_i: e_i → e_π(i) → e_ρ(π(i)) → v_ρ(π(i)).
# Three fact banks:
#   bank1 (link1): (e_i, e_π(i))       — hop-1 destination
#   bank2 (link2): (e_j, e_ρ(j))       — hop-2 destination, keyed on any entity
#                                        (we use e_π(i) natural key when computing target)
#   bank3 (link3): (e_j, v_j)          — hop-3 value, keyed on any entity
#
# Query: `Q e_t A v_ans EOS` where v_ans = v_{ρ(π(t))}.
#
# Layouts:
#   long_3hop_favorable  — bank3 · SEP · bank2 · SEP · bank1 · Q ...
#       Backward scan from Q with query e_t sees:
#         bank1 first (rightmost)  → binds e_t → e_π(t)
#         bank2 next               → binds e_π(t) → e_ρ(π(t))
#         bank3 last (leftmost)    → binds e_ρ(π(t)) → v_ρ(π(t))
#       Single backward pass composes all 3 links in order.
#   long_3hop_adversarial — bank1 · SEP · bank2 · SEP · bank3 · Q ...
#       Backward scan hits bank3 first (link3, value-terminal) — has no
#       state to bind against without link2 arriving first. Each pass
#       can chain at most 1 link → counting rule says need 3 alternations.


def _sample_3hop_graph(cfg: TaskCfg, batch: int, gen: torch.Generator):
    """Return (ents, vals, perm1, perm2, target_idx) for a fresh 3-hop graph.

    Uses K = cfg.k_facts_long entities. Values sampled with replacement.
    perm1, perm2 are per-row permutations of {0..K-1}.
    """
    K = cfg.k_facts_long
    ents, vals = _sample_pairs(cfg, batch, K, gen)
    perm1 = torch.stack([torch.randperm(K, generator=gen) for _ in range(batch)])
    perm2 = torch.stack([torch.randperm(K, generator=gen) for _ in range(batch)])
    target_idx = torch.randint(K, (batch,), generator=gen)
    return ents, vals, perm1, perm2, target_idx


def _pack_long_3hop(cfg: TaskCfg, ents, vals, perm1, perm2, target_idx, favorable: bool):
    """K link-1 pairs (e_i, e_π(i)), K link-2 pairs (e_i, e_ρ(i)),
    K link-3 pairs (e_i, v_i). Emit in favorable or adversarial order.

    Favorable emits: bank3, bank2, bank1, Q.  (backward-scan-friendly)
    Adversarial emits: bank1, bank2, bank3, Q. (each pass chains ≤ 1 link)
    """
    B, K = ents.shape
    device = ents.device
    ent_link1 = ents.gather(1, perm1)                # e_π(i)
    ent_link2 = ents.gather(1, perm2)                # e_ρ(i)  (keyed by index, not composed)

    def _bank(src, dst):
        p = []
        for i in range(K):
            p.append(src[:, i:i+1])
            p.append(dst[:, i:i+1])
            if i < K - 1:
                p.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))
        return p

    parts = [torch.full((B, 1), FACT, device=device, dtype=torch.long)]
    bank1 = _bank(ents, ent_link1)          # (e_i, e_π(i))
    bank2 = _bank(ents, ent_link2)          # (e_i, e_ρ(i))
    bank3 = _bank(ents, vals)               # (e_i, v_i)

    order = (bank3, bank2, bank1) if favorable else (bank1, bank2, bank3)
    for bi, bank in enumerate(order):
        parts.extend(bank)
        if bi < 2:
            parts.append(torch.full((B, 1), SEP, device=device, dtype=torch.long))

    q = ents.gather(1, target_idx.unsqueeze(1))                   # e_t
    mid1 = perm1.gather(1, target_idx.unsqueeze(1))               # π(t)
    mid2 = perm2.gather(1, mid1)                                  # ρ(π(t))
    a = vals.gather(1, mid2)                                      # v_ρ(π(t))

    parts.append(torch.full((B, 1), QTOK, device=device, dtype=torch.long))
    parts.append(q)
    parts.append(torch.full((B, 1), ATOK, device=device, dtype=torch.long))
    parts.append(a)
    parts.append(torch.full((B, 1), EOS, device=device, dtype=torch.long))
    seq = torch.cat(parts, dim=1)
    ans_pos = seq.shape[1] - 2
    return seq, ans_pos


def sample_3hop_batch(cfg: TaskCfg, arm: str, batch: int, gen: torch.Generator, device="cuda"):
    """Arms: 'long_3hop_favorable', 'long_3hop_adversarial'."""
    ents, vals, perm1, perm2, target_idx = _sample_3hop_graph(cfg, batch, gen)
    ents = ents.to(device); vals = vals.to(device)
    perm1 = perm1.to(device); perm2 = perm2.to(device)
    target_idx = target_idx.to(device)

    if arm == "long_3hop_favorable":
        seq, ans_pos = _pack_long_3hop(cfg, ents, vals, perm1, perm2, target_idx, favorable=True)
    elif arm == "long_3hop_adversarial":
        seq, ans_pos = _pack_long_3hop(cfg, ents, vals, perm1, perm2, target_idx, favorable=False)
    else:
        raise ValueError(arm)
    target = seq[:, ans_pos]
    return seq, ans_pos, target


# --------------------------------------------------------------------------
# inject-with-distance-padding — for rig 004's decay probe
# --------------------------------------------------------------------------
#
# Layout:  FACT q* v* [pad_ent × D] Q q* A v* EOS
#
# The pad tokens are randomly-drawn entity tokens *not equal* to q*. As D grows,
# the SSM must carry v* further through its recurrent state to still emit it at
# the answer position. This measures the decay length of the free short-context
# copy circuit that inject exploits.


def sample_inject_padded_batch(cfg: TaskCfg, distractor_pad: int,
                               batch: int, gen: torch.Generator, device="cuda"):
    """arm is implicit ('inject_padded'); distractor_pad = D pad entity tokens."""
    ents, vals = _sample_pairs(cfg, batch, 1, gen)
    ents = ents.to(device); vals = vals.to(device)
    q = ents[:, :1]
    v = vals[:, :1]
    B = batch
    device = q.device
    parts = [
        torch.full((B, 1), FACT, device=device, dtype=torch.long),
        q, v,
    ]
    if distractor_pad > 0:
        pad_ids = torch.randint(cfg.n_entities, (B, distractor_pad), generator=gen).to(device) + N_CONTROL
        # avoid pads that equal q* — resample the collisions in-place
        clash = (pad_ids == q)
        while clash.any():
            resamp = torch.randint(cfg.n_entities, clash.shape, generator=gen).to(device) + N_CONTROL
            pad_ids = torch.where(clash, resamp, pad_ids)
            clash = (pad_ids == q)
        parts.append(pad_ids)
    parts.extend([
        torch.full((B, 1), QTOK, device=device, dtype=torch.long),
        q,
        torch.full((B, 1), ATOK, device=device, dtype=torch.long),
        v,
        torch.full((B, 1), EOS, device=device, dtype=torch.long),
    ])
    seq = torch.cat(parts, dim=1)
    ans_pos = seq.shape[1] - 2
    target = seq[:, ans_pos]
    return seq, ans_pos, target
