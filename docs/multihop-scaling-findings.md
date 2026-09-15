# Test-time depth on recurrent Mamba-2: robustness, retrieval scaling, and the composition wall

_2026-09-15. Synthetic-task study on the weight-shared looped Mamba-2 in this repo
(`rd_*` arms in `recurrent_depth/model.py`). All GPU numbers use the real Mamba-2
kernel._

## Question

A looped ("recurrent-depth") block applies the same weights `r` times. Two very
different things can improve with `r`:

- **Depth-robustness** — the model already solves the task in one pass and simply
  *tolerates* extra loops without degrading.
- **Depth-scaling** — extra loops let the model solve instances it *could not*
  solve with fewer loops ("loops buy reasoning depth").

Only the second is the interesting claim. This note pins down which one we actually
observe, on tasks of increasing compositional depth.

## Setup

**Model.** `TinyLM`: token embedding → looped Mamba-2 core → linear head
(`d_model=256`, `d_state=64`, `expand=2`, `headdim=64`). Arms are
`(n_distinct_blocks) × (applies_per_block)`; total forward depth = the product.
Post-norm + input re-injection (the stabilisers from
`experiments/03_test_time_depth.py`) unless noted.

**Tasks** (`recurrent_depth/tasks.py`, and the instrumented
`experiments/diag_2hop.py`):

- **1-hop retrieval** (`long`): `K` inline `(entity, value)` facts, then a query;
  answer = the queried entity's value.
- **2-hop chain**: bank-1 `(e_i, m_π(i))` and bank-2 `(m_i, v_i)`; query `e_t`,
  answer `v_{π(t)}`. Requires composing two lookups.

**Diagnostics.** Beyond accuracy we track, at the answer position: the label *type*
of the prediction (correct / query's-own-value / other-present-value / absent /
entity), `rank(v_ans)`, the present-value logit gap, and the recency histogram over
bank slots. These separate "learned the binding" from "learned a shortcut."

## Result 1 — 1-hop: randomized-depth training buys robustness, not scaling

Training at a *randomized* per-step depth makes the looped model flat-accurate at
**1.00 for every test-time `r` from 1 to 32**, including 4× extrapolation past the
training max. Post-norm keeps the residual stream bounded; the fixed-depth model is
a fragile point solution that only works at its trained `r`.

This is depth-**robustness**. It cannot be depth-scaling, because `K`-way
associative recall is solvable in a single pass — there is no harder instance for
extra loops to unlock. A flat line at 1.00 is the tell.

## Result 2 — why naive 2-hop training plateaus at a shortcut

Trained directly on the 2-hop chain, the model's accuracy sticks at ~0.10 while loss
falls and confidence rises. Diagnosis (label-type breakdown + logit inspection):

- The prediction is almost always *some value present in the fact bank* (never an
  entity/control token), but the model assigns **equal logit to the answer, the
  query's own value, and every other present value** — a *keyless bag-of-values*
  readout.
- Root cause: **the 2-hop objective provides no gradient toward binding.** The
  "predict the most-frequent present value" shortcut scores ~0.105; a partial
  hop-1 circuit (predict `v_t`) scores only ~0.069 — *lower*. Building the
  intermediate circuit is actively penalised relative to the shortcut.
- The "favorable" bank order (bank-1 first) is in fact the *hard* layout for a
  forward-scanning SSM (the naming came from a backward-scan intuition). The
  **swapped** order (bank-2 first) admits a clean 2-apply solution.

Fixes that give binding a gradient: typed intermediate entities (disjoint vocab,
kills key-collision), distinct values (kills the count shortcut), derangement
(kills the fixed-point reward), swapped bank order, and a **50% hop-1 curriculum**
(half the training rows ask for the intermediate entity directly). Single-hop
binding, in isolation, groks cleanly to 0.95–1.0 under this stack — so there is no
harness bug; only the composition reward was missing.

## Result 3 — `rd_1x8` (one looped block): retrieval scales; composition partially forms

Recipe: one block looped 8×, post-norm+inject, fixed depth, K=16, typed + distinct +
derange + swap + 50% hop-1 curriculum, 10k steps
(`experiments/diag_2hop.py`, run6).

**Single-hop accuracy vs test-time `r` — a genuine scaling curve:**

| r | 1 | 2 | 3 | 4 | 6 | 8 | 16 |
|---|-----|-----|-----|-----|-----|-----|-----|
| acc | 0.27 | 0.81 | 0.97 | 1.00 | 1.00 | 1.00 | 0.98 |
| gap | 2.6 | 5.5 | 8.6 | 11.1 | 13.2 | 13.4 | 9.8 |

Accuracy *rises* monotonically with loops (r=1 fails, saturates by r≈6) **and** holds
out to 4× the trained depth with no collapse. One associative-recall hop in this
far-bank layout consumes ~3 applies. This is the first unambiguous "loops buy
accuracy" curve of the study — but it is **retrieval depth** (one lookup that takes
several iterations), not multi-hop reasoning.

**Two-hop:** accuracy stays at bag-of-present chance (~0.07), **but** `rank(v_ans)`
improves strongly with `r` (61 → 11.6 as r goes 1 → 8) and with training
(60 → ~12). The loops *are* doing compositional work; a single reused block just
never pushes the answer to top-1. Composition forms, incompletely.

## Result 4 — `rd_2x4` (two blocks, one per hop): worse on both axes

Hypothesis: give hop-1 and hop-2 their own block so the second hop needn't *reuse*
the first's circuitry. Same curriculum, 2 distinct blocks × 4 applies = 8 total
(run7). It fails **decisively, on both axes**:

**Single-hop rises then COLLAPSES** (contrast with `rd_1x8`):

| test-time r | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|
| **rd_1x8** | 0.27 | 0.81 | 0.97 | 1.00 | 1.00 | 1.00 | — | 0.98 |
| **rd_2x4** | 0.08 | 0.48 | 0.99 | **1.00** | 0.94 | 0.53 | 0.16 | 0.10 |

`rd_2x4` peaks exactly at its trained depth and falls apart beyond it — the classic
unstabilised over-unrolling signature.

**Two-hop: total failure** — pure chance (`rank(v_ans)` ≈ 63, acc ≈ 0.008) at every
`r`, never even loading present-values. Worse than `rd_1x8`'s partial progress.

## Conclusions

1. **Test-time scaling is real — for retrieval depth.** A single looped block turns
   more test-time loops into higher accuracy on a hard single lookup, and stays
   robust well past its trained depth. (This is not yet multi-hop *reasoning*.)
2. **Weight-sharing beats specialisation, on both robustness and composition.** The
   single looped block (a) transfers its hop-1 retrieval circuit to the 2-hop rows,
   giving partial composition, and (b) learns an *iterable* transform that
   extrapolates in depth. Splitting into per-hop blocks removes the transfer and
   overfits each block to its trained loop count. This **falsifies "one block per
   hop"** and strengthens the shared-loop recurrent-depth thesis.
3. **Two-hop composition remains unsolved** — the frontier. Both arms max out below
   top-1; the shared block only ranks the right answer ~12th.

## Caveats

Single seed per arm; K=16 toy vocabulary; fixed-depth training for the 2-hop runs
(random-depth stalls learning when low-`r` steps are computationally insufficient —
itself weak evidence for the counting-rule intuition). The single-hop scaling curve
is retrieval iteration, not reasoning-hop scaling; do not over-read it.

## Next experiments (rank order)

1. **Intermediate / chain-of-thought supervision.** On 2-hop rows, emit the
   intermediate entity `m_π(t)` *then* the value (`A m_π(t) v_ans EOS`), scoring both
   positions — an explicit scratchpad that hands the model the chain. Highest
   expected payoff for actually cracking composition.
2. **More budget for `rd_1x8` 2-hop.** It was still improving `rank(v_ans)` at 10k
   steps; longer training and/or a warmer LR tail may complete the grok without any
   architecture change.
3. **Depth-position readout / per-apply loss on the shared block.** Encourage each
   apply to advance one hop, keeping weight-sharing.

Reproduce: `experiments/diag_2hop.py` (see `--typed-mid --distinct-vals --derange
--swap --mix-hop1` flags); arms in `recurrent_depth/model.py`.
