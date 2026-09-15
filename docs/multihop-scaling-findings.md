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

## Result 5 — CoT / intermediate supervision solves 2-hop, and composition SCALES with depth

The blocker in Result 2 was the missing gradient toward the intermediate, not model
capacity. So supervise it: every 2-hop row becomes `Q e_t A m_π(t) v_ans EOS`, adding
a CE term on the intermediate mid (the value stays 2nd-from-last, so all existing eval
is unchanged). Same `rd_1x8` block, K=16, fixed depth, 10k steps
(`diag_2hop.py --cot`, run8). We report, per test-time `r`: `mid` (hop-1: predict
`m_π(t)` at the answer slot), `val` (hop-2: predict `v` from the teacher-forced mid),
`chain` (both correct).

Training: `mid` groks fast (~step 2000 — full gradient on every row now); `val`/`chain`
then grok from ~step 5500 and saturate at chain ≈ 0.99.

**Final chain accuracy vs test-time loops `r` — multi-hop scaling, resolved per hop:**

| r | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|
| mid (hop-1) | 0.23 | 0.84 | 0.99 | 1.00 | 1.00 | 1.00 | 0.99 | 0.94 |
| val (hop-2) | 0.01 | 0.02 | 0.05 | 0.29 | 0.93 | **0.99** | 0.93 | 0.93 |
| **chain** | 0.00 | 0.01 | 0.05 | 0.29 | 0.93 | **0.99** | 0.91 | 0.87 |

The two hops resolve at **different depths**: hop-1 by r≈3, hop-2 not until r≈6–8.
Chain accuracy climbs 0.00 → 0.99 as r goes 1 → 8, and holds past the trained depth
(0.87 at r=16, no collapse). This is the **"more hops need more loops" counting-rule
signature** — test-time compute buying genuine multi-hop composition, on a single
weight-shared looped block.

Caveats specific to this result: trained at fixed r=8, so the r-sweep carries a
train/test-mismatch component — but the *ordered* hop resolution (hop-1 before hop-2)
argues for genuine sequential depth, not uniform mismatch; and eval is teacher-forced
on the mid, though mid accuracy ≈ 1.0 at r≥4 so a free-running scratchpad would chain
the same. Both are named as follow-ups below.

## Result 6 — the airtight test (randomized-depth CoT): most of Result 5's ramp was train/test mismatch

Result 5 trained at fixed r=8, so its r-sweep conflates two things: genuine
compute-insufficiency at low r, and the fact that a fixed-depth model simply cannot
operate below its trained depth. To separate them, retrain the CoT task with the
per-step depth sampled from [1,8] (`--cot --random-depth --rd-range 1 8`, run9) — now
every test-time r is in-distribution.

**Chain accuracy vs test-time r — fixed-depth (Result 5) vs randomized-depth (this):**

| r | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|
| fixed r=8 (run8) | 0.00 | 0.01 | 0.05 | 0.29 | 0.93 | 0.99 | 0.91 | 0.87 |
| random [1,8] (run9) | **0.46** | **0.97** | 0.99 | 0.99 | 0.99 | 1.00 | 0.99 | 0.99 |

With depth-matched training the model solves the 2-hop chain at essentially every
r ≥ 2, and even 0.46 at r=1. So most of Result 5's dramatic 0 → 0.99 ramp was the
fixed-depth model being unable to run shallow, **not** a real per-loop reasoning
requirement — the same "randomized-depth buys robustness, not scaling" lesson as
Result 1, now for composition. Tellingly, hop-1 (mid) is 0.999 at r=1 under
randomized-depth vs 0.23 under fixed-depth: the apparent "hop-1 needs ~3 loops" was
pure mismatch.

**The genuine residual:** chain 0.46 (r=1) → 0.97 (r=2), flat after. One apply does
hop-1 and ~half of hop-2; two applies do both reliably. So the true minimum compute
for a 2-hop chain is ~2 applies (≈ one loop per hop) — a real but *modest* effect,
the honest magnitude of the counting rule once the mismatch is removed. Randomized-depth
training also gives clean depth-robustness (flat 0.99 out to r=16, no collapse).

**Leak check (free-running eval).** Re-scoring run9's checkpoint with the model
generating its *own* mid — build the prefix up to the A token (the true mid and value
are never in the input), generate the mid, append the model's own mid, then generate the
value from it — reproduces the teacher-forced numbers at every r (chain: r1≈0.43, r2≈0.98,
r≥3≈0.99; free-running vs teacher-forced differ by ≤0.03, free-running occasionally higher).
So the teacher-forced result was not leaking: the model genuinely chains on its own
scratchpad, not on a supplied intermediate.

## Result 7 — a hop-curriculum reaches 6-hop chains (no wall found); cold-start cannot

Result 6's 3-hop failed to cold-start (serial grokking + budget: each hop only learns once the
previous is stable, so cost grows with depth). Fix: a **curriculum** that ramps the chain length
H_cur from 1 to N_max, advancing a hop only once the current depth's chain accuracy clears 0.9.
Fixed vocab (levels 0..N_max) so one model spans all depths; **swapped banks** so every
already-learned hop keeps its position relative to the query → clean warm-start transfer. run12:
N_max=6, rd_1x8, randomized-depth CoT, flat LR (`diag_nhop.py --curriculum`).

It climbed 1→2→3→4→5→6 with **no breaking point**, and the per-hop grok **accelerated**
(hop-2 ~3.5k steps, hop-3 ~5k, hops 4/5/6 ~3k each) as each new hop bootstrapped off a more
capable base — where cold-start 3-hop (Result 6/run11) never solved hop-3 at all.

**Final per-depth chain accuracy (teacher-forced / free-running):**

| depth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|-----|-----|-----|-----|-----|-----|
| TF | 0.998 | 0.997 | 0.997 | 0.996 | 0.986 | 0.968 |
| FR | 0.999 | 0.998 | 0.996 | 0.994 | 0.983 | 0.975 |

All depths solved; TF ≈ FR everywhere (no leak, genuine chaining at 6 hops). Degradation with
depth is **graceful** — chain ≈ product of per-hop accuracies (~0.99^N; 0.99^6 ≈ 0.94), eroding
smoothly rather than cliff-breaking. No hard wall within 6 → the ceiling is >6 (or budget-bound).

**6-hop chain vs test-time r:** r1 0.69 → r2 0.97 → flat (0.92 at r16). ~2 applies suffice even
for 6 hops: with CoT each hop is one lookup at its own token position, so per-lookup test-time
compute is ~**independent of chain length**. CoT externalises reasoning depth into the token
sequence; the loop-count-bound regime is the internalised (no-scratchpad) version.

## Conclusions

1. **CoT supervision unlocks composition, which then has a genuine but MODEST
   test-time-depth requirement (~2 applies for 2 hops).** Under *depth-matched*
   (randomized-depth) training the CoT model solves the 2-hop chain robustly for all
   r ≥ 2 (chain 0.46 at r=1 → 0.97 at r=2, flat/robust to r=16). The dramatic
   0 → 0.99 ramp seen under *fixed-depth* training (Result 5) was largely train/test
   mismatch, not a per-loop reasoning cost — the same robustness-not-scaling lesson as
   Result 1. The honest counting-rule signal is small: the extra hop wants ~one extra
   apply. (Beware fixed-depth r-sweeps: they overstate scaling.)
2. **The plain 2-hop objective cannot learn it** — no gradient toward the
   intermediate, so the model settles into a keyless bag-of-values. Supervising the
   intermediate (a scratchpad that rewards it) unlocks composition; capacity was
   never the limit.
3. **Weight-sharing beats specialisation.** One looped block gives depth-robustness
   (holds past trained depth, no collapse) and cross-row transfer; two per-hop blocks
   are worse on *both* axes (single-hop rise-then-collapse; 2-hop never even starts).
   This falsifies "one block per hop" and strengthens the shared-loop thesis.
4. **Retrieval scales too.** Even a single far-bank lookup needs ~2–3 loops (Result 3);
   with CoT that per-lookup cost is what stacks, not the chain length.
5. **A hop-curriculum reaches deep chains cold-start cannot.** Ramping chain length and
   advancing on mastery reached ≥6-hop chains with no wall (graceful ~0.99^N degradation,
   accelerating per-hop grok), where cold-start couldn't even do 3 (Result 7). With CoT the
   per-lookup test-time compute is ~constant in chain length — depth is externalised to the
   token sequence; the loop-count limit lives in the **internalised (no-scratchpad)** version,
   the open frontier.

## Caveats

Single seed per arm; K=16 toy vocabulary; fixed-depth training for the 2-hop runs
(random-depth stalls learning when low-`r` steps are computationally insufficient —
itself weak evidence for the counting-rule intuition). The single-hop scaling curve
is retrieval iteration, not reasoning-hop scaling; do not over-read it.

## Next experiments (rank order)

CoT supervision (Result 5), its airtight randomized-depth version (Result 6), the free-running
leak check (Result 6), and the hop-curriculum to 6 hops (Result 7) are done. Remaining:

1. **Internalise the scratchpad** (the open frontier). Can the model chain without the explicit
   intermediate token — latent/implicit CoT, reasoning carried in the recurrent state across
   loops instead of on the token tape? This is the loop-count-bound regime and the most
   Ember-relevant (reasoning below the token layer). Likely approach: train with CoT then wean
   off the intermediate token, or an auxiliary latent loss.
2. **Push the hop-curriculum higher** (--hops 8/10/12) to find where 0.99^N erosion or a genuine
   wall actually bites; and vary K / d_model to see what sets the ceiling.

Reproduce: `experiments/diag_2hop.py` (see `--cot --typed-mid --distinct-vals
--derange --swap` flags, and `--mix-hop1` for the pre-CoT curriculum); arms in
`recurrent_depth/model.py`. Result JSON/checkpoints land in `results/` (gitignored).
