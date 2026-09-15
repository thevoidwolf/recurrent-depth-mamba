# Test-time depth on recurrent Mamba-2: robustness, retrieval scaling, and the composition wall

_2026-09-15. Synthetic-task study on the weight-shared looped Mamba-2 in this repo
(`rd_*` arms in `recurrent_depth/model.py`). All GPU numbers use the real Mamba-2
kernel. Reproduce via `experiments/diag_2hop.py` and `experiments/diag_nhop.py`._

## Summary (TL;DR)

The arc, and what each step actually established (numbers are chain accuracy unless noted):

1. **Randomized-depth training → depth-robustness, not scaling.** 1-hop recall stays 1.0 at every
   test-time loop count r (Result 1). A flat line is robustness; it can't show "loops buy reasoning."
2. **Plain multi-hop can't be learned** — the objective gives no gradient to the intermediate, so the
   model settles into a keyless "bag of present values" (Result 2).
3. **CoT / intermediate supervision unlocks composition**, and it's leak-free (free-running eval matches
   teacher-forced). But under *fixed-depth* training the striking r-curve was mostly train/test mismatch
   (caught and corrected in Result 6); the genuine per-hop compute is modest (Results 3–6).
4. **Weight-sharing beats per-hop blocks** on both depth-robustness and composition (Result 4).
5. **A hop-curriculum reaches ≥6-hop chains with no wall** (cold-start can't do 3). But with CoT the
   depth lives on the **token tape** — per-lookup compute is ~constant in chain length (Result 7).
6. **The core internalises reasoning via soft-weaning, and it is LOOP-BOUND.** Replacing CoT tokens
   with content-free pause tokens (gradually) puts a **2-hop** chain in-state at value 0.99, needing
   **≥2 loops** (r=1: 0.06, r=2: 0.97) — real in-core "loops buy reasoning depth" (Result 8).
7. **An apparent "~2-hop cap" was a weaning-schedule artifact; the mechanism is a per-position register
   (Result 10).** A probe shows the stalled 3-hop model computed *nothing* in-state (schedule with no
   replay had destroyed its own scaffold). A **replaying** wean internalises 3-hop fully (0.99, CoT
   retained); an equal-budget ablation shows replay — not budget — is the cause. Each hop's result is
   held **write-once at its own pause-token position** (causally verified by donor-swap and single-loop
   patching), and **N loops ≈ N hops** (2-hop needs r≥2, 3-hop r≥3). So "in-state" reasoning is
   **latent pause-token CoT** (O(N) positions, content latent), not scratchpad-free — costing ~N/2× CoT.

**Bottom line:** a looped SSM core reasons multi-hop *below the token layer* by writing each hop's
result into a **content-free pause-token position** (a residual-stream register) and holding it across
loops; loop count gates hop count (**N loops ≈ N hops**). The apparent shallow ceiling was a weaning
artifact (fixed by replaying shallower depths), not a capacity limit — and this is *latent CoT*, still
O(N) tape positions, not reasoning for free. Method lessons banked: **fixed-depth r-sweeps overstate
scaling — always confirm with randomized-depth**; prefer leak-free (free-running) evals; and **verify
an internalisation "ceiling" against forgetting before calling it capacity.**

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
Post-norm + input feedback (re-adding the embedded input before each loop; the stabilisers from
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

## Result 8 — soft-weaning internalises 2-hop reasoning, and it is LOOP-BOUND

A first internalise attempt (warm-start CoT then fine-tune on the *plain* task, run13) failed —
2-hop stuck at bag-of-values (~0.06). But that had a confound: the plain task deleted the
intermediate token **and its compute position**, leaving zero extra steps to do the hop in-state.

Soft-weaning fixes it: keep the intermediate position but fill it with a content-free **PAUSE**
token (reused the never-emitted BOS id, so vocab/checkpoints are unchanged), drop its supervision,
and ramp the fraction of fully-internal rows 0→1 (probabilistic per-row; the remaining CoT rows
reinforce the chain throughout). h=2 from scratch (`diag_nhop.py --wean`, run14).

`value@FULL_INTERNAL` (2-hop chain solved in-state, intermediate = PAUSE, leak-free by
construction) climbed *with* the wean and saturated at **0.99** by p=1 — vs run13's 0.06. So the
looped core **can** do 2-hop composition internally; run13's failure was removing the compute
budget, not a capability limit.

**And the internalised chain is loop-bound.** Final value acc vs test-time r (fully internal):

| r | 1 | 2 | 3 | 4 | 6 | 8 |
|---|-----|-----|-----|-----|-----|-----|
| acc | 0.06 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |

r=1 (one loop) fails at bag-of-values; r=2 (two loops) solves it → the internalised 2-hop chain
needs **≥2 loops ≈ one loop per hop**. This is "loops buy reasoning depth" in the pure sense, and
the key contrast with Result 7: with CoT the depth lives on the **token tape** (~2 applies
regardless of chain length, one lookup per token position); internalised, loop count directly gates
hop count.

> **Wording note (see Result 10).** "The whole chain resolves in the recurrent state" is imprecise:
> Result 10's probe shows each intermediate is held in the **residual stream at its own pause-token
> position** (a positional register), not co-located in the SSM hidden state. So this is *latent
> pause-token CoT* — one thinking position per hop, content latent — with the loops supplying the
> per-hop compute; it is not scratchpad-free reasoning.

## Result 9 — an APPARENT shallow in-state depth limit (~2 hops) — SUPERSEDED by Result 10

> **Superseded.** The "~2-hop cap" reported here was a **weaning-schedule artifact**, not a
> capacity limit. With a replaying wean schedule the 3-hop chain internalises fully (0.99) and
> the mechanism is a per-position register, not a two-intermediate bottleneck. See **Result 10**.
> The `--wean-mode hop` result below is left as-is for the record.

Does the internalised, loop-bound behaviour scale to 3 hops (would 3-hop-in-state need r≥3)?
Soft-wean a 3-hop chain, warm-started from the run12 CoT model (`--wean-depth 3` keeps run12's
6-level vocab while weaning depth 3). Two schedules:

- **Per-row all-at-once** (`--wean-mode row`, as in Result 8): failed immediately — 3-hop internal
  flat at ~0.06 through p≈0.37 (2-hop was ~0.9 by then). Flipping a row to "all 3 hops in-state at
  once" is too abrupt.
- **Per-hop gradual** (`--wean-mode hop`, internalise one trailing hop at a time): got partway, and
  revealed the limit. wean_k=1 (last hop internal — the final **2** hops carried in-state, hop-1 on
  the tape) GROKKED (loss → 0.02). wean_k=2 (all intermediates paused, full 3-hop in-state) STALLED —
  value flat at bag-of-values (~0.06) for 3500+ steps, and continued training eroded CoT.

So the core internalises **up to ~2 hops** of in-state composition (full 2-hop in Result 8; the last
2 hops of a 3-chain here) but not a 3rd fully-in-state hop with this recipe. It is **not** a loop
budget limit (eval used r=8 ≫ 3). Open whether it is a hard in-state capacity limit (carrying two
intermediates in the recurrent state) or just needs finer weaning / more steps / larger d_state — so
the "N hops ≈ N loops" law is **not** established (we couldn't get 3-hop in-state to measure it).

The real result is the **contrast**: externalised (CoT) reasoning reaches ≥6 hops (Result 7);
internalised (in-state) reasoning tops out around **2 hops** here. The token scratchpad is what buys
deep chains. *(Result 10 overturns the "~2 hops" ceiling: it was this recipe forgetting, not a limit.)*

## Result 10 — the "~2-hop cap" was a weaning artifact; the mechanism is a per-position register

Result 9 left one question — is the ~2-hop in-state ceiling a capacity limit or a training artifact?
A **mechanistic probe** (`experiments/probe_instate.py`) answers it, and the answer overturns Result 9.

**The probe.** For `rd_1x8` the single core block is looped `r` times, so a forward hook on it
captures the residual stream after every loop in one forward. We fit a linear probe to decode each
chain intermediate (m1, m2, …, value) from the residual at each answer-region position, per loop.
Validated on the solved 2-hop model (run14): m1 is decodable **1.00 at the `A` position**, the value
**0.99 at the pause**, both building with loops in lock-step with the loop-bound value curve — so
linear decodability faithfully reads out in-state computation.

**Result 9's stall was catastrophic forgetting, not capacity.** Probing the stalled run15 checkpoint,
*no* intermediate is decodable anywhere at either wean_k=1 or wean_k=2 (all ≈ chance; value ≈ 0.06) —
even though wean_k=1 *had* grokked earlier in training. Cause (in `--wean-mode hop`): at wean_k=2 only
the value position is supervised and shallower wean_k are never replayed, so the scaffold sub-circuits
(the `A`→m1 writer, the pause reader) lose all gradient, drift, and are destroyed. The ceiling was the
schedule erasing its own earlier solution.

**The fix — a replaying wean (`--wean-mode mixed`).** Each step: 50% at the current deepest wean_k,
50% replaying a uniformly-random shallower one, so the shallower chain is reinforced while the next hop
is introduced. Warm-started from the run12 CoT curriculum, 45k steps (run16). **3-hop internalises
fully**, and CoT is retained throughout (value@CoT = 1.00 the whole run):

| test-time r | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|-----|-----|-----|-----|-----|-----|-----|-----|
| 3-hop value @ full-internal | 0.05 | 0.05 | **0.84** | 0.98 | 0.99 | 0.99 | 0.98 | 0.94 |

**Equal-budget ablation (run17) isolates the cause.** Same 45k budget, only the schedule differs:
`hop` (no replay) leaves full-internal at chance (0.06) *and* collapses CoT (1.00 → 0.05); `mixed`
(replay) gives 0.99 with CoT held at 1.00. Replay is the sole difference — so it is the **schedule**,
not the budget. (Seed-1 replicates the whole picture: 3-hop → 0.99, same staircase, same register map.)

**The mechanism — one write-once register per token position.** Probing run16 at full-internal, r=8,
linear-decode accuracy by position (chance ≈ 0.008 over the 128-pool):

| position | m1 | m2 | value |
|---|---|---|---|
| query `e_t` | 0.01 | 0.01 | 0.01 |
| `A` (marker) | **1.00** | 0.01 | 0.01 |
| PAUSE₁ (`t0`) | 0.09 | **0.99** | 0.01 |
| PAUSE₂ (`t1`, value predicted here) | 0.02 | 0.18 | **0.99** |

Hop-1's result lives at the `A` position, hop-2's at the first pause, the value at the second — **both
derived intermediates held simultaneously, at different positions.** So there is no "can't hold two
partials" bottleneck; capacity is **sidestepped** by using positions as registers, not refuted.

**Causal, not just correlational (`experiments/patch_registers.py`).** For each row, run a *donor*
query (t′≠t) over the same banks, capture the looped block's output at one position per loop, and
overwrite that position in the *target* run. Fraction of rows whose predicted value equals the target
vs the donor chain (r=8): patching `A` flips it to the **donor's** value (target 0.99→0.001, donor
0.001→0.99); patching the first pause flips it too; patching the query does nothing. And patching `A`
on **loop 1 only** (loops 2..r run normally) *still* flips the answer — the register is **written once
and then held** (a fixed point), which is exactly why extra loops past r_min are harmless.

**N loops ≈ N hops.** The value at the answer-forming position appears one loop after its predecessor
register fills: m1@`A` by r=1, m2@pause by r=2, value by r=3. So the loop count needed = the number of
in-state hops: 2-hop-in-state solves at r≥2 (Result 8), full 3-hop-in-state at r≥3 (staircase r2 0.05
→ r3 0.84 → r4 0.98). Because run16 trains at randomized depth [1,8], the staircase is in-distribution,
**not** the fixed-depth mismatch of Result 6 — a genuine per-hop compute requirement.

**What this actually is, and its cost.** "Internalised" reasoning here is **latent pause-token
chain-of-thought**: it still spends one tape position per hidden hop (as a content-free PAUSE), only
the *content* is latent in the residual stream. So it is not scratchpad-free — it is O(N) positions
*and* r≥N loops, i.e. ~N/2× the compute of explicit CoT (which needs ~1–2 loops per token regardless
of N, Result 7). The large-N bottleneck is therefore compute and bank interference (~0.99^N, shared
with CoT), **not** a recurrent-state capacity limit — the partials are positional, so their *number*
is not bounded by `d_state`. This is consistent with SSMs' fixed-state limits (Merrill et al. 2024)
pushing sequential reasoning onto token positions rather than the hidden state.

## Concurrent work

The closest work is **Kohli, Parthasarathy, Sun & Yao, "Loop, Think, & Generalize: Implicit Reasoning
in Recurrent-Depth Transformers" (COLM 2026, arXiv:2604.07822)**, which independently studies a
recurrent-depth **transformer** on synthetic k-hop chains and reports logit-lens position-decoding,
activation patching, and loop-count-controls-hop-depth (super-linear depth extrapolation). This study
differs in substrate (a looped **state-space model**), in method (internalising *explicit CoT* via
soft-weaning, vs implicit-from-scratch training), in the **write-once / loop-invariance** causal test
(patching a single loop — not run there), and in the **schedule-artifact + replay** result (Result 10),
which has no analogue there. Also relevant: the recurrent-depth architecture (Geiping et al. 2025,
Huginn), stepwise CoT internalisation (Deng et al. 2024), pause/filler tokens (Goyal et al. 2023; Pfau
et al. 2024), and the SSM state-tracking limit (Merrill et al. 2024).

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
   token sequence.
6. **The core internalises reasoning via soft-weaning, and it is loop-bound (Result 8).** Replacing
   CoT tokens with content-free pause tokens moves a **2-hop** chain in-state (value 0.99), needing
   ≥2 loops (r=1: 0.06, r=2: 0.97) — the genuine in-core "loops buy reasoning depth" (run13's earlier
   failure was deleting the compute *position*, not a capability limit).
7. **The mechanism is a per-position write-once register; the apparent depth ceiling was a weaning
   artifact (Result 10).** Result 9's "~2-hop cap" was **catastrophic forgetting** from a schedule
   that stops supervising shallower steps (probe: the stalled model computed nothing in-state). A
   **replaying** wean internalises 3-hop fully (0.99, CoT retained), and an equal-budget ablation
   pins replay — not budget — as the cause. Each hop's result is held **write-once at its own
   pause-token position** (causally verified by donor-swap and single-loop patching), giving
   **N loops ≈ N hops** (2-hop r≥2, 3-hop r≥3). So "in-state" reasoning is **latent pause-token CoT**:
   O(N) positions with latent content, ~N/2× the compute of explicit CoT — not scratchpad-free, and
   its large-N bottleneck is compute + bank interference, **not** recurrent-state capacity (partials
   are positional). Consistent with the fixed-state limit of SSMs (Merrill et al. 2024).

## Caveats

Single seed per arm; K=16 toy vocabulary; fixed-depth training for the 2-hop runs
(random-depth stalls learning when low-`r` steps are computationally insufficient —
itself weak evidence for the counting-rule intuition). The single-hop scaling curve
is retrieval iteration, not reasoning-hop scaling; do not over-read it.

## Next experiments (rank order)

CoT supervision (Result 5), its airtight randomized-depth version (Result 6), the free-running
leak check (Result 6), the hop-curriculum to 6 hops (Result 7), 2-hop internalisation (Result 8),
and the probe + register mechanism + 3-hop internalisation + N-loops≈N-hops (Result 10) are done.
Remaining:

1. **How deep does in-state reasoning go — where is the real ceiling?** Result 10 removed the false
   ~2-hop ceiling (3-hop internalises with replay). Push the mixed wean to 4/5/6 hops (`--wean-depth
   4..6`) and find where it breaks — and whether the break is compute (r must track N), bank
   interference (~0.99^N, shared with CoT), or loop starvation at deep wean_k (raise `--rd-range`).
2. **Position-necessity test.** Drop the pause positions after internalising (tail `A v`, no register
   slots) — the write-once-register account predicts failure. This is the experiment that decides
   whether "in-state" is really positional/latent-CoT or something more.
3. **Push the CoT hop-curriculum higher** (--hops 8/10/12) for the externalised ceiling / 0.99^N erosion.
4. Beyond the toy: relax the typed-disjoint-vocab and clean-permutation assumptions toward realistic
   multi-hop, and vary K / d_model to see what sets the ceilings.

Reproduce: `experiments/diag_2hop.py` (see `--cot --typed-mid --distinct-vals
--derange --swap` flags, and `--mix-hop1` for the pre-CoT curriculum); arms in
`recurrent_depth/model.py`. Result JSON/checkpoints land in `results/` (gitignored).

Result 10 specifically:

```
# 3-hop internalisation with the replaying wean (warm-start from a CoT curriculum ckpt)
python experiments/diag_nhop.py --wean --wean-mode mixed --wean-depth 3 --hops 6 --k 16 \
    --load results/run12_curriculum.pt --finetune --random-depth --rd-range 1 8 \
    --steps 45000 --wean-start 2000 --wean-steps 18000 --ckpt results/run16_wean3_mixed.pt
# equal-budget ablation: swap --wean-mode mixed -> hop  (isolates schedule vs budget)
# probe the register mechanism (linear decode of each intermediate by position, per loop)
python experiments/probe_instate.py --load results/run16_wean3_mixed.pt --hops 3 --applies 8
# causal check (donor-swap + single-loop patching; forward passes only)
python experiments/patch_registers.py --load results/run16_wean3_mixed.pt --hops 3
```
