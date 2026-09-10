# Recurrent-depth Mamba

Can you make a small state-space model cheaper by *reusing* a layer instead of
stacking more of them?

That is the whole question here. The usual way to make a model deeper is to add
more layers, each with its own weights. But there is an old trick from the
transformer world (Universal Transformers, Looped Transformers): keep *one*
layer and just run it several times in a row. Same depth of computation, a
fraction of the parameters. I wanted to know whether that trick works on
**Mamba**, a state-space model, where as far as I could find it had not been
tried.

Short version: it works, with an honest asterisk. On an easy task you get about
**3.3× fewer parameters for roughly a quarter more training**. On a harder task
you can halve the parameters at *no* speed penalty, but you pay for it in
**reliability** instead. This README is the plain-language walkthrough; the code
is the technical version.

## What "recurrent depth" means

A normal 4-layer model looks like this: `block1 -> block2 -> block3 -> block4`,
four sets of weights. A recurrent-depth model reuses blocks: for example one
block applied four times, `block1 -> block1 -> block1 -> block1`. The
forward pass is just as deep (four steps of computation), but there is only one
set of weights to store and train, so it is about a quarter of the size.

I test five configurations, written as `(distinct blocks) x (times each is
applied)`:

| name | structure | what it probes |
|---|---|---|
| `baseline_4` | 4 blocks, each once | the normal 4-layer model (the reference) |
| `rd_1x4` | 1 block, applied 4× | maximum sharing: quarter the parameters, same depth |
| `rd_2x2` | 2 blocks, each 2× | the halfway point: half the parameters |
| `rd_4x2` | 4 blocks, each 2× | same parameters as baseline but double the compute |
| `rd_1x8` | 1 block, applied 8× | minimum parameters, maximum compute |

The last two exist to answer a specific question: is it the *parameters* that
matter, or the *amount of computation*? They add compute without adding
parameters, so if compute were the lever, they would help.

## Why it matters

If knowledge can be pushed out of a model's weights (which a companion repo,
[ssm-retrieval-efficiency](https://github.com/thevoidwolf/ssm-retrieval-efficiency),
argues for by showing retrieval is far cheaper than in-context recall), then the
model's core can be smaller. Recurrent depth is a second, independent lever on
the same goal: get the depth of a bigger model out of a smaller one. For anyone
training on a single GPU rather than a cluster, "same result, a third of the
parameters" is the difference between a plan that fits and one that does not. The
catch, below, is that the saving is not free, and being honest about the price is
the point of this repo.

## The task

A tiny model (256-wide Mamba-2 blocks, ~1.9M parameters at baseline) is trained
on a synthetic lookup task. Two versions:

- **inject** (easy): the one needed fact is placed right before the question. The
  model just has to copy the answer. Converges in tens of steps.
- **long** (harder): K facts are laid out inline and the model must stream them
  all, then answer a query about one of them (associative recall). Converges in
  thousands of steps, so it is a fairer stress test.

I measure one thing: **how many training steps it takes to reach 95% accuracy**
on fresh held-out samples (`step_to_95`). Fewer is better. All arms are run
across multiple random seeds, and I report every seed, because the seed spread
turns out to be the whole story on the hard task.

## Result 1: the easy task (inject)

Five arms, 3 seeds each, 300 steps. Every arm reaches 100% accuracy; the
question is only *how fast*.

| arm | params | step_to_95 (per seed) | mean | vs baseline |
|---|---:|---|---:|---|
| `baseline_4` | 1,863,008 | 45, 45, 45 | 45 | reference |
| `rd_2x2` | 998,960 | 50, 50, 50 | 50 | 1.9× fewer params, ~11% slower |
| `rd_1x4` | 566,936 | 55, 55, 60 | 57 | **3.3× fewer params, ~27% slower** |
| `rd_4x2` | 1,863,008 | 45, 45, 45 | 45 | same params, 2× compute: no change |
| `rd_1x8` | 566,936 | 60, 55, 60 | 58 | same params as rd_1x4, 2× compute: no change |

Two things fall out of this:

1. **The headline trade.** One block applied four times (`rd_1x4`) has 3.3×
   fewer parameters than the four-block baseline and learns the task in about a
   quarter more steps (57 vs 45). Parameters interpolate cleanly: the halfway
   arm (`rd_2x2`, half the parameters) sits neatly in the middle at 50 steps.
2. **Compute is not the lever, parameters are.** `rd_4x2` has the *same*
   parameter count as the baseline but does twice the computation, and it matches
   the baseline exactly (45). `rd_1x8` doubles the compute of `rd_1x4` at the
   same parameter count and gets nothing for it (58 vs 57). Extra passes over a
   fixed set of weights buy you nothing. **What you pay for is the parameters.**

## Result 2: the hard task (long), and the honest walk-back

This is where it gets interesting, and where an early version of this result was
too optimistic. On the long associative-recall task, 3 arms, 5 seeds each, 8000
steps:

| arm | params | step_to_95 (5 seeds) | converged | mean of converged |
|---|---:|---|:---:|---:|
| `baseline_4` | 1,863,008 | 5500, 3600, 3900, 3900, 6900 | **5/5** | 4760 |
| `rd_2x2` | 998,960 | 3800, 4200, n/c, 4900, n/c | **3/5** | 4300 |
| `rd_1x4` | 566,936 | 5300, n/c, 4100, n/c, 7900 | **3/5** | 5767 |

("n/c" means the arm never reached 95% accuracy within the 8000-step budget. The
two `rd_1x4` misses stalled near 0.41, barely above the noise floor; the two
`rd_2x2` misses got much closer, to 0.93 and 0.82.)

Here is the correction worth flagging, and it falls straight out of these five
seeds. **If you only looked at the first two seeds, `rd_2x2` looks like a free
lunch:** seeds 0 and 1 converge in 3800 and 4200 steps (mean 4000), *beating* the
baseline's first two seeds (5500 and 3600, mean 4550). Half the parameters and
faster. That is exactly the read an early two-seed check gave, and it is wrong.

The full five seeds dissolve it:

- On the seeds that converge, `rd_2x2` is within seed noise of the baseline (mean
  4300 vs 4760, if anything nominally faster). So there is **no real speed
  penalty at half the parameters**, but there is no free speed-up either. The
  two-seed sample had just caught the fast end of a wide spread.
- The real cost is reliability. At half the parameters, **2 of 5 seeds fail to
  reach 95%** within the budget (they top out at 0.93 and 0.82). The
  maximum-sharing `rd_1x4` arm (a quarter of the parameters) is worse still: also
  3/5, and its two failures collapse almost to chance (~0.41), not just short of
  the line.

So the honest claim is: **you can roughly halve the parameters at parity
convergence speed, but you take on a real reliability cost, on the order of two
seeds in five failing to converge here.** In practice that means pairing weight
sharing with a multi-seed protocol (train a few, keep the one that converges),
which eats back part of the saving. It is still a lever worth having, just not a
free one, and "it is actually faster" was a two-seed mirage I would rather show
than bury.

## Prior work this builds on

Weight-shared, iterated computation is well studied for transformers: Universal
Transformers, Looped Transformers, and the depth-recurrent-transformer line all
run one attention layer repeatedly. What is not well documented is doing the same
inside a **selective state-space model** like Mamba, where each block application
also evolves an internal recurrent state. This repo is a small, honest data point
on that specific question: it works, at a measurable and non-free cost.

## How to run it

Plain PyTorch. A CUDA GPU with the real Mamba-2 kernels reproduces the numbers; a
CPU falls back to a slow pure-PyTorch mixer that runs the pipeline for smoke
tests but does not reproduce the figures (it is a different kernel).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
# For the real numbers you also need the Mamba-2 CUDA kernels:
#   pip install mamba-ssm

# Smoke test (runs on a CPU in a minute):
python experiments/01_inject_depth_sweep.py --smoke

# The real sweeps (GPU):
python experiments/01_inject_depth_sweep.py --full   # 5 arms x 3 seeds, ~7 min
python experiments/02_long_depth_sweep.py --full     # 3 arms x 5 seeds, ~3 hours
```

Each run writes a JSON into `results/` with the full accuracy curve. The tables
above are read from those files.

## Layout

```
recurrent_depth/
  model.py        Mamba-2 block (real kernel or CPU fallback) + RecurrentDepthStack
  tasks.py        the inject and long lookup task samplers
  diagnostics.py  accuracy / logit-margin evaluation
  sweep.py        one training run of one arm (shared by both experiments)
  util.py         seeding, timing, result writing, LR schedule
experiments/
  01_inject_depth_sweep.py   the easy-task sweep
  02_long_depth_sweep.py     the hard-task sweep
results/          per-run JSON output (git-ignored; regenerated by the sweeps)
```

The backbone is the standard `mamba_ssm.Mamba2` block; every number above comes
from running the code in this repo on it.

MIT licensed. Built by one person on one GPU, and written to be readable by
someone who is not.
