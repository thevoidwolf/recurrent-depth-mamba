"""Regenerate the README figures from the result JSONs in results/.

    python docs/make_figs.py            # writes docs/figs/*.png

Needs the run{8,9,12,14} result JSONs (gitignored runtime outputs); the PNGs it
produces are committed so the README renders without them.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# colourblind-friendly (Wong)
BLUE, ORANGE, GREEN, VERM, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#999999"
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def load(name):
    return json.load(open(os.path.join(RES, name)))["final"]


def save(fig, name):
    fig.tight_layout()
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


RS = [1, 2, 3, 4, 6, 8, 12, 16]

# --- Fig 1: fixed-depth vs depth-matched CoT 2-hop (the correction) ---------
r8, r9 = load("run8_cot_rd1x8.json"), load("run9_cot_randdepth.json")
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(RS, [r8[str(r)]["cot_chain_acc"] for r in RS], "o-", color=ORANGE,
        label="trained at one fixed depth (8 loops)")
ax.plot(RS, [r9[str(r)]["cot_chain_acc"] for r in RS], "s-", color=BLUE,
        label="trained across depths (1–8 loops)")
ax.set(xlabel="thinking loops at test time (r)", ylabel="2-hop answer accuracy",
       title="Two-hop reasoning with a scratchpad\n(fixed-depth training exaggerates the 'more loops' effect)",
       ylim=(-0.03, 1.03))
ax.legend(loc="lower right", framealpha=0.9)
save(fig, "fig1_depth_matched.png")

# --- Fig 2: curriculum reaches 6-hop chains (with a scratchpad) --------------
r12 = load("run12_curriculum.json")
hs = [1, 2, 3, 4, 5, 6]
chain = [r12[f"h{h}"]["tf"]["chain"] for h in hs]
fig, ax = plt.subplots(figsize=(7, 4.4))
bars = ax.bar([str(h) for h in hs], chain, color=GREEN, width=0.6)
for b, c in zip(bars, chain):
    ax.text(b.get_x() + b.get_width() / 2, c + 0.015, f"{c:.2f}", ha="center", fontsize=10)
ax.set(xlabel="length of the reasoning chain (hops)", ylabel="whole-chain accuracy",
       title="A step-by-step curriculum reaches six-hop chains\n(reasoning written out as scratchpad tokens)",
       ylim=(0, 1.08))
save(fig, "fig2_curriculum.png")

# --- Fig 3: internalised reasoning is loop-bound ----------------------------
r14 = load("run14_wean2.json")
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(RS, [r14["wean0"][str(r)] for r in RS], "s-", color=BLUE,
        label="with a scratchpad (writes the step out)")
ax.plot(RS, [r14["wean1"][str(r)] for r in RS], "o-", color=VERM,
        label="in the model's state (no scratchpad)")
ax.axvline(2, color=GREY, ls="--", lw=1)
ax.set(xlabel="thinking loops at test time (r)", ylabel="2-hop answer accuracy",
       title="Reasoning in the model's state is 'loop-bound'\n(a 2-hop chain needs at least 2 loops)",
       ylim=(-0.03, 1.03))
ax.legend(loc="center right", framealpha=0.9)
save(fig, "fig3_loop_bound.png")

# --- Fig 4: depth ceiling — on the page vs in the head ----------------------
fig, ax = plt.subplots(figsize=(6.2, 4.4))
labels = ["with a scratchpad\n(reasoning as tokens)", "in the model's state\n(no scratchpad)"]
vals = [6, 3]
bars = ax.bar(labels, vals, color=[GREEN, VERM], width=0.55)
ax.text(0, 6.1, "≥6 (tested max,\nno wall hit)", ha="center", fontsize=9)
ax.text(1, 3.1, "≥3 (internalises;\nceiling not yet found)", ha="center", fontsize=9)
ax.set(ylabel="reasoning steps reliably solved", ylim=(0, 7.5),
       title="How far the reasoning goes\n(on the page vs in the model's head)")
save(fig, "fig4_ceiling.png")

# --- Fig 5: internalisation mechanism — N loops ≈ N hops + per-position registers ---
r14w = load("run14_wean2.json")        # 2-hop in-state = wean1 (WD-1)
r16w = load("run16_wean3_mixed.json")  # 3-hop in-state = wean2 (WD-1)
probe = json.load(open(os.path.join(RES, "probe_run16_wk2.json")))
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
# left: N loops ≈ N hops staircase (fully-internal value vs test-time loops)
axL.plot(RS, [r14w["wean1"][str(r)] for r in RS], "o-", color=BLUE, label="2-hop in the head")
axL.plot(RS, [r16w["wean2"][str(r)] for r in RS], "s-", color=VERM, label="3-hop in the head")
axL.axvline(2, color=BLUE, ls="--", lw=1, alpha=0.6)
axL.axvline(3, color=VERM, ls="--", lw=1, alpha=0.6)
axL.set(xlabel="thinking loops at test time (r)", ylabel="whole-chain accuracy",
        title="N loops ≈ N hops\n(2 hops need ≥2 loops; 3 hops need ≥3)", ylim=(-0.03, 1.03))
axL.legend(loc="lower right", framealpha=0.9)
# right: one register per step — linear decodability of each step's result by position (r=max)
dec = probe["decode"][str(max(probe["r_list"]))]
positions = ["A", "t0", "t1"]
poslabels = ["marker\n(step 1)", "think 1\n(step 2)", "think 2\n(step 3)"]
inters, interlabels, cols = ["m1", "m2", "v"], ["step-1 result", "step-2 result", "final answer"], [BLUE, ORANGE, GREEN]
x, w = np.arange(len(positions)), 0.26
for i, (it, lab, c) in enumerate(zip(inters, interlabels, cols)):
    axR.bar(x + (i - 1) * w, [dec[p][it] for p in positions], w, color=c, label=lab)
axR.set_xticks(x); axR.set_xticklabels(poslabels)
axR.set(ylabel="how readable it is (linear decode)", ylim=(0, 1.08),
        title="One register per step\n(each step's result sits at its own position)")
axR.legend(loc="upper center", framealpha=0.9, fontsize=9)
save(fig, "fig5_registers.png")

print("done")
