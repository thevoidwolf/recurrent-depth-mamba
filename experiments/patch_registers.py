"""Causal activation-patching check of the position-register mechanism (forward passes only).

For each row build TWO sequences over the SAME banks: target query t and donor query t' != t.
Run the donor at applies=R and record the looped block's output at one tail column per loop.
Run the target with a forward hook that OVERWRITES that column's block output with the donor's
at every loop. If the column is a genuine register that carries hop-k's result downstream, the
value predicted at the last PAUSE flips from v(t) to v(t') (chain of the donor).
Since perms are bijections and nodes distinct within a row, v(t') != v(t) always.
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from recurrent_depth.model import ARM_SPECS, CoreConfig, make_recurrent_model
from recurrent_depth.tasks import BOS, FACT, SEP, QTOK, ATOK, EOS, N_CONTROL

ap = argparse.ArgumentParser()
ap.add_argument("--load", default="results/run16_wean3_mixed.pt")
ap.add_argument("--hops", type=int, default=3)
ap.add_argument("--wean-k", type=int, default=None)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--batch", type=int, default=500)
ap.add_argument("--r-list", type=int, nargs="*", default=[3, 4, 8])
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--device", default="cuda")
ap.add_argument("--block", default="mamba2")
ap.add_argument("--out", default=None)
args = ap.parse_args()
device = args.device
K, H, P = args.k, args.hops, 128
PAUSE = BOS
WEAN_K = args.wean_k if args.wean_k is not None else H - 1


def _level_off(l):
    return N_CONTROL + l * P


def _deranged(k, gen):
    for _ in range(50):
        p = torch.randperm(k, generator=gen)
        if not bool((p == torch.arange(k)).any()):
            return p
    return p


def sample_raw(B, gen, h):
    nodes = [torch.stack([torch.randperm(P, generator=gen)[:K] + _level_off(l) for _ in range(B)])
             for l in range(h + 1)]
    perms = [None] + [torch.stack([_deranged(K, gen) for _ in range(B)]) for _ in range(h)]
    t = torch.randint(K, (B,), generator=gen)
    return nodes, perms, t


def pack(nodes, perms, t, h, wean_k):
    B = nodes[0].shape[0]
    banks = [(nodes[i - 1], nodes[i].gather(1, perms[i])) for i in range(1, h + 1)]
    order = list(reversed(banks))                                   # swapped bank order
    parts = [torch.full((B, 1), FACT, dtype=torch.long)]
    for bi, (src, dst) in enumerate(order):
        for k in range(K):
            parts.append(src[:, k:k + 1]); parts.append(dst[:, k:k + 1])
            if k < K - 1:
                parts.append(torch.full((B, 1), SEP, dtype=torch.long))
        if bi < len(order) - 1:
            parts.append(torch.full((B, 1), SEP, dtype=torch.long))
    idx = t; inter = []
    for i in range(1, h + 1):
        idx = perms[i].gather(1, idx.unsqueeze(1)).squeeze(1)
        inter.append(nodes[i].gather(1, idx.unsqueeze(1)))
    q = nodes[0].gather(1, t.unsqueeze(1))
    parts += [torch.full((B, 1), QTOK, dtype=torch.long), q, torch.full((B, 1), ATOK, dtype=torch.long)]
    tail = list(inter)
    for j in range(h - 1 - wean_k, h - 1):
        tail[j] = torch.full((B, 1), PAUSE, dtype=torch.long)
    parts += tail
    parts.append(torch.full((B, 1), EOS, dtype=torch.long))
    seq = torch.cat(parts, 1)
    return seq, seq.shape[1] - 2, torch.cat(inter, 1)


sd = torch.load(args.load, map_location=device)
VOCAB = sd["embed.weight"].shape[0]
spec = ARM_SPECS["rd_1x8"]
cfg = CoreConfig(d_model=256, n_layers=1, d_state=64, d_conv=4, expand=2, headdim=64,
                 block=args.block, norm_position="post", inject_input=True)
model = make_recurrent_model(VOCAB, device, cfg, 1, spec["applies_per_block"])
model.load_state_dict(sd); model.eval()
blk = model.core.blocks[0]
print(f"loaded {args.load} vocab={VOCAB} H={H} wean_k={WEAN_K} r_list={args.r_list} n={args.n}", flush=True)

# tail columns relative to ans_pos (value slot): q=-(H+1), A=-H, t_j = -(H-1-j)
col_names = ["q", "A"] + [f"t{j}" for j in range(H - 1)]
col_offs = [-(H + 1), -H] + [-(H - 1 - j) for j in range(H - 1)]
pred_off = -1                                       # last PAUSE (value predicted here)
val_lo, val_hi = _level_off(H), _level_off(H) + P
gen = torch.Generator().manual_seed(args.seed)


@torch.no_grad()
def run_capture(seq, R, col):
    fired = []
    hd = blk.register_forward_hook(lambda m, i, o: fired.append(o[:, col, :].clone()))
    try:
        model(seq, applies=R)
    finally:
        hd.remove()
    assert len(fired) == R
    return fired


@torch.no_grad()
def run_patched(seq, R, col, donor_acts, loops=None):
    """Overwrite block output at column `col` with donor_acts[j] at loop j (all loops unless `loops`)."""
    state = {"j": 0}
    def hook(m, i, o):
        j = state["j"]; state["j"] += 1
        if loops is None or j in loops:
            o = o.clone(); o[:, col, :] = donor_acts[j]
        return o
    hd = blk.register_forward_hook(hook)
    try:
        logits = model(seq, applies=R)
    finally:
        hd.remove()
    assert state["j"] == R
    return logits


results = {}
for R in args.r_list:
    tot = {nm: {"v_target": 0, "v_donor": 0} for nm in col_names + ["none", "bank_mid", "A_loop1_only", "A+t0"]}
    n = 0
    for _ in range(args.n // args.batch):
        nodes, perms, t = sample_raw(args.batch, gen, H)
        delta = torch.randint(1, K, (args.batch,), generator=gen)
        t2 = (t + delta) % K                                          # donor query, same banks
        seq_t, ans_pos, inter_t = pack(nodes, perms, t, H, WEAN_K)
        seq_d, _, inter_d = pack(nodes, perms, t2, H, WEAN_K)
        seq_t, seq_d = seq_t.to(device), seq_d.to(device)
        v_t, v_d = inter_t[:, -1].to(device), inter_d[:, -1].to(device)
        assert not bool((v_t == v_d).any())
        p = ans_pos + pred_off
        def score(logits, nm):
            pred = logits[:, p, val_lo:val_hi].argmax(-1) + val_lo
            tot[nm]["v_target"] += int((pred == v_t).sum()); tot[nm]["v_donor"] += int((pred == v_d).sum())
        score(model(seq_t, applies=R), "none")                       # no patch: sanity
        for nm, off in zip(col_names, col_offs):
            col = ans_pos + off
            score(run_patched(seq_t, R, col, run_capture(seq_d, R, col)), nm)
        # control: a mid-bank column (inside bank 2's pair list), same donor protocol
        col = ans_pos // 2
        score(run_patched(seq_t, R, col, run_capture(seq_d, R, col)), "bank_mid")
        # A patched at loop 1 only (register re-written each loop? then later loops restore v_t)
        colA = ans_pos - H
        score(run_patched(seq_t, R, colA, run_capture(seq_d, R, colA), loops={0}), "A_loop1_only")
        # A and t0 both patched: should flip as strongly as A alone
        colT0 = ans_pos - (H - 1)
        dA, dT0 = run_capture(seq_d, R, colA), run_capture(seq_d, R, colT0)
        state = {"j": 0}
        def hook2(m, i, o):
            j = state["j"]; state["j"] += 1
            o = o.clone(); o[:, colA, :] = dA[j]; o[:, colT0, :] = dT0[j]; return o
        hd = blk.register_forward_hook(hook2)
        try:
            score(model(seq_t, applies=R), "A+t0")
        finally:
            hd.remove()
        n += args.batch
    results[R] = {nm: {k: v / n for k, v in d.items()} for nm, d in tot.items()}
    print(f"\n=== r={R}  (n={n}; fraction of rows whose predicted value == target chain v(t) / donor chain v(t'))")
    for nm in ["none"] + col_names + ["bank_mid", "A_loop1_only", "A+t0"]:
        d = results[R][nm]
        print(f"  patch {nm:>12}: v_target={d['v_target']:.3f}  v_donor={d['v_donor']:.3f}")
if args.out:
    json.dump({"args": vars(args), "results": results}, open(args.out, "w"), indent=1)
    print("wrote", args.out)
