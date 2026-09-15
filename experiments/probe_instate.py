"""Mechanistic probe: WHY does in-state (weaned) reasoning cap at ~2 hops?

Loads a soft-weaned checkpoint (e.g. run15 = stalled full-3-hop-in-state, or run14 =
solved 2-hop-in-state) and LINEARLY decodes each chain intermediate (m1, m2, ..., v)
from the recurrent-depth residual stream at the answer region, as a function of the
loop count r. For the weight-shared rd_1x8 arm the single core block is looped r times,
so a forward hook on it captures the residual stream after every apply in ONE forward
(the loop is deterministic and depth-independent, so firing j == "state after j loops").

The capacity hypothesis (CONTINUE.md Lever 4): the recurrent state can hold ~2 chained
partials. For a full 3-hop-in-state chain [PAUSE PAUSE v] the value-forming position
must have computed m1 AND m2 in-state to look up v. Prediction if capacity-bound:
  - m1 decodable (first in-state hop fires),
  - m2 NOT decodable / collides with m1 (can't hold the second partial),
  - v fails.
Contrast: the SAME run15 model grokked wean_k=1 ([x1 PAUSE v]) where only ONE partial
(m2) is derived in-state (m1 is on the tape) -> holds one, fails at two.

Data construction is copied faithfully from experiments/diag_nhop.py (typed disjoint
128-pools per level, deranged perms, swapped bank order). Probe uses its OWN fresh
samples; VOCAB is inferred from the checkpoint's embedding so run14/run15 both load.

  # CPU wiring smoke (no GPU): needs a fallback-kernel checkpoint -- a real-kernel .pt will NOT
  # load into the CPU fallback (different mixer params). Mint one, then probe it, e.g.:
  #   python -c "import torch;from recurrent_depth.model import *;\
  #     torch.save(make_recurrent_model(903,'cpu',CoreConfig(n_layers=1,block='fallback',\
  #     norm_position='post',inject_input=True),1,8).state_dict(),'/tmp/fb.pt')"
  #   CUDA_VISIBLE_DEVICES="" python experiments/probe_instate.py --load /tmp/fb.pt \
  #     --hops 3 --device cpu --block fallback --n 256 --applies 4 --probe-steps 50

  # GPU (needs the real Mamba-2 kernel) -- run16 full 3-hop in-state (wean_k = h-1):
  python experiments/probe_instate.py --load results/run16_wean3_mixed.pt \
      --hops 3 --applies 8 --n 6000 --out results/probe_run16_wk2.json
  # ...and the within-model control (one derived intermediate held on the tape): add  --wean-k 1
"""
import sys, os, argparse, json, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, torch.nn.functional as F
from recurrent_depth.model import ARM_SPECS, CoreConfig, make_recurrent_model
from recurrent_depth.tasks import BOS, FACT, SEP, QTOK, ATOK, EOS, N_CONTROL

ap = argparse.ArgumentParser()
ap.add_argument("--load", required=True, help="weaned checkpoint (state_dict) to probe")
ap.add_argument("--arm", default="rd_1x8")
ap.add_argument("--hops", type=int, default=3, help="chain depth to probe (task h)")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--wean-k", type=int, default=None,
                help="# trailing intermediates replaced by PAUSE (default h-1 = full internal)")
ap.add_argument("--no-swap", action="store_true")
ap.add_argument("--applies", type=int, default=8, help="max loop count; probe reads states after 1..R")
ap.add_argument("--r-list", type=int, nargs="*", default=None,
                help="which loop counts to report (default: 1,2,3,4,6,8 clipped to --applies)")
ap.add_argument("--n", type=int, default=6000, help="probe samples (train+test)")
ap.add_argument("--batch", type=int, default=500)
ap.add_argument("--probe-steps", type=int, default=400)
ap.add_argument("--probe-lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--device", default="cuda")
ap.add_argument("--block", default="mamba2")
ap.add_argument("--out", default=None)
args = ap.parse_args()

device = args.device
torch.manual_seed(args.seed)
K, H, P = args.k, args.hops, 128
SWAP = not args.no_swap
PAUSE = BOS
WEAN_K = args.wean_k if args.wean_k is not None else (H - 1)   # h-1 = fully internal


def _level_off(l):
    return N_CONTROL + l * P


def _deranged(k, gen):
    for _ in range(50):
        p = torch.randperm(k, generator=gen)
        if not bool((p == torch.arange(k)).any()):
            return p
    return p


# --- data: faithful copy of diag_nhop.sample_raw / pack (parameterised, no globals) ---
def sample_raw(B, gen, h):
    nodes = [torch.stack([torch.randperm(P, generator=gen)[:K] + _level_off(l) for _ in range(B)])
             for l in range(h + 1)]
    perms = [None] + [torch.stack([_deranged(K, gen) for _ in range(B)]) for _ in range(h)]
    t = torch.randint(K, (B,), generator=gen)
    return nodes, perms, t


def pack(nodes, perms, t, h, swap=SWAP, wean_k=0):
    """Returns (seq, ans_pos, inter) where inter[i] = m_{i+1} tokens [B] (level i+1); inter[-1]=v.
    Tail (CoT): A x1 x2 .. x_{h-1} v EOS; trailing wean_k intermediates -> PAUSE. ans_pos = v pos."""
    B = nodes[0].shape[0]
    banks = []
    for i in range(1, h + 1):
        banks.append((nodes[i - 1], nodes[i].gather(1, perms[i])))
    order = list(reversed(banks)) if swap else banks
    parts = [torch.full((B, 1), FACT, dtype=torch.long)]
    for bi, (src, dst) in enumerate(order):
        for k in range(K):
            parts.append(src[:, k:k + 1]); parts.append(dst[:, k:k + 1])
            if k < K - 1:
                parts.append(torch.full((B, 1), SEP, dtype=torch.long))
        if bi < len(order) - 1:
            parts.append(torch.full((B, 1), SEP, dtype=torch.long))
    idx = t
    inter = []
    for i in range(1, h + 1):
        idx = perms[i].gather(1, idx.unsqueeze(1)).squeeze(1)
        inter.append(nodes[i].gather(1, idx.unsqueeze(1)))       # [B,1]
    q = nodes[0].gather(1, t.unsqueeze(1))
    parts += [torch.full((B, 1), QTOK, dtype=torch.long), q, torch.full((B, 1), ATOK, dtype=torch.long)]
    tail = list(inter)                                            # [x1..x_{h-1}, v]
    for j in range(h - 1 - wean_k, h - 1):                        # trailing wean_k -> PAUSE
        tail[j] = torch.full((B, 1), PAUSE, dtype=torch.long)
    parts += tail
    parts.append(torch.full((B, 1), EOS, dtype=torch.long))
    seq = torch.cat(parts, 1)
    inter_tok = torch.cat(inter, 1)                               # [B,h] ground-truth m1..m_{h-1},v
    return seq, seq.shape[1] - 2, inter_tok


# --- build model with VOCAB inferred from the checkpoint embedding ---
sd = torch.load(args.load, map_location=device)
VOCAB = sd["embed.weight"].shape[0]
spec = ARM_SPECS[args.arm]
assert spec["n_distinct"] == 1, "hook logic assumes a single looped block (rd_1x*)"
cfg = CoreConfig(d_model=256, n_layers=spec["n_distinct"], d_state=64, d_conv=4, expand=2,
                 headdim=64, block=args.block, norm_position="post", inject_input=True)
model = make_recurrent_model(VOCAB, device, cfg, spec["n_distinct"], spec["applies_per_block"])
model.load_state_dict(sd)
model.eval()
print(f"loaded {args.load}  vocab={VOCAB}  arm={args.arm}  hops={H} K={K} swap={SWAP} "
      f"wean_k={WEAN_K} (full-internal={WEAN_K==H-1})  applies={args.applies} device={device} block={args.block}",
      flush=True)

R_max = args.applies
r_list = args.r_list or [r for r in (1, 2, 3, 4, 6, 8, 12, 16) if r <= R_max]

# --- capture: single forward at applies=R_max; hook the looped block, keep tail cols ---
# tail offsets from ans_pos: q(-(h+1)), A(-h), t0..t_{h-2} intermediates, v(0)
offsets = [-(H + 1), -H] + [-(H - 1 - j) for j in range(H)]      # q, A, t0..t_{h-1}(=v)
off_names = ["q", "A"] + [f"t{j}" for j in range(H)]             # t{h-1} == value slot


@torch.no_grad()
def capture(seq):
    """Run applies=R_max; return acts[a] = residual stream after a loops at the tail offsets.
    acts shape: dict a(1..R_max) -> tensor [B, len(offsets), D]."""
    fired = []
    blk = model.core.blocks[0]
    ans_pos = seq.shape[1] - 2
    cols = [ans_pos + o for o in offsets]
    def hook(_m, _inp, out):
        fired.append(out[:, cols, :].detach())                  # [B, n_off, D]
    hd = blk.register_forward_hook(hook)
    try:
        model(seq, applies=R_max)
    finally:
        hd.remove()
    assert len(fired) == R_max, f"hook fired {len(fired)} != {R_max}"
    return {a + 1: fired[a] for a in range(R_max)}


# accumulate activations + labels over minibatches
gen = torch.Generator().manual_seed(args.seed)
acc = {a: [] for a in range(1, R_max + 1)}
labels = []          # [N, H] intermediate tokens m1..v
val_pred_raw = {a: [] for a in range(1, R_max + 1)}   # model value-slot argmax (level-H) per r
done = 0
D = cfg.d_model
norm_f, head = model.core.norm_f, model.head
val_off_idx = off_names.index(f"t{H-1}")                 # value slot column (== ans_pos-1 producer? see below)
# NOTE: value is PREDICTED at ans_pos-1 (the last pause, t_{H-2}); its head logits -> v.
pred_off_idx = off_names.index(f"t{H-2}") if H >= 2 else off_names.index("A")
val_lo, val_hi = _level_off(H), _level_off(H) + P
while done < args.n:
    B = min(args.batch, args.n - done)
    raw = sample_raw(B, gen, H)
    seq, ans_pos, inter = pack(*raw, h=H, wean_k=WEAN_K)
    seq = seq.to(device)
    caps = capture(seq)
    for a in range(1, R_max + 1):
        acc[a].append(caps[a])
        # model's own value prediction after a loops: head(norm_f(x_a))[:, pred position]
        h_pred = caps[a][:, pred_off_idx, :]                 # [B,D] residual at the value-producing pos
        lg = head(norm_f(h_pred))[:, val_lo:val_hi]
        val_pred_raw[a].append(lg.argmax(-1) + val_lo)
    labels.append(inter.to(device))
    done += B
acts = {a: torch.cat(acc[a], 0) for a in range(1, R_max + 1)}    # [N, n_off, D]
labels = torch.cat(labels, 0)                                    # [N, H]
val_pred = {a: torch.cat(val_pred_raw[a], 0) for a in range(1, R_max + 1)}
N = labels.shape[0]
ntr = int(N * 0.75)
print(f"captured N={N}  tail cols={off_names}  R_max={R_max}  D={D}", flush=True)

# model value accuracy vs r (reproduces wean_eval on these samples)
v_true = labels[:, H - 1]
model_val_acc = {a: float((val_pred[a][ntr:] == v_true[ntr:]).float().mean()) for a in r_list}
print("model value acc (in-state, full chain) vs r: " +
      " ".join(f"r{a}:{model_val_acc[a]:.3f}" for a in r_list), flush=True)


def fit_probe(X, y, n_classes, lo):
    """Linear probe X[N,D] -> class in [lo, lo+n_classes); y are absolute token ids.
    Returns test top-1 accuracy. Trains on first ntr rows, tests on rest."""
    yc = (y - lo).long()
    W = torch.nn.Linear(D, n_classes).to(device)
    opt = torch.optim.Adam(W.parameters(), lr=args.probe_lr)
    Xtr, ytr = X[:ntr], yc[:ntr]
    for _ in range(args.probe_steps):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(W(Xtr), ytr)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = W(X[ntr:]).argmax(-1)
    return float((pred == yc[ntr:]).float().mean())


# --- probe grid: for each r, each tail position, decode each intermediate level ---
# report[r][pos][mi] = decode acc of intermediate mi (level mi+1) from residual at pos
report = {}
t0 = time.time()
for a in r_list:
    report[a] = {}
    for oi, oname in enumerate(off_names):
        X = acts[a][:, oi, :].float()
        row = {}
        for mi in range(H):                                      # m1..m_{H-1}, v = level mi+1
            lvl = mi + 1
            lo = _level_off(lvl)
            y = labels[:, mi]
            row[f"m{mi+1}" if mi < H - 1 else "v"] = fit_probe(X, y, P, lo)
        report[a][oname] = row
    print(f"[{time.time()-t0:5.0f}s] r={a:2d} probed", flush=True)

# --- pretty print: focus on the value-producing position (last pause) ---
inter_names = [f"m{i+1}" for i in range(H - 1)] + ["v"]
chance = 1.0 / P
print(f"\n=== linear decodability (test top-1; chance≈{chance:.3f} over the 128-pool, ~{1/K:.3f} among K present) ===")
print(f"KEY position = '{off_names[pred_off_idx]}' (value is predicted here). Decode of each intermediate vs r:")
hdr = "  r  | " + " ".join(f"{nm:>6}" for nm in inter_names) + " | model_v"
print(hdr); print("  " + "-" * (len(hdr) - 2))
for a in r_list:
    row = report[a][off_names[pred_off_idx]]
    cells = " ".join(f"{row[nm]:6.3f}" for nm in inter_names)
    print(f"  {a:2d} | {cells} |  {model_val_acc[a]:.3f}")

print("\n=== full position x intermediate map at r=max ===")
amax = r_list[-1]
print("  pos  | " + " ".join(f"{nm:>6}" for nm in inter_names))
for oname in off_names:
    row = report[amax][oname]
    print(f"  {oname:>4} | " + " ".join(f"{row[nm]:6.3f}" for nm in inter_names))

if args.out:
    json.dump({"args": vars(args), "vocab": VOCAB, "off_names": off_names,
               "pred_pos": off_names[pred_off_idx], "r_list": r_list,
               "model_value_acc": model_val_acc, "decode": report,
               "chance_pool": chance, "chance_present": 1.0 / K}, open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
