"""N-hop chain-of-thought scaling: how many hops before a weight-shared looped Mamba breaks?

Generalises diag_2hop.py to an N-hop retrieval chain  e_t -> x1 -> x2 -> ... -> v  with
CoT supervision (emit every intermediate, then the value). Construction (the clean,
shortcut-free one from the 2-hop study, baked in):
  - typed levels: level l (0=query entities .. H=values) has its own disjoint 128-token
    pool, so a key can only match its own bank (no cross-bank key collision);
  - deranged permutations per hop (no self-loop shortcut);
  - swapped bank order by default (value bank first, query-adjacent bank last), which the
    2-hop study found is the SSM-favourable layout.

Reports, per test-time loop count r: per-hop and chain accuracy, both TEACHER-FORCED and
FREE-RUNNING (leak-free: the model generates each intermediate and conditions the next hop
on its own generation; the true intermediates/value are never in the input).

  # CPU wiring smoke (no GPU):
  CUDA_VISIBLE_DEVICES="" python experiments/diag_nhop.py --hops 3 --device cpu --block fallback --k 4 --steps 6 --eval-every 6 --depth-eval 1 2

  # GPU 3-hop (randomized-depth CoT, the airtight recipe):
  python experiments/diag_nhop.py --hops 3 --k 16 --steps 12000 --sched-total 12000 \
      --random-depth --rd-range 1 8 --depth-eval 1 2 3 4 6 8 12 16 \
      --out results/run11_nhop3.json --ckpt results/run11_nhop3.pt
"""
import sys, os, argparse, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, torch.nn.functional as F
from recurrent_depth.model import ARM_SPECS, CoreConfig, make_recurrent_model
from recurrent_depth.tasks import FACT, SEP, QTOK, ATOK, EOS, N_CONTROL
from recurrent_depth.util import cosine_lr, seed_all

ap = argparse.ArgumentParser()
ap.add_argument("--hops", type=int, default=3)
ap.add_argument("--arm", default="rd_1x8")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--sched-total", type=int, default=12000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--random-depth", action="store_true")
ap.add_argument("--rd-range", type=int, nargs=2, default=[1, 8])
ap.add_argument("--no-cot", action="store_true", help="disable CoT (emit only the value)")
ap.add_argument("--no-swap", action="store_true", help="banks in forward order (default: value bank first)")
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--depth-eval", type=int, nargs="*", default=[1, 2, 3, 4, 6, 8, 12, 16])
# curriculum: ramp the hop count 1 -> --hops, advancing when the current depth's chain acc
# clears --advance-thr. --hops is then the MAX depth (N_max). Uses a flat LR (phases unknown).
ap.add_argument("--curriculum", action="store_true", help="ramp hop count from --curriculum-start to --hops")
ap.add_argument("--curriculum-start", type=int, default=1)
ap.add_argument("--advance-thr", type=float, default=0.9, help="chain acc at H_cur needed to advance a hop")
ap.add_argument("--advance-patience", type=int, default=6000,
                help="if H_cur hasn't advanced in this many steps, call it the breaking point and stop")
ap.add_argument("--device", default="cuda")
ap.add_argument("--block", default="mamba2", help="'mamba2' (GPU) or 'fallback' (CPU smoke)")
ap.add_argument("--out", default=None)
ap.add_argument("--ckpt", default=None)
ap.add_argument("--ckpt-every", type=int, default=0, help="also save --ckpt every N steps (crash-safety); 0=only at end")
ap.add_argument("--load", default=None)
args = ap.parse_args()

device = args.device
torch.backends.cuda.matmul.allow_tf32 = True
seed_all(args.seed)
K, H, P = args.k, args.hops, 128
COT = not args.no_cot
SWAP = not args.no_swap
VOCAB = N_CONTROL + (H + 1) * P          # control + one 128-token pool per level 0..H


def _level_off(l):
    return N_CONTROL + l * P


def _deranged(k, gen):
    """A permutation of {0..k-1} with no fixed point (rejection; trivial for k>=4)."""
    for _ in range(50):
        p = torch.randperm(k, generator=gen)
        if not bool((p == torch.arange(k)).any()):
            return p
    return p


def sample_raw(B, gen, h=H):
    # nodes[l]: [B,K] tokens at level l (distinct within a row); perms[i]: [B,K] deranged map for hop i
    nodes = [torch.stack([torch.randperm(P, generator=gen)[:K] + _level_off(l) for _ in range(B)])
             for l in range(h + 1)]
    perms = [None] + [torch.stack([_deranged(K, gen) for _ in range(B)]) for _ in range(h)]
    t = torch.randint(K, (B,), generator=gen)
    return nodes, perms, t


def pack(nodes, perms, t, h=H, cot=COT, swap=SWAP):
    """Banks: bank_i pairs (nodes[i-1][j], nodes[i][perm_i[j]]).  Query: e_t; chain follows
    perm_1..perm_H. CoT tail: A x1 x2 .. x_{H-1} v EOS  (value 2nd-from-last => ans_pos=len-2).
    Non-CoT tail: A v EOS."""
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
    # cumulative chain: idx_0 = t; idx_i = perm_i[idx_{i-1}]; intermediate/value = nodes[i][idx_i]
    idx = t
    inter = []
    for i in range(1, h + 1):
        idx = perms[i].gather(1, idx.unsqueeze(1)).squeeze(1)
        inter.append(nodes[i].gather(1, idx.unsqueeze(1)))       # [B,1]
    q = nodes[0].gather(1, t.unsqueeze(1))
    parts += [torch.full((B, 1), QTOK, dtype=torch.long), q, torch.full((B, 1), ATOK, dtype=torch.long)]
    parts += inter if cot else [inter[-1]]
    parts.append(torch.full((B, 1), EOS, dtype=torch.long))
    seq = torch.cat(parts, 1)
    return seq, seq.shape[1] - 2                                  # ans_pos = value position


# number of supervised answer tokens at depth h: h (x1..x_{h-1}, v) for CoT, else 1 (value)
def _na(h):
    return h if COT else 1


@torch.no_grad()
def tf_eval(r, gen, h=H, n_batches=2, batch=512):
    """Teacher-forced: score each answer token given the true previous ones (depth h)."""
    na = _na(h)
    model.eval()
    n = 0; hop_c = [0] * na; chain_c = 0
    for _ in range(n_batches):
        raw = sample_raw(batch, gen, h); seq, ans_pos = pack(*raw, h=h); seq = seq.to(device)
        logits = model(seq, applies=r)
        allc = None
        for j in range(na):
            p = ans_pos - na + j                                  # logits at p predict token p+1
            c = logits[:, p].argmax(-1) == seq[:, p + 1]
            hop_c[j] += c.sum().item()
            allc = c if allc is None else (allc & c)
        chain_c += allc.sum().item(); n += batch
    model.train()
    return {"hop_acc": [h_ / n for h_ in hop_c], "chain": chain_c / n}


@torch.no_grad()
def fr_eval(r, gen, h=H, n_batches=2, batch=512):
    """Free-running (leak-free), depth h: build the prefix up to the A token (no answer tokens),
    then generate each answer token and condition the next on the model's OWN generation."""
    na = _na(h)
    model.eval()
    n = 0; hop_c = [0] * na; chain_c = 0
    for _ in range(n_batches):
        raw = sample_raw(batch, gen, h); seq, ans_pos = pack(*raw, h=h); seq = seq.to(device)
        atok_pos = ans_pos - na                                   # position of the A token
        cur = seq[:, :atok_pos + 1]                               # prefix ending at A
        tgts = [seq[:, ans_pos - na + 1 + j] for j in range(na)]  # true x1..v
        allc = None
        for j in range(na):
            nxt = model(cur, applies=r)[:, -1].argmax(-1)
            c = nxt == tgts[j]
            hop_c[j] += c.sum().item()
            allc = c if allc is None else (allc & c)
            cur = torch.cat([cur, nxt.unsqueeze(1)], 1)
        chain_c += allc.sum().item(); n += batch
    model.train()
    return {"hop_acc": [h_ / n for h_ in hop_c], "chain": chain_c / n}


def _fmt(d):
    return "hops=[" + " ".join(f"{h:.2f}" for h in d["hop_acc"]) + f"] chain={d['chain']:.3f}"


spec = ARM_SPECS[args.arm]
cfg = CoreConfig(d_model=256, n_layers=spec["n_distinct"], d_state=64, d_conv=4, expand=2,
                 headdim=64, block=args.block, norm_position="post", inject_input=True)
model = make_recurrent_model(VOCAB, device, cfg, spec["n_distinct"], spec["applies_per_block"])
print(f"arm={args.arm} hops={H} K={K} vocab={VOCAB} cot={COT} swap={SWAP} "
      f"random_depth={args.random_depth} steps={args.steps} device={device} block={args.block}", flush=True)

if args.load:
    model.load_state_dict(torch.load(args.load, map_location=device))
else:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    train_gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 10_000)
    depth_gen = torch.Generator().manual_seed(args.seed + 20_000)
    t0 = time.time()
    H_cur = args.curriculum_start if args.curriculum else H     # current training depth
    last_advance = 0
    reached = H_cur
    for step in range(args.steps):
        # curriculum uses a flat LR after warmup (phase boundaries unknown); else cosine.
        if args.curriculum:
            lr = 3e-4 * min(1.0, (step + 1) / 50)
        else:
            lr = cosine_lr(step, warmup=50, total=args.sched_total, base=3e-4, floor=3e-5)
        for pg in opt.param_groups:
            pg["lr"] = lr
        h = H_cur
        na = _na(h)
        raw = sample_raw(args.batch, train_gen, h); seq, ans_pos = pack(*raw, h=h); seq = seq.to(device)
        r = int(torch.randint(args.rd_range[0], args.rd_range[1] + 1, (1,), generator=depth_gen).item()) \
            if args.random_depth else None
        logits = model(seq, applies=r)
        loss = sum(F.cross_entropy(logits[:, ans_pos - na + j], seq[:, ans_pos - na + j + 1])
                   for j in range(na)) / na
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            d = tf_eval(spec["applies_per_block"], eval_gen, h=h)
            tag = f"H_cur={h} " if args.curriculum else ""
            print(f"step {step+1:5d} loss {loss.item():.3f} [{time.time()-t0:5.0f}s] {tag}{_fmt(d)}", flush=True)
            if args.curriculum:
                if d["chain"] >= args.advance_thr and H_cur < H:
                    H_cur += 1; reached = H_cur; last_advance = step + 1
                    print(f"  --> chain>= {args.advance_thr} at H={h}; ADVANCE curriculum to H_cur={H_cur} (step {step+1})", flush=True)
                elif step + 1 - last_advance >= args.advance_patience:
                    print(f"  --> no advance for {args.advance_patience} steps at H_cur={H_cur}: BREAKING POINT. stopping.", flush=True)
                    break
        if args.ckpt and args.ckpt_every and (step + 1) % args.ckpt_every == 0:
            torch.save(model.state_dict(), args.ckpt)      # crash-safety: overwrite periodically
            print(f"  [ckpt] saved at step {step+1} (H_cur={H_cur}); resume via --load {args.ckpt} --curriculum-start {H_cur}", flush=True)
    if args.ckpt:
        torch.save(model.state_dict(), args.ckpt)
    if args.curriculum:
        print(f"\ncurriculum reached H_cur={reached} of max {H}", flush=True)

print("\n=== final: per-hop + chain vs test-time r (TF = teacher-forced, FR = free-running/leak-free) ===")
final = {}
eval_gen = torch.Generator().manual_seed(args.seed + 30_000)
if args.curriculum:
    # per-depth profile at the trained loop count, plus an r-sweep at the deepest trained depth
    for h in range(1, H + 1):
        tf = tf_eval(spec["applies_per_block"], eval_gen, h=h, n_batches=4)
        fr = fr_eval(spec["applies_per_block"], eval_gen, h=h, n_batches=4)
        final[f"h{h}"] = {"tf": tf, "fr": fr}
        print(f"depth h={h:2d}  TF {_fmt(tf)}   |   FR {_fmt(fr)}")
    for r in args.depth_eval:
        tf = tf_eval(r, eval_gen, h=H, n_batches=4)
        final[f"h{H}_r{r}"] = {"tf": tf}
        print(f"[h={H}] r={r:2d}  TF {_fmt(tf)}")
else:
    for r in args.depth_eval:
        tf = tf_eval(r, eval_gen, h=H, n_batches=4)
        fr = fr_eval(r, eval_gen, h=H, n_batches=4)
        final[r] = {"tf": tf, "fr": fr}
        print(f"r={r:2d}  TF {_fmt(tf)}   |   FR {_fmt(fr)}")
if args.out:
    json.dump({"args": vars(args), "final": final}, open(args.out, "w"), indent=1)
