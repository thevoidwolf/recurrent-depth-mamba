"""Plateau diagnostic for long_2hop: retrain to the plateau, then classify every
prediction (correct / own-value v_t shortcut / reverse-hop / other present value /
absent value / entity), rank of v_ans and v_t, recency histogram, fixed-point split.

Variants (off by default, replicate the repo's task exactly):
  --derange     force pi(t) != t for the queried t (removes the 1/K reward for the v_t shortcut)
  --typed-mid   hop-1 destinations come from a DISJOINT mid-entity vocab (e -> m -> v):
                the query key e_t can never match a hop-2 key, removing key collision
"""
import sys, os, argparse, time, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, torch.nn.functional as F
from recurrent_depth.model import ARM_SPECS, CoreConfig, make_recurrent_model
from recurrent_depth import tasks
from recurrent_depth.tasks import FACT, SEP, QTOK, ATOK, EOS, N_CONTROL
from recurrent_depth.util import cosine_lr, seed_all

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="rd_1x8")
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--steps", type=int, default=2000)
ap.add_argument("--sched-total", type=int, default=6000, help="cosine horizon (match the 6000-step runs)")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--random-depth", action="store_true")
ap.add_argument("--rd-range", type=int, nargs=2, default=[1, 8])
ap.add_argument("--derange", action="store_true")
ap.add_argument("--typed-mid", action="store_true")
ap.add_argument("--swap", action="store_true", help="bank2 before bank1")
ap.add_argument("--distinct-vals", action="store_true", help="values without replacement (kills the most-frequent-value shortcut)")
ap.add_argument("--mix-hop1", type=float, default=0.0,
                help="fraction of TRAIN samples whose query token is Q1 (answer = hop-1 destination, "
                     "i.e. the mid entity) instead of Q (answer = 2-hop value); eval is always 2-hop")
ap.add_argument("--eval-every", type=int, default=250)
ap.add_argument("--depth-eval", type=int, nargs="*", default=[1, 2, 4, 8, 16])
ap.add_argument("--ckpt", default=None)
ap.add_argument("--load", default=None)
ap.add_argument("--out", default=None)
args = ap.parse_args()

device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
seed_all(args.seed)
K = args.k
N_E, N_V = 128, 128
N_M = 128 if args.typed_mid else 0
ENT0 = N_CONTROL
MID0 = ENT0 + N_E
VAL0 = MID0 + N_M
Q1TOK = VAL0 + N_V                 # extra control token appended after the values
VOCAB = VAL0 + N_V + 1


def sample_raw(B, gen):
    ents = torch.stack([torch.randperm(N_E, generator=gen)[:K] + ENT0 for _ in range(B)])
    if args.typed_mid:
        mids = torch.stack([torch.randperm(N_M, generator=gen)[:K] + MID0 for _ in range(B)])
    else:
        mids = ents
    if args.distinct_vals:
        vals = torch.stack([torch.randperm(N_V, generator=gen)[:K] + VAL0 for _ in range(B)])
    else:
        vals = torch.randint(N_V, (B, K), generator=gen) + VAL0
    perm = torch.stack([torch.randperm(K, generator=gen) for _ in range(B)])
    t = torch.randint(K, (B,), generator=gen)
    if args.derange:
        ar = torch.arange(B)
        fp = perm[ar, t] == t
        if fp.any():
            # swap perm[t] with perm[t+1] on fixed-point rows -> still a permutation, perm[t] != t
            t1 = (t + 1) % K
            a, b = perm[ar, t].clone(), perm[ar, t1].clone()
            perm[ar[fp], t[fp]] = b[fp]
            perm[ar[fp], t1[fp]] = a[fp]
    return ents, mids, vals, perm, t


def pack(ents, mids, vals, perm, t, hop1=None):
    """bank1: (e_i, m_pi(i)) ; bank2: (m_i, v_i) ; Q e_t A v_pi(t).  With mids==ents this is
    exactly tasks._pack_long_2hop.  Rows with hop1[b]=True instead get  Q1 e_t A m_pi(t)."""
    B = ents.shape[0]
    dest = mids.gather(1, perm)
    def bank(src, dst):
        p = []
        for i in range(K):
            p.append(src[:, i:i+1]); p.append(dst[:, i:i+1])
            if i < K - 1:
                p.append(torch.full((B, 1), SEP, dtype=torch.long))
        return p
    b1 = bank(ents, dest); b2 = bank(mids, vals)
    parts = [torch.full((B, 1), FACT, dtype=torch.long)]
    first, second = (b2, b1) if args.swap else (b1, b2)
    parts += first; parts.append(torch.full((B, 1), SEP, dtype=torch.long)); parts += second
    ar = torch.arange(B)
    q = ents[ar, t].unsqueeze(1)
    a = vals[ar, perm[ar, t]].unsqueeze(1)
    qtok = torch.full((B, 1), QTOK, dtype=torch.long)
    if hop1 is not None:
        a = torch.where(hop1.unsqueeze(1), mids[ar, perm[ar, t]].unsqueeze(1), a)
        qtok = torch.where(hop1.unsqueeze(1), torch.full_like(qtok, Q1TOK), qtok)
    parts += [qtok, q,
              torch.full((B, 1), ATOK, dtype=torch.long), a,
              torch.full((B, 1), EOS, dtype=torch.long)]
    seq = torch.cat(parts, 1)
    return seq, seq.shape[1] - 2


def sample(B, gen, mix=0.0):
    raw = sample_raw(B, gen)
    hop1 = (torch.rand(B, generator=gen) < mix) if mix > 0 else None
    seq, ans_pos = pack(*raw, hop1=hop1)
    return seq.to(device), ans_pos, seq[:, ans_pos].to(device), [r.to(device) for r in raw]


spec = ARM_SPECS[args.arm]
cfg = CoreConfig(d_model=256, n_layers=spec["n_distinct"], d_state=64, d_conv=4, expand=2,
                 headdim=64, block="mamba2", norm_position="post", inject_input=True)
model = make_recurrent_model(VOCAB, device, cfg, spec["n_distinct"], spec["applies_per_block"])
print(f"arm={args.arm} K={K} vocab={VOCAB} typed_mid={args.typed_mid} distinct_vals={args.distinct_vals} "
      f"derange={args.derange} swap={args.swap} mix_hop1={args.mix_hop1} random_depth={args.random_depth} "
      f"steps={args.steps}", flush=True)


@torch.no_grad()
def breakdown(r, gen, n_batches=4, batch=512):
    model.eval()
    c = dict(n=0, correct=0, own=0, rev=0, other_present=0, absent_value=0, mid_tok=0, entity=0, control=0,
             pred_is_mode=0, pred_count=0.0, ans_count=0.0, mode_is_ans=0,
             fp_n=0, fp_correct=0, nfp_n=0, nfp_correct=0, rank_ans=0.0, rank_own=0.0,
             logit_ans=0.0, logit_own=0.0, logit_other_present=0.0, logit_absent=0.0, loss=0.0)
    pos_hist = torch.zeros(K)
    for _ in range(n_batches):
        seq, ans_pos, target, (ents, mids, vals, perm, t) = sample(batch, gen)
        logits = model(seq, applies=r)[:, ans_pos - 1].float()
        B = seq.shape[0]; ar = torch.arange(B, device=device)
        pred = logits.argmax(-1)
        v_ans = target
        v_own = vals[ar, t]
        inv = perm.argsort(1); v_rev = vals[ar, inv[ar, t]]
        correct = pred == v_ans
        own = (pred == v_own) & ~correct
        rev = (pred == v_rev) & ~correct & ~own
        present = (pred.unsqueeze(1) == vals).any(1)
        other_present = present & ~correct & ~own & ~rev
        is_val = pred >= VAL0
        absent = is_val & ~present
        is_mid = (pred >= MID0) & (pred < VAL0) if args.typed_mid else torch.zeros_like(is_val)
        is_ent = (pred >= ENT0) & (pred < MID0)
        ctrl = pred < ENT0
        fp = perm[ar, t] == t
        c["n"] += B; c["correct"] += correct.sum().item(); c["own"] += own.sum().item()
        c["rev"] += rev.sum().item(); c["other_present"] += other_present.sum().item()
        c["absent_value"] += absent.sum().item(); c["mid_tok"] += is_mid.sum().item()
        c["entity"] += is_ent.sum().item(); c["control"] += ctrl.sum().item()
        c["fp_n"] += fp.sum().item(); c["fp_correct"] += (correct & fp).sum().item()
        c["nfp_n"] += (~fp).sum().item(); c["nfp_correct"] += (correct & ~fp).sum().item()
        # most-frequent-value shortcut: how often is the prediction the modal value of bank 2,
        # and how many times does the predicted / true value occur in bank 2?
        cnt = (vals.unsqueeze(2) == vals.unsqueeze(1)).sum(2)            # [B,K] multiplicity of each slot
        mode_val = vals[ar, cnt.argmax(1)]
        c["pred_is_mode"] += (pred == mode_val).sum().item()
        c["mode_is_ans"] += (mode_val == v_ans).sum().item()
        c["pred_count"] += (pred.unsqueeze(1) == vals).sum(1).float().sum().item()
        c["ans_count"] += (v_ans.unsqueeze(1) == vals).sum(1).float().sum().item()
        ranks = (logits > logits[ar, v_ans].unsqueeze(1)).sum(1)
        c["rank_ans"] += ranks.float().sum().item()
        c["rank_own"] += (logits > logits[ar, v_own].unsqueeze(1)).sum(1).float().sum().item()
        c["logit_ans"] += logits[ar, v_ans].sum().item()
        c["logit_own"] += logits[ar, v_own].sum().item()
        # mean logit over present values other than v_ans / v_own
        lv = logits.gather(1, vals)
        mask = (vals != v_ans.unsqueeze(1)) & (vals != v_own.unsqueeze(1))
        c["logit_other_present"] += (lv * mask).sum(1).div(mask.sum(1).clamp(min=1)).sum().item()
        c["logit_absent"] += logits[:, VAL0:].mean(1).sum().item()
        c["loss"] += F.cross_entropy(logits, target, reduction="sum").item()
        # which bank-2 slot did the prediction come from (recency profile)?
        hit = (pred.unsqueeze(1) == vals)
        pos_hist += hit.float().sum(0).cpu()
    n = c["n"]
    out = {k: (v / n if k not in ("n", "fp_n", "nfp_n", "fp_correct", "nfp_correct") else v) for k, v in c.items()}
    out["acc_fixedpoint"] = c["fp_correct"] / max(1, c["fp_n"])
    out["acc_nonfixedpoint"] = c["nfp_correct"] / max(1, c["nfp_n"])
    out["pos_hist"] = (pos_hist / pos_hist.sum().clamp(min=1)).tolist()
    model.train()
    return out


@torch.no_grad()
def hop1_eval(r, gen, n_batches=2, batch=512):
    """Score Q1 queries: answer = m_pi(t) (hop-1 destination). Reports acc, and whether the
    prediction is at least some hop-1 destination present in bank 1 (bag-of-mids) and its logit gap."""
    model.eval()
    n = correct = present = 0; gap = 0.0
    for _ in range(n_batches):
        raw = sample_raw(batch, gen)
        hop1 = torch.ones(batch, dtype=torch.bool)
        seq, ans_pos = pack(*raw, hop1=hop1)
        seq = seq.to(device); target = seq[:, ans_pos]
        ents, mids, vals, perm, t = [x.to(device) for x in raw]
        logits = model(seq, applies=r)[:, ans_pos - 1].float()
        pred = logits.argmax(-1); ar = torch.arange(batch, device=device)
        correct += (pred == target).sum().item()
        present += (pred.unsqueeze(1) == mids).any(1).sum().item()
        lm = logits.gather(1, mids)                      # logits of all K mids
        mask = mids != target.unsqueeze(1)
        gap += (logits[ar, target] - (lm * mask).sum(1) / mask.sum(1)).sum().item()
        n += batch
    model.train()
    return {"hop1_acc": correct / n, "hop1_pred_in_bank": present / n, "hop1_logit_gap": gap / n}


def fmt(o):
    return (f"acc={o['correct']:.3f} own(v_t)={o['own']:.3f} rev={o['rev']:.3f} otherPresent={o['other_present']:.3f} "
            f"absentVal={o['absent_value']:.3f} mid={o['mid_tok']:.3f} ent={o['entity']:.3f} ctrl={o['control']:.3f} | "
            f"acc@fp={o['acc_fixedpoint']:.3f} acc@nonfp={o['acc_nonfixedpoint']:.3f} | "
            f"pred=mode {o['pred_is_mode']:.3f} (mode=ans {o['mode_is_ans']:.3f}) "
            f"mult(pred)={o['pred_count']:.2f} mult(ans)={o['ans_count']:.2f} | "
            f"rank(v_ans)={o['rank_ans']:.1f} rank(v_t)={o['rank_own']:.1f} | "
            f"logit ans={o['logit_ans']:.2f} own={o['logit_own']:.2f} otherPres={o['logit_other_present']:.2f} "
            f"absent={o['logit_absent']:.2f} loss={o['loss']:.2f}")


if args.load:
    model.load_state_dict(torch.load(args.load, map_location=device))
else:
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, betas=(0.9, 0.95))
    train_gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 10_000)
    depth_gen = torch.Generator().manual_seed(args.seed + 20_000)
    t0 = time.time()
    for step in range(args.steps):
        lr = cosine_lr(step, warmup=50, total=args.sched_total, base=3e-4, floor=3e-5)
        for pg in opt.param_groups:
            pg["lr"] = lr
        seq, ans_pos, target, _ = sample(args.batch, train_gen, mix=args.mix_hop1)
        r = None
        if args.random_depth:
            r = int(torch.randint(args.rd_range[0], args.rd_range[1] + 1, (1,), generator=depth_gen).item())
        logits = model(seq, applies=r)
        loss = F.cross_entropy(logits[:, ans_pos - 1], target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            o = breakdown(spec["applies_per_block"], eval_gen, n_batches=2)
            h = hop1_eval(spec["applies_per_block"], eval_gen) if args.mix_hop1 > 0 else {}
            hs = (f" || HOP1 acc={h['hop1_acc']:.3f} inBank={h['hop1_pred_in_bank']:.3f} gap={h['hop1_logit_gap']:.2f}"
                  if h else "")
            print(f"step {step+1:5d} loss {loss.item():.3f} [{time.time()-t0:5.0f}s] {fmt(o)}{hs}", flush=True)
    if args.ckpt:
        torch.save(model.state_dict(), args.ckpt)

print("\n=== final breakdown vs test-time r ===")
final = {}
eval_gen = torch.Generator().manual_seed(args.seed + 30_000)
for r in args.depth_eval:
    o = breakdown(r, eval_gen, n_batches=4)
    final[r] = o
    print(f"r={r:2d}  {fmt(o)}")
    if args.mix_hop1 > 0:
        h = hop1_eval(r, eval_gen); final[r].update(h)
        print(f"       HOP1 (Q1 queries): acc={h['hop1_acc']:.3f} pred_in_bank1={h['hop1_pred_in_bank']:.3f} logit_gap={h['hop1_logit_gap']:.2f}")
    print(f"       bank-2 slot histogram of predictions (slot 0 = earliest .. {K-1} = latest): "
          + " ".join(f"{p:.2f}" for p in o["pos_hist"]))
if args.out:
    json.dump({"args": vars(args), "final": final}, open(args.out, "w"), indent=1)
