# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""exp163 MVP sampling-regime eval (the trained/deployment objective, no matrix
confound). Per val protein, SAMPLE contact sets (top-k disabled = under-gen fix)
and score F1 vs GT:
  base@K0     : fresh base rollout (no candidates)
  refiner@K0  : refiner, no candidates
  refiner@K16 : refiner conditioned on 16 candidate blocks  <- the refined output
  best-cand   : max F1 over the 16 shown candidates (oracle upper bound)
  consensus   : F1 of the >=30%-vote consensus set
Headline: refiner@K16 vs base@K0 / vs refiner@K0 / vs best-cand.

    uv run --no-sync python mvp_sample_eval.py --base <hf> --refiner <dir> \
        --rollouts val_preds.parquet --targets <exp98>/targets.parquet --out samp.csv
"""
from __future__ import annotations
import argparse, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/bizon/git/MarinFold/.claude/worktrees/protein-rollout-post-training-e203d8")
for p in ("exp82_evals_contacts_v1_contact_prediction", "exp89_evals_contacts_v1_model_on_eval_set",
          "exp98_data_generate_rollouts_contacts_v1_train"):
    sys.path.insert(0, str(REPO / "experiments" / p))
from eval_contact_prediction import Scorer, BEGIN
from score_eval_set import prefix_and_positions
from rollout_metrics import parse_pred, gt_by_band, score_rollout

MARKER = "<contacts-and-distances-v1>"; MIN_SEP = 6

def canon(flat):
    a = np.asarray(flat).reshape(-1, 2)
    if a.size == 0: return set()
    lo = np.minimum(a[:, 0], a[:, 1]); hi = np.maximum(a[:, 0], a[:, 1])
    k = (hi - lo) >= MIN_SEP
    return set(zip(lo[k].tolist(), hi[k].tolist()))

def emit_block(pairs, seq_pos, rng):
    toks = [MARKER]; order = list(pairs); rng.shuffle(order)
    for (i, j) in order:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        toks += ["<contact>", f"<p{seq_pos[a]}>", f"<p{seq_pos[b]}>"]
    return toks

def prefix_with(prefix, seq_pos, blocks):
    toks = [prefix[: prefix.rindex(BEGIN)].rstrip()]
    for blk in blocks: toks += blk
    toks.append(BEGIN); return " ".join(toks)

def consensus(pool, L, frac, M, ncap, rng):
    idx = rng.choice(len(pool), min(M, len(pool)), replace=False); votes = Counter()
    for t in idx:
        for (i, j) in canon(pool[t]):
            if i < L and j < L: votes[(i, j)] += 1
    thr = max(2, int(frac * len(idx)))
    keep = sorted([p for p, c in votes.items() if c >= thr], key=lambda p: -votes[p])
    return set(keep[:ncap])

def samp(scorer, prefix, pos_to_seq, gtb, L, n):
    ids = scorer.tok(prefix, add_special_tokens=False).input_ids
    texts = scorer.rollouts(ids, n_rollouts=n, temperature=1.0, top_p=0.95, max_new=4 * L + 64)
    fa, fl, npd = [], [], []
    for t in texts:
        s = score_rollout(parse_pred(t, pos_to_seq), gtb)
        fa.append(s["all_f1"]); fl.append(s["long_f1"]); npd.append(s["all_npred"])
    return float(np.nanmean(fa)), float(np.nanmean(fl)), float(np.mean(npd))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--refiner", required=True)
    ap.add_argument("--rollouts", required=True); ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--n-cap", type=int, default=120)
    ap.add_argument("--n-samp", type=int, default=2); ap.add_argument("--limit", type=int, default=40)
    a = ap.parse_args()
    tgt = pd.read_parquet(a.targets).set_index("entry_id")
    roll = pd.read_parquet(a.rollouts, columns=["entry_id", "r", "pred"])
    preds_by = {e: list(g["pred"].to_numpy()) for e, g in roll.groupby("entry_id")}
    eids = [e for e in preds_by if e in tgt.index][: a.limit]
    base = Scorer(a.base); ref = Scorer(a.refiner)
    rng = np.random.default_rng(0); rows = []
    for n, eid in enumerate(eids):
        built = prefix_and_positions(eid, tgt.loc[eid, "sequence"])
        if built is None: continue
        prefix, seq_pos, L = built
        pos_to_seq = {seq_pos[k]: k for k in range(L)}
        gt = {(i, j) for (i, j) in canon(np.concatenate([np.asarray(p).ravel()
               for p in tgt.loc[eid, "gt_contacts"]])) if i < L and j < L}
        if len(gt) < 5: continue
        gtb = gt_by_band(gt)
        cand_sets = []
        for ri in rng.choice(len(preds_by[eid]), min(a.k, len(preds_by[eid])), replace=False):
            c = {(i, j) for (i, j) in canon(preds_by[eid][ri]) if i < L and j < L}
            if c: cand_sets.append(set(list(c)[: a.n_cap]))
        best_all = max((score_rollout(c, gtb)["all_f1"] for c in cand_sets), default=np.nan)
        best_long = max((score_rollout(c, gtb)["long_f1"] for c in cand_sets), default=np.nan)
        cons = consensus(preds_by[eid], L, 0.3, 32, a.n_cap, rng)
        cons_all = score_rollout(cons, gtb)["all_f1"]
        p16 = prefix_with(prefix, seq_pos, [emit_block(list(c), seq_pos, rng) for c in cand_sets])
        b_a, b_l, b_np = samp(base, prefix, pos_to_seq, gtb, L, a.n_samp)
        r0_a, r0_l, r0_np = samp(ref, prefix, pos_to_seq, gtb, L, a.n_samp)
        r16_a, r16_l, r16_np = samp(ref, p16, pos_to_seq, gtb, L, a.n_samp)
        rows.append(dict(entry_id=eid, L=L, n_gt=len(gt),
                         base_all=b_a, ref0_all=r0_a, ref16_all=r16_a, bestc_all=best_all, cons_all=cons_all,
                         base_long=b_l, ref0_long=r0_l, ref16_long=r16_l, bestc_long=best_long,
                         base_np=b_np, ref16_np=r16_np))
        if (n + 1) % 10 == 0: print(f"  {n+1}/{len(eids)}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(a.out, index=False)
    print(f"\n=== SAMPLING F1  ({len(d)} val proteins, n_samp={a.n_samp}) ===", flush=True)
    for band in ("all", "long"):
        print(f"  {band:>4}:  base@K0={d[f'base_{band}'].mean():.4f}  ref@K0={d[f'ref0_{band}'].mean():.4f}  "
              f"ref@K16={d[f'ref16_{band}'].mean():.4f}  best-cand(oracle)={d[f'bestc_{band}'].mean():.4f}"
              + (f"  consensus={d['cons_all'].mean():.4f}" if band == 'all' else ''), flush=True)
    print(f"  ref@K16 vs base@K0: dAll={ (d.ref16_all-d.base_all).mean():+.4f} (win {100*(d.ref16_all>d.base_all).mean():.0f}%)  "
          f"dLong={ (d.ref16_long-d.base_long).mean():+.4f}", flush=True)
    print(f"  ref@K16 vs ref@K0 : dAll={ (d.ref16_all-d.ref0_all).mean():+.4f}   "
          f"ref@K16 vs best-cand: dAll={ (d.ref16_all-d.bestc_all).mean():+.4f}", flush=True)
    print(f"  n_pred: base@K0={d.base_np.mean():.0f}  ref@K16={d.ref16_np.mean():.0f}  (gt≈{d.n_gt.mean():.0f})", flush=True)

if __name__ == "__main__":
    main()
