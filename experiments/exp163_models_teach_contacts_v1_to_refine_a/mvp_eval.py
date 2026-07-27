# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""exp163 MVP eval: calibrated-matrix R-precision under three candidate contexts
for a given model, on the val proteins:
  K0        : no candidates (seq + BEGIN)                -> one-shot readout
  raw       : K raw rollout blocks (13% precision each)  -> the trained format
  consensus : ONE block = contacts with vote >= frac*M over M sampled rollouts
              (deployable HIGH-precision partial; the Step-2 lever)

    uv run --no-sync python mvp_eval.py --model <base|refiner> --rollouts val_preds.parquet \
        --targets <exp98>/targets.parquet --out eval.csv
"""
from __future__ import annotations
import argparse, sys
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path("/home/bizon/git/MarinFold/.claude/worktrees/protein-rollout-post-training-e203d8")
sys.path.insert(0, str(REPO / "experiments/exp82_evals_contacts_v1_contact_prediction"))
sys.path.insert(0, str(REPO / "experiments/exp89_evals_contacts_v1_model_on_eval_set"))
from eval_contact_prediction import Scorer, BEGIN
from score_eval_set import prefix_and_positions, score_matrix
from compute_metrics import true_matrix, resolved_pairs, metric_rows

MARKER = "<contacts-and-distances-v1>"
MIN_SEP = 6

def canon(flat):
    a = np.asarray(flat).reshape(-1, 2)
    if a.size == 0: return []
    lo = np.minimum(a[:, 0], a[:, 1]); hi = np.maximum(a[:, 0], a[:, 1])
    k = (hi - lo) >= MIN_SEP
    return sorted(set(zip(lo[k].tolist(), hi[k].tolist())))

def emit_block(pairs, seq_pos, rng):
    toks = [MARKER]
    order = list(pairs); rng.shuffle(order)
    for (i, j) in order:
        a, b = (i, j) if rng.random() < 0.5 else (j, i)
        toks += ["<contact>", f"<p{seq_pos[a]}>", f"<p{seq_pos[b]}>"]
    return toks

def prefix_with(prefix, seq_pos, blocks):
    head = prefix[: prefix.rindex(BEGIN)].rstrip()
    toks = [head]
    for blk in blocks: toks += blk
    toks.append(BEGIN)
    return " ".join(toks)

def rprec(scorer, prefix, seq_pos, L, gt):
    sym = score_matrix(scorer, prefix, seq_pos)
    tmat = true_matrix(L, [(i, j, 1.0) for (i, j) in gt])
    pi, pj, psep = resolved_pairs(np.arange(L))
    rows = metric_rows(sym.astype(np.float64), tmat, pi, pj, psep, L, with_precision=True)
    d = {(r["range"], r["cut"]): r for r in rows}
    return d[("all", "R")]["precision"], d[("long", "R")]["precision"]

def consensus(pool, L, frac, M, ncap, rng):
    idx = rng.choice(len(pool), min(M, len(pool)), replace=False)
    votes = Counter()
    for t in idx:
        for (i, j) in canon(pool[t]):
            if i < L and j < L: votes[(i, j)] += 1
    thr = max(2, int(frac * len(idx)))
    keep = [p for p, c in votes.items() if c >= thr]
    keep.sort(key=lambda p: -votes[p])
    return keep[:ncap]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", required=True); ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16); ap.add_argument("--n-cap", type=int, default=120)
    ap.add_argument("--cons-frac", type=float, default=0.3); ap.add_argument("--cons-pool", type=int, default=32)
    ap.add_argument("--limit", type=int, default=60)
    a = ap.parse_args()
    tgt = pd.read_parquet(a.targets).set_index("entry_id")
    roll = pd.read_parquet(a.rollouts, columns=["entry_id", "r", "pred"])
    preds_by = {e: list(g["pred"].to_numpy()) for e, g in roll.groupby("entry_id")}
    eids = [e for e in preds_by if e in tgt.index][: a.limit]
    scorer = Scorer(a.model); rng = np.random.default_rng(0); rows = []
    for n, eid in enumerate(eids):
        built = prefix_and_positions(eid, tgt.loc[eid, "sequence"])
        if built is None: continue
        prefix, seq_pos, L = built
        gt = [(i, j) for (i, j) in canon(np.concatenate([np.asarray(p).ravel()
                for p in tgt.loc[eid, "gt_contacts"]])) if i < L and j < L]
        if len(gt) < 5: continue
        gts = set(gt)
        r0a, r0l = rprec(scorer, prefix, seq_pos, L, gt)
        # raw: K rollout blocks
        raw = []
        for ri in rng.choice(len(preds_by[eid]), min(a.k, len(preds_by[eid])), replace=False):
            p = [(i, j) for (i, j) in canon(preds_by[eid][ri]) if i < L and j < L]
            if p: raw.append(emit_block(p[: a.n_cap], seq_pos, rng))
        rRa, rRl = rprec(scorer, prefix_with(prefix, seq_pos, raw), seq_pos, L, gt)
        # consensus: one high-precision block
        cons = consensus(preds_by[eid], L, a.cons_frac, a.cons_pool, a.n_cap, rng)
        cprec = (len(set(cons) & gts) / len(cons)) if cons else float("nan")
        cRa, cRl = rprec(scorer, prefix_with(prefix, seq_pos, [emit_block(cons, seq_pos, rng)]), seq_pos, L, gt)
        rows.append(dict(entry_id=eid, L=L, n_gt=len(gt), R0_all=r0a, Rraw_all=rRa, Rcons_all=cRa,
                         R0_long=r0l, Rraw_long=rRl, Rcons_long=cRl, cons_n=len(cons), cons_prec=cprec))
        if (n + 1) % 20 == 0: print(f"  {n+1}/{len(eids)}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(a.out, index=False)
    print(f"\n=== {a.model}  ({len(d)} val proteins) ===", flush=True)
    print(f"  consensus set: mean size={d.cons_n.mean():.0f}  mean precision={d.cons_prec.mean():.3f}", flush=True)
    for band in ("all", "long"):
        r0, rr, rc = d[f"R0_{band}"].mean(), d[f"Rraw_{band}"].mean(), d[f"Rcons_{band}"].mean()
        print(f"  {band:>4}: K0={r0:.4f}  raw-K{a.k}={rr:.4f} (d{rr-r0:+.4f})  "
              f"consensus={rc:.4f} (d{rc-r0:+.4f})", flush=True)

if __name__ == "__main__":
    main()
