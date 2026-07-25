# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""exp163 MVP eval: does conditioning on K candidate blocks beat no-candidates,
for a given model?  Runs the exp89 calibrated matrix at K=0 and K=16 on the
val proteins.  Run for BOTH the base E8 and the fine-tuned refiner and compare:
  - refiner: R(K16) > R(K0)?      -> learned to USE candidates (the headline)
  - refiner R(K0) vs base R(K0)   -> did fine-tuning hurt one-shot?
  - base:    R(K16) vs R(K0)      -> base is poisoned by noisy candidates (Step 2)

    uv run --no-sync python mvp_eval.py --model <base|refiner> --rollouts val_preds.parquet \
        --targets <exp98>/targets.parquet --out eval_<tag>.csv
"""
from __future__ import annotations
import argparse, sys
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

def cand_prefix(prefix, seq_pos, cands, rng, marker=MARKER):
    head = prefix[: prefix.rindex(BEGIN)].rstrip()
    toks = [head]
    for pairs in cands:
        toks.append(marker)
        order = list(pairs); rng.shuffle(order)
        for (i, j) in order:
            a, b = (i, j) if rng.random() < 0.5 else (j, i)
            toks += ["<contact>", f"<p{seq_pos[a]}>", f"<p{seq_pos[b]}>"]
    toks.append(BEGIN)
    return " ".join(toks)

def rprec(scorer, prefix, seq_pos, L, gt):
    sym = score_matrix(scorer, prefix, seq_pos)
    tmat = true_matrix(L, [(i, j, 1.0) for (i, j) in gt])
    pi, pj, psep = resolved_pairs(np.arange(L))
    rows = metric_rows(sym.astype(np.float64), tmat, pi, pj, psep, L, with_precision=True)
    d = {(r["range"], r["cut"]): r for r in rows}
    return d[("all", "R")]["precision"], d[("long", "R")]["precision"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rollouts", required=True)
    ap.add_argument("--targets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n-cap", type=int, default=120)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    tgt = pd.read_parquet(a.targets).set_index("entry_id")
    roll = pd.read_parquet(a.rollouts, columns=["entry_id", "r", "pred"])
    preds_by = {e: list(g["pred"].to_numpy()) for e, g in roll.groupby("entry_id")}
    eids = [e for e in preds_by if e in tgt.index]
    if a.limit: eids = eids[: a.limit]
    scorer = Scorer(a.model)
    rng = np.random.default_rng(0)
    rows = []
    for n, eid in enumerate(eids):
        seq = tgt.loc[eid, "sequence"]
        built = prefix_and_positions(eid, seq)
        if built is None: continue
        prefix, seq_pos, L = built
        gt = [(i, j) for (i, j) in canon(np.concatenate([np.asarray(p).ravel()
                for p in tgt.loc[eid, "gt_contacts"]])) if i < L and j < L]
        if len(gt) < 5: continue
        r0a, r0l = rprec(scorer, prefix, seq_pos, L, gt)            # K=0
        cands = []
        for ri in rng.choice(len(preds_by[eid]), min(a.k, len(preds_by[eid])), replace=False):
            p = [(i, j) for (i, j) in canon(preds_by[eid][ri]) if i < L and j < L]
            if p: cands.append(p[: a.n_cap])
        pfx = cand_prefix(prefix, seq_pos, cands, rng)
        rKa, rKl = rprec(scorer, pfx, seq_pos, L, gt)               # K=16
        rows.append(dict(entry_id=eid, L=L, n_gt=len(gt), K=len(cands),
                         R0_all=r0a, RK_all=rKa, R0_long=r0l, RK_long=rKl))
        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(eids)}", flush=True)
    d = pd.DataFrame(rows); d.to_csv(a.out, index=False)
    print(f"\n=== {a.model}  ({len(d)} val proteins, K={a.k}) ===", flush=True)
    print(f"  all-band : R(K0)={d.R0_all.mean():.4f}  R(K{a.k})={d.RK_all.mean():.4f}  "
          f"dR={ (d.RK_all-d.R0_all).mean():+.4f}  (K wins {100*(d.RK_all>d.R0_all).mean():.0f}%)", flush=True)
    print(f"  long-band: R(K0)={d.R0_long.mean():.4f}  R(K{a.k})={d.RK_long.mean():.4f}  "
          f"dR={ (d.RK_long-d.R0_long).mean():+.4f}", flush=True)

if __name__ == "__main__":
    main()
