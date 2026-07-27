"""Mean contacts asserted per rollout, from the saved vote matrices.

M[i,j] counts how many of the N rollouts asserted pair (i,j), so
sum(triu(M)) / N is the mean number of distinct sep>=6 contacts a single
rollout emits. Compare against the resolved-restricted GT count the eval
credits (n_true for range=all) to read off under/over-generation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

GT = Path("/home/bizon/git/MarinFold/.claude/worktrees/vibrant-hermann-12cd27/"
          "experiments/exp89_evals_contacts_v1_model_on_eval_set/data/gt_universe.jsonl")
MIN_DEG, MIN_SEP = 0.001, 6
N = 100

recs = [json.loads(x) for x in GT.open()]
dirs = [(lbl, Path(d)) for lbl, d in (a.split("=", 1) for a in sys.argv[1:])]
# only proteins present in EVERY dir, so the comparison is paired
keys = None
for _, d in dirs:
    have = {f"{r['dataset']}__{r['stem']}" for r in recs
            if (d / f"{r['dataset']}__{r['stem']}.npz").exists()}
    keys = have if keys is None else (keys & have)
print(f"paired on {len(keys)} proteins")

for lbl, d in dirs:
    pred, gt, ratios = [], [], []
    for r in recs:
        k = f"{r['dataset']}__{r['stem']}"
        if k not in keys:
            continue
        L = r["L"]
        M = np.load(d / f"{k}.npz")["score"].astype(np.float64)
        iu = np.triu_indices(L, k=1)
        npred = float(M[iu].sum()) / N
        resolved = set(int(x) for x in r["resolved"])
        ngt = sum(1 for i, j, dg in r["contacts"]
                  if dg >= MIN_DEG and abs(int(j) - int(i)) >= MIN_SEP
                  and int(i) in resolved and int(j) in resolved)
        pred.append(npred); gt.append(ngt)
        if ngt:
            ratios.append(npred / ngt)
    pred, gt, ratios = np.array(pred), np.array(gt), np.array(ratios)
    print(f"{lbl:28s} mean contacts/rollout = {pred.mean():7.1f}   "
          f"mean GT = {gt.mean():7.1f}   pooled pred/GT = {pred.sum() / gt.sum():.3f}   "
          f"median per-protein ratio = {np.median(ratios):.3f}")
