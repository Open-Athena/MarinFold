# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 4 — score the KNN matrices with exp89's metric and merge for comparison.

To put the KNN baselines on the same axis as MarinFold / Protenix / ESMFold /
ESMFold2, the contact metrics MUST come from exp89's *exact* implementation —
mixing metric impls disagrees by up to 0.4/protein (float16 tie-breaking + small
proteins). So the metric functions below are **copied verbatim from exp89's
compute_metrics.py** (keep identical).

For every ``_scratch/scores/k{K}_{self,noself}`` directory and each score key
(``score`` = tie-broken, ``count`` = plain integer votes) we emit precision@{L,
L/2,L/5,R} + AUC per range over the resolved universe, labelled
``seq-knn-k{K}[-noself][-plain]`` / ``predictor=knn``. Writes:

* ``data/knn_precision_all.csv`` — per-protein KNN rows (carries strata).
* ``data/knn_comparison.csv``    — mean precision per (model,range,cut), KNN rows
  concatenated with the existing predictors' per-protein rows (``--base-csv``),
  ready for the plots.

    uv run python compute_knn_metrics.py --gt _scratch/gt_universe.jsonl \
        --scores-root _scratch/scores --base-csv <exp82>/_scratch/contact_precision_with_rollout.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# --- verbatim from exp89 compute_metrics.py (DO NOT EDIT — must stay identical) ---
RANGES = {"all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None)}
CUTS = (("L", lambda L, c: L), ("L/2", lambda L, c: max(1, L // 2)),
        ("L/5", lambda L, c: max(1, L // 5)), ("R", lambda L, c: c))
MIN_DEG, MIN_SEP = 0.001, 6
STRATA_COLS = ["neff_tier", "fold_verdict", "seq_leakage", "msa_neff", "length"]


def true_matrix(L, contacts):
    m = np.zeros((L, L), bool)
    for i, j, d in contacts:
        i, j = int(i), int(j)
        if d >= MIN_DEG and (j - i) >= MIN_SEP and i < j < L:
            m[i, j] = True
    return m


def resolved_pairs(resolved):
    a, b = np.triu_indices(len(resolved), k=1)
    i, j = resolved[a], resolved[b]
    return i, j, (j - i)


def metric_rows(score, tmat, pi, pj, psep, L, *, with_precision):
    cs, cg = score[pi, pj], tmat[pi, pj].astype(int)
    rows = []
    for rng, (lo, hi) in RANGES.items():
        inr = psep >= lo
        if hi is not None:
            inr = inr & (psep <= hi)
        s, g = cs[inr], cg[inr]
        nc, nt = int(s.size), int(g.sum())
        if with_precision:
            order = np.argsort(-s, kind="mergesort") if nc else None
            gs = g[order] if nc else None
            for cut, fn in CUTS:
                tgt = int(fn(L, nt))
                if nc == 0 or tgt <= 0:
                    rows.append(dict(range=rng, cut=cut, precision=float("nan"),
                                     n_candidate=nc, n_true=nt, n_top=0))
                else:
                    top = min(tgt, nc)
                    rows.append(dict(range=rng, cut=cut, precision=float(gs[:top].sum()) / top,
                                     n_candidate=nc, n_true=nt, n_top=top))
        auc = float(roc_auc_score(g, s)) if (nc and 0 < nt < nc) else float("nan")
        rows.append(dict(range=rng, cut="AUC", precision=auc,
                         n_candidate=nc, n_true=nt, n_top=nc))
    return rows


def stamp(rows, *, rec, model, mode, predictor):
    strata = rec.get("strata", {}) or {}
    base = dict(dataset=rec["dataset"], stem=rec["stem"], n_residues=rec["L"],
                model=model, mode=mode, predictor=predictor)
    for k in STRATA_COLS:
        base[k] = strata.get(k)
    return [{**base, **r} for r in rows]
# --- end verbatim ---


def label(k: int, mode: str, key: str) -> str:
    suffix = ("" if mode == "self" else "-noself") + ("" if key == "score" else "-plain")
    return f"seq-knn-k{k}{suffix}"


def score_dir(scores_dir: Path, gt: list[dict], k: int, mode: str, key: str) -> list[dict]:
    rows: list[dict] = []
    model = label(k, mode, key)
    n = 0
    for rec in gt:
        npz = scores_dir / f"{rec['dataset']}__{rec['stem']}.npz"
        if not npz.exists():
            continue
        L = rec["L"]
        arr = np.load(npz)
        if key not in arr:
            continue
        score = arr[key].astype(np.float64)
        if score.shape != (L, L):
            print(f"  {rec['stem']}: score shape {score.shape} != L={L}; skipping")
            continue
        resolved = np.asarray(rec["resolved"], dtype=np.int64)
        tmat = true_matrix(L, rec["contacts"])
        pi, pj, psep = resolved_pairs(resolved)
        rows += stamp(metric_rows(score, tmat, pi, pj, psep, L, with_precision=True),
                      rec=rec, model=model, mode="single_seq", predictor="knn")
        n += 1
    print(f"[metrics] {model}: scored {n}/{len(gt)}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--scores-root", type=Path, required=True)
    ap.add_argument("--base-csv", type=Path, required=True,
                    help="per-protein rows for the existing predictors (exp82 with-rollout, "
                         "or exp89 contact_precision_all.csv)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    ap.add_argument("--modes", nargs="+", default=["self", "noself"])
    ap.add_argument("--keys", nargs="+", default=["score", "count"])
    ap.add_argument("--out-per-protein", type=Path, default=Path("data/knn_precision_all.csv"))
    ap.add_argument("--out-comparison", type=Path, default=Path("data/knn_comparison.csv"))
    args = ap.parse_args()

    gt = [json.loads(line) for line in args.gt.open()]
    rows: list[dict] = []
    for k in args.ks:
        for mode in args.modes:
            d = args.scores_root / f"k{k}_{mode}"
            if not d.exists():
                print(f"[metrics] missing {d}; skipping")
                continue
            for key in args.keys:
                rows += score_dir(d, gt, k, mode, key)

    knn = pd.DataFrame(rows)
    args.out_per_protein.parent.mkdir(parents=True, exist_ok=True)
    # Commit only the tie-broken (`score`) configs per-protein to keep the CSV in
    # git lean; the plain-integer-vote (`-plain`) variants survive in aggregate in
    # knn_comparison.csv below, which is all the plain-vs-tiebreak view needs.
    knn[~knn.model.str.endswith("-plain")].to_csv(args.out_per_protein, index=False)
    print(f"[metrics] wrote {len(knn)} KNN rows ({(~knn.model.str.endswith('-plain')).sum()} "
          f"tie-broken, per-protein) -> {args.out_per_protein}", flush=True)

    base = pd.read_csv(args.base_csv)
    combined = pd.concat([base, knn], ignore_index=True)
    # Group by mode too: protenix-v2 carries both single_seq and msa under one
    # model name, and they must not be averaged together.
    agg = (combined.groupby(["model", "mode", "predictor", "range", "cut"])["precision"]
           .mean().reset_index().rename(columns={"precision": "mean_precision"}))
    agg.to_csv(args.out_comparison, index=False)
    print(f"[metrics] wrote per-model means -> {args.out_comparison}", flush=True)

    # Headline: long-range R-precision, KNN k-sweep next to the reference predictors.
    lr = agg[(agg.range == "long") & (agg.cut == "R")].sort_values("mean_precision")
    print("\nlong-range R-precision (mean over eval set):")
    for _, r in lr.iterrows():
        print(f"  {r['mean_precision']:.3f}  {r['model']} ({r['mode']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
