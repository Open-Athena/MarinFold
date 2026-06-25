# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Append rollout+resample to exp89's unified predictor-comparison table.

To compare rollout+resample against ESMFold / ESMFold2 / Protenix-v2 /
MarinFold-pairwise under ONE metric implementation, its contact metrics must be
computed by exp89's *exact* metric code — our exp82 ``metrics()`` disagrees with
exp89's ``compute_metrics.py`` by up to 0.4/protein (float16 tie-breaking + small
proteins). So the metric functions below are **copied verbatim from exp89's
compute_metrics.py** (keep them identical). We read the saved rollout+resample
vote-score matrices (``score_rollout_resample_eval.py``), compute
precision@{L,L/2,L/5,R} + AUC per range over the resolved universe, label the rows
``model=marinfold-cv1-rollout-resample`` / ``predictor=lm``, and concat them onto
exp89's ``contact_precision_all.csv``.

Run in a venv with pandas + scikit-learn (exp89's venv has them)::

    <exp89-venv>/bin/python build_comparison_table.py \
        --gt <gt_universe.jsonl> --scores _scratch/scores_rollout_resample \
        --exp89-csv <exp89>/data/contact_precision_all.csv \
        --out _scratch/contact_precision_with_rollout.csv \
        --summary data/eval_comparison_summary.csv
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

LABEL_PLAIN = "marinfold-cv1-rollout-resample"
LABEL_TIE = "marinfold-cv1-rollout-resample-tiebreak"


def tiebreak_matrix(count, pairwise):
    """count primary; pairwise breaks ties — count + min-max(pairwise) scaled to [0, 0.5).

    The vote counts are integers (gaps >= 1), so adding a pairwise term bounded to
    [0, 0.5) only ever reorders pairs that are *tied* on votes; it cannot move a
    pair across a count boundary. min-max is monotonic, so within a tie group this
    is identical to ranking by the raw pairwise score.
    """
    iu = np.triu_indices(count.shape[0], k=1)
    s = pairwise[iu]
    lo, hi = float(s.min()), float(s.max())
    return count + (pairwise - lo) / (hi - lo + 1e-9) * 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True, help="rollout+resample vote matrices")
    ap.add_argument("--pairwise-scores", type=Path, default=None,
                    help="exp89 pairwise score matrices; if given, also emit the tie-broken model")
    ap.add_argument("--exp89-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()

    gt = [json.loads(line) for line in args.gt.open()]
    rows, n, n_tie = [], 0, 0
    for rec in gt:
        npz = args.scores / f"{rec['dataset']}__{rec['stem']}.npz"
        if not npz.exists():
            continue
        L = rec["L"]
        score = np.load(npz)["score"].astype(np.float64)
        if score.shape != (L, L):
            print(f"  {rec['stem']}: score shape {score.shape} != L={L}; skipping")
            continue
        resolved = np.asarray(rec["resolved"], dtype=np.int64)
        tmat = true_matrix(L, rec["contacts"])
        pi, pj, psep = resolved_pairs(resolved)
        rows += stamp(metric_rows(score, tmat, pi, pj, psep, L, with_precision=True),
                      rec=rec, model=LABEL_PLAIN, mode="single_seq", predictor="lm")
        n += 1
        if args.pairwise_scores is not None:
            pwz = args.pairwise_scores / f"{rec['dataset']}__{rec['stem']}.npz"
            if pwz.exists():
                pw = np.load(pwz)["score"].astype(np.float64)
                if pw.shape == (L, L):
                    comb = tiebreak_matrix(score, pw)
                    rows += stamp(metric_rows(comb, tmat, pi, pj, psep, L, with_precision=True),
                                  rec=rec, model=LABEL_TIE, mode="single_seq", predictor="lm")
                    n_tie += 1
    print(f"rollout+resample scored: {n}/{len(gt)} | tie-break: {n_tie} -> {len(rows)} rows")

    new = pd.DataFrame(rows)
    existing = pd.read_csv(args.exp89_csv)
    existing = existing[~existing.model.isin([LABEL_PLAIN, LABEL_TIE])]  # idempotent
    combined = pd.concat([existing, new], ignore_index=True)
    combined.to_csv(args.out, index=False)
    print(f"wrote {len(combined)} rows ({len(new)} new) -> {args.out}")

    agg = (combined.groupby(["model", "predictor", "range", "cut"])["precision"]
           .mean().reset_index().rename(columns={"precision": "mean_precision"}))
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.summary, index=False)
    print(f"wrote per-model summary -> {args.summary}")
    print(agg[(agg.range == "long") & (agg.cut.isin(["R", "L", "AUC"]))]
          .pivot_table(index="model", columns="cut", values="mean_precision").round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
