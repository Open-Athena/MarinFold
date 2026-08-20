# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-protein contact metrics for one or more rollout+resample score dirs.

Generalises ``build_comparison_table.py`` to N models: pass ``--model
label=dir`` once per scored checkpoint. Emits a tidy per-protein CSV in exp89's
schema so the rows concatenate straight onto exp89's committed
``contact_precision_all.csv`` at plot time.

The metric functions are **copied verbatim from exp89's compute_metrics.py**
(via ``build_comparison_table.py`` — keep all three identical). exp82's own
``metrics()`` disagrees with exp89's by up to 0.4/protein on small proteins
(float16 tie-breaking), so anything compared against ESMFold / Protenix has to
be scored by exp89's implementation, not ours.

    uv run python build_rollout_rows.py \
        --gt <gt_universe.jsonl> \
        --model marinfold-cv1-exp75-rollout=_scratch/scores_exp75_nok \
        --model marinfold-cv1-exp117-rollout=_scratch/scores_exp117_nok \
        --out data/where_we_stand_rows.csv.gz --summary data/where_we_stand_summary.csv
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, metavar="LABEL=DIR",
                    help="repeatable; npz['score'] vote matrices for one checkpoint")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    args = ap.parse_args()

    models = []
    for spec in args.model:
        label, _, d = spec.partition("=")
        if not d:
            ap.error(f"--model needs LABEL=DIR, got {spec!r}")
        models.append((label, Path(d)))

    gt = [json.loads(line) for line in args.gt.open()]
    rows = []
    for label, sdir in models:
        n = 0
        for rec in gt:
            npz = sdir / f"{rec['dataset']}__{rec['stem']}.npz"
            if not npz.exists():
                continue
            L = rec["L"]
            score = np.load(npz)["score"].astype(np.float64)
            if score.shape != (L, L):
                print(f"  {label} {rec['stem']}: shape {score.shape} != L={L}; skipping")
                continue
            resolved = np.asarray(rec["resolved"], dtype=np.int64)
            tmat = true_matrix(L, rec["contacts"])
            pi, pj, psep = resolved_pairs(resolved)
            rows += stamp(metric_rows(score, tmat, pi, pj, psep, L, with_precision=True),
                          rec=rec, model=label, mode="single_seq", predictor="lm")
            n += 1
        print(f"{label}: scored {n}/{len(gt)} proteins")
        if n != len(gt):
            print(f"  !! INCOMPLETE — {len(gt) - n} of {len(gt)} units missing for {label}")

    new = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(args.out, index=False)
    print(f"wrote {len(new)} rows -> {args.out}")

    agg = (new.groupby(["model", "range", "cut"])["precision"].mean().reset_index()
           .rename(columns={"precision": "mean_precision"}))
    agg.to_csv(args.summary, index=False)
    print(f"wrote summary -> {args.summary}")
    print(agg[agg.cut.isin(["R", "AUC"])]
          .pivot_table(index="model", columns=["range", "cut"], values="mean_precision").round(4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
