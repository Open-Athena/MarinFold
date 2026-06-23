# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step C — contact-prediction metrics for MarinFold + every other predictor.

Consumes the GT universe (``prepare_gt_universe.py``) and the MarinFold score
matrices (``score_eval_set.py``), and produces one unified, tidy table:

* **MarinFold** (model=``marinfold-contacts-v1``, predictor=``lm``): precision @
  {L, L/2, L/5, R} + **AUC**, per range (all / short / medium / long), scored on
  the exact resolved-residue universe the other predictors use.
* **AUC for the existing predictors** (Protenix-v2 single_seq/msa·structure,
  ESMFold, ESMFold2) computed here from their saved per-pair contacts over the
  same universe — the issue asks for AUC "for all predictors". Their precision
  rows come from exp74/exp78 unchanged and are concatenated in.

Output ``data/contact_precision_all.csv`` is the exp78 schema plus the new
MarinFold + AUC rows — ready for the combined plots.

Run in the exp89 venv::

    uv run python compute_metrics.py \
        --gt data/gt_universe.jsonl --scores _scratch/scores \
        --exp78-precision <exp78>/data/contact_precision_all.csv \
        --exp78-raw _scratch/contacts_raw_all.parquet \
        --exp74-raw _scratch/contacts_raw_exp74.parquet \
        --out data/contact_precision_all.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

RANGES: dict[str, tuple[int, int | None]] = {
    "all": (6, None), "short": (6, 11), "medium": (12, 23), "long": (24, None),
}
CUTS = (
    ("L", lambda L, c: L),
    ("L/2", lambda L, c: max(1, L // 2)),
    ("L/5", lambda L, c: max(1, L // 5)),
    ("R", lambda L, c: c),
)
MIN_DEG, MIN_SEP = 0.001, 6
STRATA_COLS = ["neff_tier", "fold_verdict", "seq_leakage", "msa_neff", "length"]
OUT_COLS = ["dataset", "stem", "n_residues", "model", "mode", "predictor",
            "range", "cut", "precision", "n_candidate", "n_true", "n_top", *STRATA_COLS]


def true_matrix(L: int, contacts) -> np.ndarray:
    m = np.zeros((L, L), bool)
    for i, j, d in contacts:
        i, j = int(i), int(j)
        if d >= MIN_DEG and (j - i) >= MIN_SEP and i < j < L:
            m[i, j] = True
    return m


def degree_matrix(L: int, rows) -> np.ndarray:
    m = np.zeros((L, L))
    for i, j, d in rows:
        i, j = int(i), int(j)
        if i < j < L:
            m[i, j] = max(m[i, j], float(d))
    return m


def resolved_pairs(resolved: np.ndarray):
    a, b = np.triu_indices(len(resolved), k=1)
    i, j = resolved[a], resolved[b]
    return i, j, (j - i)


def metric_rows(score, tmat, pi, pj, psep, L, *, with_precision: bool) -> list[dict]:
    """precision@{L,L/2,L/5,R} (optional) + AUC, per range."""
    cs, cg = score[pi, pj], tmat[pi, pj].astype(int)
    rows: list[dict] = []
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


def load_gt(path: Path) -> list[dict]:
    """One record per (dataset, stem); 7ur7_A/8ah9_A appear in two datasets."""
    return [json.loads(line) for line in path.open()]


def stamp(rows, *, rec, model, mode, predictor) -> list[dict]:
    strata = rec.get("strata", {}) or {}
    base = dict(dataset=rec["dataset"], stem=rec["stem"], n_residues=rec["L"],
                model=model, mode=mode, predictor=predictor)
    for k in STRATA_COLS:
        base[k] = strata.get(k)
    return [{**base, **r} for r in rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True,
                    help="primary MarinFold model scores dir (eric's #61/#75 2.7566)")
    ap.add_argument("--extra", action="append", default=[],
                    help="additional MarinFold model(s) as label=dir (repeatable); e.g. the "
                         "#67 model or the K=10 ensemble. npz['score'] is read.")
    ap.add_argument("--exp78-precision", type=Path, required=True)
    ap.add_argument("--exp78-raw", type=Path, required=True)
    ap.add_argument("--exp74-raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gt = load_gt(args.gt)
    new_rows: list[dict] = []

    # --- MarinFold model(s): precision + AUC from the saved score matrices ---
    models = [("marinfold-contacts-v1", args.scores)]
    for spec in args.extra:
        label, _, d = spec.partition("=")
        models.append((label, Path(d)))
    for label, scores_dir in models:
        n_scored = 0
        for rec in gt:
            npz = scores_dir / f"{rec['dataset']}__{rec['stem']}.npz"
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
            rows = metric_rows(score, tmat, pi, pj, psep, L, with_precision=True)
            new_rows += stamp(rows, rec=rec, model=label, mode="single_seq", predictor="lm")
            n_scored += 1
        print(f"[metrics] {label} scored: {n_scored}/{len(gt)} proteins")

    # --- AUC for the existing predictors over the same universe ---
    def auc_for(raw: pd.DataFrame, model: str, mode: str):
        pred = raw[(raw.role == "pred") & (raw["mode"] == mode)]
        by_key = {(d, s): list(g[["i", "j", "degree"]].itertuples(index=False, name=None))
                  for (d, s), g in pred.groupby(["dataset", "stem"])}
        added = 0
        for rec in gt:
            key = (rec["dataset"], rec["stem"])
            if key not in by_key:
                continue
            L = rec["L"]
            resolved = np.asarray(rec["resolved"], dtype=np.int64)
            tmat = true_matrix(L, rec["contacts"])
            pi, pj, psep = resolved_pairs(resolved)
            score = degree_matrix(L, by_key[key])
            rows = metric_rows(score, tmat, pi, pj, psep, L, with_precision=False)
            new_rows.extend(stamp(rows, rec=rec, model=model, mode=mode, predictor="structure"))
            added += 1
        print(f"[metrics] AUC added for {model}/{mode}: {added} proteins")

    exp78_raw = pd.read_parquet(args.exp78_raw)
    for m in ("esmfold", "esmfold2"):
        sub = exp78_raw[exp78_raw.model == m].copy()
        sub["mode"] = "single_seq"
        auc_for(sub, m, "single_seq")
    exp74_raw = pd.read_parquet(args.exp74_raw)
    for mode in ("single_seq", "msa"):
        auc_for(exp74_raw, "protenix-v2", mode)

    new_df = pd.DataFrame(new_rows)
    # MarinFold-only convenience table (both models if present)
    new_df[new_df.model.str.startswith("marinfold")].to_csv(
        args.out.parent / "marinfold_precision.csv", index=False)

    # Unified table: existing exp78 precision rows + everything new.
    existing = pd.read_csv(args.exp78_precision)
    combined = pd.concat([existing, new_df], ignore_index=True)
    for c in OUT_COLS:
        if c not in combined.columns:
            combined[c] = np.nan
    combined = combined[OUT_COLS + [c for c in combined.columns if c not in OUT_COLS]]
    combined.to_csv(args.out, index=False)
    print(f"[metrics] wrote {len(combined)} rows ({len(new_df)} new) -> {args.out}")
    print("models x cuts:\n", combined.groupby(["model", "cut"]).size().unstack(fill_value=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
