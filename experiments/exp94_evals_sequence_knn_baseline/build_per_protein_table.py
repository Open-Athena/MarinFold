# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build one tidy per-protein comparison table for digging into where we win/lose.

Wide format, one row per eval protein (554), with long-range R-precision for every
predictor side by side plus the analysis-relevant strata (viral flag + source
organism, nearest-train-neighbor identity, fold/seq-leakage labels, length). Built
to seed a "which proteins do we do well/poorly on" investigation (e.g. the viral
out-of-distribution case).

Sources: the KNN per-protein metrics (`data/knn_precision_all.csv`), the existing
predictors' per-protein metrics (`--base-csv`, exp82's with-rollout table), the
taxonomy annotation (`data/eval_taxonomy.csv`), and the KNN hit summary
(`data/knn_hit_summary.csv`).

    uv run python build_per_protein_table.py --base-csv <exp82>/_scratch/contact_precision_with_rollout.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# column name -> (model, mode, predictor) in the metric tables.
PREDICTORS = {
    "marinfold_61_rollout": ("marinfold-cv1-rollout-resample-tiebreak", "single_seq", "lm"),
    "marinfold_lm_pairwise": ("marinfold-contacts-v1", "single_seq", "lm"),
    "marinfold_ens10": ("marinfold-cv1-ens10", "single_seq", "lm"),
    "seq_knn_k10": ("seq-knn-k10", "single_seq", "knn"),
    "seq_knn_k50": ("seq-knn-k50", "single_seq", "knn"),
    "protenix_single_seq": ("protenix-v2", "single_seq", "structure"),
    "protenix_msa": ("protenix-v2", "msa", "structure"),
    "esmfold": ("esmfold", "single_seq", "structure"),
    "esmfold2": ("esmfold2", "single_seq", "structure"),
}


def long_r(df: pd.DataFrame, model: str, mode: str, predictor: str, col: str) -> pd.DataFrame:
    sub = df[(df.model == model) & (df["mode"] == mode) & (df.predictor == predictor)
             & (df.range == "long") & (df.cut == "R")]
    return sub[["dataset", "stem", "precision"]].rename(columns={"precision": col})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", type=Path, required=True)
    ap.add_argument("--knn", type=Path, default=Path("data/knn_precision_all.csv"))
    ap.add_argument("--taxonomy", type=Path, default=Path("data/eval_taxonomy.csv"))
    ap.add_argument("--hit-summary", type=Path, default=Path("data/knn_hit_summary.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/per_protein_comparison.csv"))
    args = ap.parse_args()

    knn = pd.read_csv(args.knn)
    base = pd.read_csv(args.base_csv)
    tax = pd.read_csv(args.taxonomy)
    hs = pd.read_csv(args.hit_summary)
    hs[["dataset", "stem"]] = hs["query"].str.split("__", n=1, expand=True)

    # Spine: every protein + strata (length/fold_verdict/seq_leakage live in base rows).
    # Coalesce across a protein's rows so a NaN in one predictor's rows doesn't win.
    strata = (base[["dataset", "stem", "n_residues", "fold_verdict", "seq_leakage", "neff_tier"]]
              .sort_values("n_residues").groupby(["dataset", "stem"], as_index=False).last())
    out = (tax[["dataset", "stem", "gt_chain", "source_organism", "is_viral"]]
           .merge(strata, on=["dataset", "stem"], how="left")
           .merge(hs[["dataset", "stem", "n_hits", "best_fident"]], on=["dataset", "stem"], how="left")
           .rename(columns={"n_hits": "n_train_seq_hits", "best_fident": "best_train_identity",
                            "n_residues": "length"}))

    for col, (model, mode, predictor) in PREDICTORS.items():
        src = knn if predictor == "knn" else base
        out = out.merge(long_r(src, model, mode, predictor, col), on=["dataset", "stem"], how="left")

    out = out.sort_values(["dataset", "stem"]).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[table] wrote {len(out)} proteins x {out.shape[1]} cols -> {args.out}")
    print("columns:", list(out.columns))
    print("\nmean long-range R by viral flag:")
    print(out.groupby("is_viral")[list(PREDICTORS)].mean().round(3).T)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
