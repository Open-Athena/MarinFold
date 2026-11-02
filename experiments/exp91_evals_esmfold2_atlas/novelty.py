# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Estimate how structurally novel the Atlas is vs MarinFold's training set.

MarinFold trains on ``afdb-24M``. exp41 already built and published a Foldseek
DB of all 1.33 M afdb-24M structural-cluster **representatives**
(``hf://buckets/silterra/afdb-24M-foldseek-train-reps``) plus the reusable
``query_similarity.py`` that ``foldseek easy-search``es candidate structures
against it and emits a per-candidate ``novel_fold | same_fold | redundant``
verdict (TM thresholds 0.5 / 0.9 on ``qtmscore``).

This script is a thin orchestrator: it points exp41's tool at our decoded Atlas
structure sample (``data/struct_sample/*.pdb``) and summarizes the result —
what fraction of sampled Atlas folds have *no* near structural match in our
training reps (TM < 0.5), i.e. would add genuinely new structural clusters.

Prereqs:
- ``hf buckets sync hf://buckets/silterra/afdb-24M-foldseek-train-reps db_full``
  (the ~2.6 GB DB → ``db_full/db/targetDB*`` + ``db_full/reps_manifest.csv``).
- Foldseek is auto-installed by exp41's ``foldseek_env`` on first use.

Usage::

    uv run python novelty.py --db-dir db_full
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXP41 = HERE.parent / "exp41_evals_foldseek_train_similarity"


def run_query_similarity(candidate_dir: Path, db_dir: Path, out_csv: Path,
                         fold_tm: float, redundant_tm: float) -> None:
    """Shell out to exp41's query_similarity.py (run in its own dir/venv)."""
    # query_similarity runs with cwd=EXP41, so every path must be absolute.
    db = (db_dir / "db" / "targetDB").resolve()
    manifest = (db_dir / "reps_manifest.csv").resolve()
    candidate_dir = candidate_dir.resolve()
    out_csv = out_csv.resolve()
    for p in (db.with_suffix(".index"), manifest):
        if not p.exists():
            sys.exit(f"missing {p}; run: hf buckets sync "
                     f"hf://buckets/silterra/afdb-24M-foldseek-train-reps {db_dir}")
    cmd = [
        "uv", "run", "python", "query_similarity.py",
        "--candidate-dir", str(candidate_dir),
        "--db", str(db),
        "--reps-manifest", str(manifest),
        "--out", str(out_csv),
        "--tm-field", "qtmscore",
        "--fold-tm", str(fold_tm),
        "--redundant-tm", str(redundant_tm),
        "--db-tag", "afdb-24M-full-reps-1331330",
    ]
    print(f"[novelty] running exp41 query_similarity (cwd={EXP41}) …", flush=True)
    subprocess.run(cmd, cwd=EXP41, check=True)


def summarize(out_csv: Path) -> None:
    df = pd.read_csv(out_csv)
    tm = "best_train_qtmscore" if "best_train_qtmscore" in df else "best_qtmscore"
    vc = df["verdict"].value_counts()
    summary = pd.DataFrame([{
        "n_candidates": len(df),
        "frac_novel_fold": float((df["verdict"] == "novel_fold").mean()),
        "frac_same_fold": float((df["verdict"] == "same_fold").mean()),
        "frac_redundant": float((df["verdict"] == "redundant").mean()),
        "median_nearest_train_tm": float(df[tm].median()),
        "frac_tm_lt_0.5": float((df[tm].fillna(0.0) < 0.5).mean()),
    }])
    summary.to_csv(DATA / "novelty_summary.csv", index=False)
    print("[novelty] verdicts:\n", vc.to_string())
    print("[novelty] summary:\n", summary.T.to_string(header=False))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db-dir", type=Path, default=HERE / "db_full")
    ap.add_argument("--candidate-dir", type=Path, default=DATA / "struct_sample")
    ap.add_argument("--out", type=Path,
                    default=DATA / "atlas_vs_afdb_reps_similarity.csv")
    ap.add_argument("--fold-tm", type=float, default=0.5)
    ap.add_argument("--redundant-tm", type=float, default=0.9)
    args = ap.parse_args(argv)

    DATA.mkdir(parents=True, exist_ok=True)
    run_query_similarity(args.candidate_dir, args.db_dir, args.out,
                         args.fold_tm, args.redundant_tm)
    summarize(args.out)


if __name__ == "__main__":
    main()
