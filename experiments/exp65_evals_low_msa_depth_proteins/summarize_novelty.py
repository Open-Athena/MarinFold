# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize the fold-novelty verdicts from the exp41 Foldseek query.

Each ``data/<source>_vs_afdb_reps_similarity.csv`` (written by
``exp41_evals_foldseek_train_similarity/query_similarity.py``) carries one
row per candidate with its nearest AFDB-training representative, that rep's
split, the TM-score, the free sequence identity (``best_train_fident``), and a
``verdict`` (redundant >= 0.9 / same_fold >= 0.5 / novel_fold). This collapses
the three candidate sources into one table + a stacked verdict bar, so we can
read off how far each source actually sits from MarinFold's training set.

This is the **fold-novelty axis** of the 2-D eval label
(``notes/low-msa-eval-curation.md`` section 6); the MSA-depth axis comes from
``msa_depth.py`` once ColabFold MSAs are computed for these candidates.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
VERDICT_ORDER = ["redundant", "same_fold", "novel_fold"]
VERDICT_COLORS = {"redundant": "#888888", "same_fold": "#3366cc", "novel_fold": "#dc3912"}

# Each source -> (similarity CSV, human label).
SOURCES = {
    "de novo (PDB)": HERE / "data" / "denovo_vs_afdb_reps_similarity.csv",
    "CASP FM": HERE / "data" / "casp_fm_vs_afdb_reps_similarity.csv",
    "CAMEO hard": HERE / "data" / "cameo_hard_vs_afdb_reps_similarity.csv",
}


def load() -> pd.DataFrame:
    """Concatenate the available similarity CSVs, tagged with their source."""
    frames = []
    for label, path in SOURCES.items():
        if not path.exists():
            print(f"  (skipping {label}: {path.name} not found yet)")
            continue
        df = pd.read_csv(path)
        df.insert(0, "source", label)
        frames.append(df)
    if not frames:
        raise SystemExit("no *_vs_afdb_reps_similarity.csv found; run the query first")
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame, out_csv: Path) -> pd.DataFrame:
    """Print + persist a per-source verdict breakdown; return the combined df."""
    df.to_csv(out_csv, index=False)
    counts = (
        df.groupby(["source", "verdict"]).size().unstack(fill_value=0)
        .reindex(columns=VERDICT_ORDER, fill_value=0)
    )
    counts["n"] = counts.sum(axis=1)
    # Median nearest-train TM (the field the verdict keys on) per source.
    med = df.groupby("source")["best_train_qtmscore"].median().round(3)
    counts["median_train_qtm"] = med
    print(f"\nWrote combined table -> {out_csv}  ({len(df)} candidates)\n")
    print(counts.to_string())
    return counts


def plot(counts: pd.DataFrame, out_path: Path) -> Path:
    sources = list(counts.index)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bottom = [0] * len(sources)
    for v in VERDICT_ORDER:
        vals = counts[v].tolist()
        ax.bar(sources, vals, bottom=bottom, label=v, color=VERDICT_COLORS[v])
        bottom = [b + x for b, x in zip(bottom, vals)]
    for i, src in enumerate(sources):
        ax.text(i, counts.loc[src, VERDICT_ORDER].sum(), f"n={int(counts.loc[src, 'n'])}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("candidates")
    ax.set_title("Fold-novelty vs AFDB-24M training reps (Foldseek TM verdict)")
    ax.legend(title="verdict")
    fig.tight_layout()
    return save_plot_with_meta(
        fig,
        out_path,
        caption=(
            "Fold-novelty of each candidate source vs the 1.33M AFDB-24M training "
            "representatives (exp41 Foldseek query; qtmscore verdict, redundant>=0.9 / "
            "same_fold>=0.5 / novel_fold). Compare to FoldBench-100's 1/100 novel."
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-csv", type=Path, default=HERE / "data" / "novelty_by_source.csv")
    p.add_argument("--out-plot", type=Path, default=HERE / "plots" / "novelty_by_source.png")
    args = p.parse_args()
    df = load()
    counts = summarize(df, args.out_csv)
    path = plot(counts, args.out_plot)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
