# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot a candidate-set similarity CSV (from ``query_similarity.py``).

Two figures, saved with sidecar metadata for ``build_summary.py``:

1. ``verdict_hist.png`` — count of candidates per verdict
   (novel_fold / same_fold / redundant).
2. ``struct_vs_seq.png`` — nearest-training-rep structural similarity
   (``best_train_qtmscore``) vs sequence identity (``best_train_fident``).
   The point: a single-sequence model can face structures that are
   structurally close to training yet sequence-distant — those sit
   bottom-right, and are exactly what foldseek catches and a sequence
   filter would miss.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

HERE = Path(__file__).resolve().parent
_VERDICT_ORDER = ["novel_fold", "same_fold", "redundant"]
_VERDICT_COLOR = {"novel_fold": "#4477AA", "same_fold": "#CCBB44", "redundant": "#EE6677"}


def plot_verdict_hist(df: pd.DataFrame, out: Path, *, title_suffix: str) -> Path:
    counts = df["verdict"].value_counts().reindex(_VERDICT_ORDER, fill_value=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=[_VERDICT_COLOR[v] for v in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("candidates")
    ax.set_title(f"Train-similarity verdicts ({title_suffix})")
    fig.tight_layout()
    return save_plot_with_meta(
        fig, out,
        caption=(
            f"Per-candidate verdict counts vs the training-set representative DB "
            f"({title_suffix}). Verdict keys off the nearest TRAIN cluster's "
            f"query-normalized TM-score: redundant >= 0.9, same_fold >= 0.5, "
            f"else novel_fold."
        ),
    )


def plot_struct_vs_seq(df: pd.DataFrame, out: Path, *, title_suffix: str) -> Path:
    sub = df.dropna(subset=["best_train_qtmscore", "best_train_fident"])
    fig, ax = plt.subplots(figsize=(6, 5))
    for verdict in _VERDICT_ORDER:
        s = sub[sub["verdict"] == verdict]
        if not s.empty:
            ax.scatter(
                s["best_train_qtmscore"], s["best_train_fident"],
                s=28, alpha=0.8, label=verdict, color=_VERDICT_COLOR[verdict],
            )
    ax.axvline(0.5, ls="--", lw=1, color="grey")
    ax.set_xlabel("structural similarity to nearest train rep (qtmscore)")
    ax.set_ylabel("sequence identity to that rep (fident)")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Structure vs sequence similarity ({title_suffix})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save_plot_with_meta(
        fig, out,
        caption=(
            "Each candidate's nearest TRAIN representative: structural TM (x) vs "
            "sequence identity (y). Bottom-right = structurally close but "
            "sequence-distant, the leakage a sequence-only filter would miss. "
            "Dashed line = 0.5 same-fold boundary."
        ),
    )


def plot_timing(timings_csv: Path, out: Path, *, title_suffix: str) -> Path | None:
    """Per-candidate foldseek-search wall-time vs sequence length."""
    tdf = pd.read_csv(timings_csv)
    per = tdf[tdf["mode"] == "per_candidate"]
    if per.empty:
        return None
    n_db = int(per["n_db_reps"].iloc[0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(per["n_residues"], per["elapsed_seconds"], s=22, alpha=0.8, color="#228833")
    ax.set_xlabel("candidate length (residues)")
    ax.set_ylabel("foldseek easy-search wall-time (s)")
    ax.set_title(f"Per-candidate search time ({title_suffix})")
    med = per["elapsed_seconds"].median()
    ax.axhline(med, ls="--", lw=1, color="grey")
    ax.text(0.98, 0.04, f"median {med*1000:.0f} ms · DB={n_db} reps",
            transform=ax.transAxes, ha="right", fontsize=9, color="#555")
    fig.tight_layout()
    return save_plot_with_meta(
        fig, out,
        caption=(
            f"Per-candidate TM-align search time vs length, against a {n_db}-rep "
            f"target DB. At this tiny DB size the cost is dominated by Foldseek "
            f"per-call setup, not DB scaling — absolute numbers will rise with the "
            f"full ~1.3M-rep training DB. Batched throughput is in the '__ALL__' row."
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--csv", type=Path,
        default=HERE / "data" / "foldbench_vs_full_reps_similarity.csv",
        help="Similarity CSV from query_similarity.py",
    )
    ap.add_argument(
        "--timings-csv", type=Path, default=HERE / "data" / "timings.csv",
        help="Timing CSV from collect_timings.py (plotted if present)",
    )
    ap.add_argument("--suffix", default="FoldBench-100 vs afdb-24M full train reps (1.33M)",
                    help="Label noting which candidate set / DB this is")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    plots = HERE / "plots"
    plot_verdict_hist(df, plots / "verdict_hist.png", title_suffix=args.suffix)
    plot_struct_vs_seq(df, plots / "struct_vs_seq.png", title_suffix=args.suffix)
    made = ["verdict_hist.png", "struct_vs_seq.png"]
    if args.timings_csv.exists():
        if plot_timing(args.timings_csv, plots / "search_timing.png", title_suffix=args.suffix):
            made.append("search_timing.png")
    print(f"Wrote {', '.join('plots/' + m for m in made)}")


if __name__ == "__main__":
    main()
