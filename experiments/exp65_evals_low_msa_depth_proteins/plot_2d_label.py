# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the 2-D eval label: fold-novelty x MSA-depth, + per-dataset depth.

Reads ``data/candidate_2d_label.csv`` (from ``combine_axes.py``) and renders
two panels into ``plots/two_axis_label.png``:

- a heatmap of fold-novelty (rows) x MSA-depth tier (cols) with per-cell
  counts -- the headline grid; the bottom-right "novel_fold x orphan/low" is
  the regime FoldBench-100 couldn't reach;
- per-dataset stacked MSA-depth-tier bars, showing which sources deliver the
  shallow tail.
"""

import argparse
import collections
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
FOLD_ORDER = ["redundant", "same_fold", "novel_fold"]
TIER_ORDER = ["orphan", "low", "marginal", "deep"]
DATASETS = {"denovo_pdb": "de novo", "casp_fm": "CASP FM", "cameo_hard": "CAMEO hard"}
TIER_COLORS = {"orphan": "#d7301f", "low": "#fc8d59", "marginal": "#fdcc8a", "deep": "#bdbdbd"}


def plot(df: pd.DataFrame, out_path: Path) -> Path:
    fig, (ax_h, ax_b) = plt.subplots(1, 2, figsize=(12.5, 4.6),
                                     gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel 1: fold x MSA-depth heatmap. Global grid -> dedup by stem so the 3
    # proteins cross-listed in both the de novo and CAMEO sets aren't counted
    # twice (the per-dataset bars below keep them under both, by membership).
    uniq = df.drop_duplicates(subset="stem")
    grid = np.zeros((len(FOLD_ORDER), len(TIER_ORDER)), dtype=int)
    counts = collections.Counter(zip(uniq["fold_verdict"], uniq["neff_tier"]))
    for i, f in enumerate(FOLD_ORDER):
        for j, t in enumerate(TIER_ORDER):
            grid[i, j] = counts.get((f, t), 0)
    im = ax_h.imshow(grid, cmap="Blues", aspect="auto")
    ax_h.set_xticks(range(len(TIER_ORDER)), labels=TIER_ORDER)
    ax_h.set_yticks(range(len(FOLD_ORDER)), labels=FOLD_ORDER)
    ax_h.set_xlabel("MSA depth (Neff tier)")
    ax_h.set_ylabel("fold novelty")
    ax_h.set_title("Candidates by fold-novelty x MSA-depth")
    thresh = grid.max() / 2
    for i in range(len(FOLD_ORDER)):
        for j in range(len(TIER_ORDER)):
            ax_h.text(j, i, grid[i, j], ha="center", va="center",
                      color="white" if grid[i, j] > thresh else "black", fontsize=11)
    # Highlight the hard corner (novel_fold x orphan/low).
    ax_h.add_patch(plt.Rectangle((-0.5, 2.5), 2, 1, fill=False, edgecolor="#d7301f", lw=2.5))

    # Panel 2: per-dataset stacked MSA-depth tiers.
    labels = list(DATASETS.values())
    bottoms = np.zeros(len(labels))
    for tier in TIER_ORDER:
        vals = [int(((df["dataset"] == ds) & (df["neff_tier"] == tier)).sum()) for ds in DATASETS]
        ax_b.bar(labels, vals, bottom=bottoms, label=tier, color=TIER_COLORS[tier])
        bottoms += vals
    ax_b.set_ylabel("candidates")
    ax_b.set_title("MSA-depth tier by dataset")
    ax_b.legend(title="Neff tier", fontsize=8)

    fig.tight_layout()
    return save_plot_with_meta(
        fig, out_path,
        caption=(
            "exp65 2-D eval label. Left: fold-novelty (vs AFDB-24M train reps, "
            "Foldseek) x MSA-depth (Neff, ColabFold); red box = the novel-fold x "
            "shallow-MSA corner FoldBench-100 couldn't populate. Right: which "
            "datasets supply the shallow-MSA tail."
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=HERE / "data" / "candidate_2d_label.csv")
    p.add_argument("--out", type=Path, default=HERE / "plots" / "two_axis_label.png")
    args = p.parse_args()
    df = pd.read_csv(args.csv)
    df = df[df["neff_tier"].isin(TIER_ORDER)]  # drop any pending/missing
    print(f"wrote {plot(df, args.out)}")


if __name__ == "__main__":
    main()
