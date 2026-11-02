# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Figures for the decontamination pass — the three that carry the argument.

1. **Survival per tier.** What each tier costs each corpus. The gap between
   Tier A and Tier C is the whole H1-vs-H0 question in one panel.
2. **Axis decomposition.** Of the documents Tier C removes from AFDB, how many
   a sequence filter would already have caught, and how many are
   structure-only. The structure-only bar is the direct measure of whether
   #41's warning was worth acting on.
3. **Why Tier C costs what it costs.** The distribution of best TM from the 554
   eval structures to AFDB training cluster representatives, with the 0.5 and
   0.9 lines drawn. If the mass sits above 0.5, fold-level disjointness is
   expensive by construction, not by an unlucky threshold.

Plus the E-value sensitivity curve, which says whether Tier A's headline is a
statement about contamination or about an mmseqs flag.

    uv run python plot_decontam.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402
from decontam_lib import (  # noqa: E402
    ARM_AFDB,
    ARM_ESM,
    ARM_LABELS,
    STRUCT_FOLD_TM,
    STRUCT_REDUNDANT_TM,
    TIERS,
    TIER_LABELS,
)

HERE = Path(__file__).resolve().parent
ARM_COLOUR = {ARM_AFDB: "#3f6ea8", ARM_ESM: "#c26a3d"}


def plot_survival(by_tier: pd.DataFrame, out: Path) -> None:
    """Grouped bars: percent of each corpus dropped, per tier."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    width = 0.38
    for i, arm in enumerate((ARM_AFDB, ARM_ESM)):
        rows = by_tier[by_tier["arm"] == arm].set_index("tier").reindex(TIERS)
        offsets = [j + (i - 0.5) * width for j in range(len(TIERS))]
        values = rows["pct_dropped"].fillna(0.0)
        ax.bar(offsets, values, width, label=ARM_LABELS[arm], color=ARM_COLOUR[arm])
        for x, (tier, value) in zip(offsets, rows["pct_dropped"].items()):
            if pd.isna(value):
                ax.text(x, 0.4, "not\nmeasurable", ha="center", va="bottom",
                        fontsize=7, color="0.35", style="italic")
            else:
                ax.text(x, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([TIER_LABELS[t] for t in TIERS], fontsize=9)
    ax.set_ylabel("training documents dropped (%)")
    ax.set_title("What each decontamination tier costs the two training corpora")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_plot_with_meta(
        fig, str(out),
        caption="Share of each published train corpus removed at each tier. ESM-Atlas has "
                "no structural database, so its Tier B/C cells are unmeasured rather than "
                "zero — building one is the job this table gates.",
    )


#: Stack order, shared by both panels of the decomposition.
AXIS_PARTS = (
    ("sequence_only", "caught by sequence only", "#3f6ea8"),
    ("both", "caught by both axes", "#7f9dc0"),
    ("structure_only", "caught by structure only", "#b5482f"),
)


def _stacked_axes(ax, rows: pd.DataFrame) -> None:
    bottom = pd.Series(0.0, index=rows.index)
    for column, label, colour in AXIS_PARTS:
        ax.bar(range(len(rows)), rows[column], 0.5, bottom=bottom, label=label, color=colour)
        bottom = bottom + rows[column]
    for i, total in enumerate(bottom):
        ax.text(i, total, f"{total:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([TIER_LABELS[t] for t in rows.index], fontsize=8)
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)


def plot_axis_decomposition(by_axis: pd.DataFrame, out: Path) -> None:
    """Stacked bars per tier (AFDB), split into two panels.

    Tier C is fifteen times Tier B, so one shared y-axis would flatten A and B
    into invisibility — and the A-vs-B difference is exactly the cheap
    structural decontamination the tier ladder exists to price. Two panels at
    their own scales, with the ratio stated in the caption.
    """
    rows = by_axis[by_axis["arm"] == ARM_AFDB].set_index("tier").reindex(TIERS)
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(8.4, 4.2), gridspec_kw={"width_ratios": [2, 1]}
    )
    _stacked_axes(left, rows.loc[["A", "B"]])
    _stacked_axes(right, rows.loc[["C"]])
    left.set_ylabel("AFDB training documents dropped")
    left.set_title("Tiers A and B", fontsize=10)
    right.set_title("Tier C (note the scale)", fontsize=10)
    left.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Which axis is doing the work (AFDB arm)")
    fig.tight_layout()
    save_plot_with_meta(
        fig, str(out),
        # Two lines is the caption budget: three overlap the figure's own title.
        caption="Red = what #41 predicted and nobody had measured — documents a sequence "
                "filter keeps and a structural one removes: 22,320 at Tier B, 1,462,710 at C.",
    )


def plot_tm_distribution(clusters: pd.DataFrame, out: Path) -> None:
    """Where the eval set sits relative to the training folds."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(clusters["best_qtm"], bins=60, color="#3f6ea8", alpha=0.85)
    ax.set_yscale("log")
    top = ax.get_ylim()[1]
    for threshold, label, colour in (
        (STRUCT_FOLD_TM, f"same fold (TM {STRUCT_FOLD_TM})", "#b5482f"),
        (STRUCT_REDUNDANT_TM, f"redundant (TM {STRUCT_REDUNDANT_TM})", "#6a3d9a"),
    ):
        ax.axvline(threshold, color=colour, ls="--", lw=1.2)
        ax.annotate(f" {label}", xy=(threshold, top * 0.05), rotation=90,
                    va="bottom", ha="left", fontsize=7, color=colour)
    ax.set_xlabel("best TM-score to any of the 554 eval structures (query-normalised)")
    ax.set_ylabel("AFDB train cluster representatives (log)")
    ax.set_title("How close the training folds are to the eval set")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_plot_with_meta(
        fig, str(out),
        caption="945,861 of the 1,304,911 AFDB train representatives got a hit and are "
                "counted here; the other ~359k had none and are off the left edge.",
    )


def plot_evalue_sensitivity(sweep: pd.DataFrame, out: Path) -> None:
    """Is Tier A a statement about contamination or about an mmseqs flag?"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for arm in (ARM_AFDB, ARM_ESM):
        ax.plot(sweep["report_evalue_ceiling"], sweep[f"{arm}_pct_dropped"],
                marker="o", color=ARM_COLOUR[arm], label=ARM_LABELS[arm])
    ax.axvline(10.0, color="0.4", ls="--", lw=1.0)
    ax.text(10.0, ax.get_ylim()[1], " tier ceiling (exp65 / #213)", rotation=90,
            va="top", ha="left", fontsize=7, color="0.4")
    ax.set_xscale("log")
    ax.set_xlabel("MMseqs2 reporting ceiling applied at reduce time (E-value)")
    ax.set_ylabel("training documents dropped by Tier A (%)")
    ax.set_title("How much of Tier A is the threshold rather than the contamination")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_plot_with_meta(
        fig, str(out),
        caption="The E-value arm of Tier A contributes nothing above 1e-3; everything the "
                "curve gains after that is the identity arm, which has no significance "
                "floor of its own.",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=HERE / "data")
    ap.add_argument("--work", type=Path, default=Path("/data/exp225_decontam"))
    ap.add_argument("--plots", type=Path, default=HERE / "plots")
    args = ap.parse_args()
    args.plots.mkdir(parents=True, exist_ok=True)

    plot_survival(pd.read_csv(args.data / "survival_by_tier.csv"),
                  args.plots / "survival_by_tier.png")
    plot_axis_decomposition(pd.read_csv(args.data / "survival_by_axis.csv"),
                            args.plots / "axis_decomposition.png")
    plot_tm_distribution(pd.read_parquet(args.work / "droplist_structure_afdb.parquet"),
                         args.plots / "tm_distribution.png")
    sweep = args.data / "evalue_sensitivity.csv"
    if sweep.exists():
        plot_evalue_sensitivity(pd.read_csv(sweep), args.plots / "evalue_sensitivity.png")
    else:
        print(f"[plot] {sweep} not built; skipping the sensitivity panel", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
