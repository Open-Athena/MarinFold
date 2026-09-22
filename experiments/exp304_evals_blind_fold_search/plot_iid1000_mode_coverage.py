#!/usr/bin/env python
"""Plot cumulative contact-mode coverage of 29 proteins through 1000 iid draws."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
CURVE = HERE / "data" / "iid1000_primary_test_curve.csv"
OUTPUT = HERE / "plots" / "iid1000_mode_coverage.png"
COLORS = {"both": "#126a62", "one": "#ca7d24", "neither": "#5b6595"}
LABELS = {"both": "Both modes", "one": "One mode", "neither": "Neither mode"}


def main() -> None:
    """Show the recorded sequence and its random-order conditional mean."""
    curve = pd.read_csv(CURVE)
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.1), sharex=True, sharey=True)
    fig.patch.set_facecolor("white")
    for axis, order_mean in zip(axes, (False, True), strict=True):
        for category in ("both", "one", "neither"):
            values = curve[f"expected_{category}" if order_mean else category]
            axis.plot(
                curve.budget, values, color=COLORS[category], linewidth=2.5,
                drawstyle="default" if order_mean else "steps-post",
                label=LABELS[category],
            )
            axis.text(1014, float(values.iloc[-1]), f"{values.iloc[-1]:.0f}",
                      va="center", ha="left", color=COLORS[category], fontsize=10,
                      fontweight="bold")
        axis.axvline(500, color="#565656", linewidth=1.1, linestyle="--", alpha=0.7)
        axis.grid(axis="y", color="#e4e6e8", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.set_xlim(0, 1050)
        axis.set_ylim(-0.5, 30.5)
        axis.set_xticks([0, 100, 200, 500, 750, 1000])
        axis.set_xlabel("Number of iid rollouts per protein")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Observed draw order", loc="left", fontweight="bold")
    axes[1].set_title("Average over random draw orders", loc="left", fontweight="bold")
    axes[0].set_ylabel("Number of proteins (out of 29)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Does more iid sampling reveal both fold-like contact modes?", fontsize=15,
                 fontweight="bold", y=0.98)
    fig.text(0.5, 0.01,
             "Each protein is in exactly one category. These are reference-aware contact hits, "
             "not verified 3D folds. Dashed line: previous 500-draw limit.\n"
             "Right panel averages permutations of these same 1,000 maps; "
             "it is not a forecast beyond 1,000.",
             ha="center", va="bottom", fontsize=9, color="#4a4a4a")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.19, wspace=0.13)
    save_plot_with_meta(
        fig, OUTPUT, dpi=180,
        caption="Cumulative oracle contact-mode coverage among 29 primary test proteins. "
                "Left: exact sampled order; right: mean over permutations of the observed "
                "1,000 maps for each protein. Fold-like means a reference-aware contact "
                "screening hit, not a verified alternate 3D structure.",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
