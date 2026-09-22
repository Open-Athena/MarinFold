#!/usr/bin/env python
"""Compare contact-screen and structure-screen iid coverage curves."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE / "plots" / "helico_iid_structural_coverage.png"
COLORS = {"both": "#126a62", "one": "#ca7d24", "neither": "#5b6595"}


def draw(axis, curve: pd.DataFrame, title: str) -> None:
    """Draw one mutually exclusive three-state coverage curve."""
    for category, label in (("both", "Both modes"), ("one", "One mode"),
                            ("neither", "Neither mode")):
        axis.plot(curve.budget, curve[category], color=COLORS[category], linewidth=2.5,
                  drawstyle="steps-post", label=label)
        axis.text(1012, float(curve[category].iloc[-1]), f"{curve[category].iloc[-1]:.0f}",
                  color=COLORS[category], va="center", fontweight="bold")
    axis.axvline(500, color="#666666", linestyle="--", linewidth=1)
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_xlim(0, 1050)
    axis.set_ylim(-0.5, 30.5)
    axis.set_xticks([0, 100, 200, 500, 750, 1000])
    axis.grid(axis="y", color="#e4e6e8")
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlabel("Number of iid rollouts per protein")


def main() -> None:
    contact = pd.read_csv(DATA / "iid1000_primary_test_curve.csv")
    structure = pd.read_csv(DATA / "helico_iid_structural_curve.csv")
    final = structure.iloc[-1]
    n_assessable = int(final.assessable_neither + final.assessable_one
                       + final.assessable_both)
    fig, axes = plt.subplots(1, 2, figsize=(12.7, 5.1), sharex=True, sharey=True)
    draw(axes[0], contact, "Contact-screen oracle")
    draw(axes[1], structure, "Individual Helico structures (control-gated)")
    axes[0].set_ylabel("Number of proteins (out of 29)")
    axes[0].legend(loc="upper right", frameon=False)
    fig.suptitle("Does iid sampling reveal both experimentally observed folds?",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5, 0.015,
        "Post-hoc structural hit: common-position Kabsch GDT-TS ≥ max(0.35, 90% of the true-contact "
        "control) and switching-region advantage ≥ 0.10.\n"
        f"Each Helico prediction uses one individual MarinFold map; both controls pass for "
        f"{n_assessable}/29 proteins; dashed line: 500 draws.",
        ha="center", va="bottom", fontsize=9, color="#444444",
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.19, wspace=0.13)
    save_plot_with_meta(
        fig, OUTPUT, dpi=180,
        caption="Cumulative oracle coverage of two fold modes under the contact-map screen "
                "and under cross-reference structural scoring of every individual Helico output. "
                "The structural thresholds are shown on the figure; raw cross-reference Kabsch GDT-TS "
                "and CA RMSD values are retained for threshold sensitivity analyses.",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
