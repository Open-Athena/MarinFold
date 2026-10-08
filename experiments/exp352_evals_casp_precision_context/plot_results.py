"""Render cached eval-val precision and MSA-depth results; no rescoring."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_summary import save_plot_with_meta
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Create the publication-size figure from the committed summary table."""
    frame = pd.read_csv(HERE / "data/summary.csv")
    frame = frame[frame.definition.eq("cb8")]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(12.5, 5.3), gridspec_kw={"width_ratios": [1.08, 1]}
    )
    colors = {"all": "#288a86", "long": "#335bb5"}
    cuts = ["L/5", "L/2", "L"]
    for offset, span, name in (
        (-0.13, "all", "All nonlocal pairs (sep ≥6)"),
        (0.13, "long", "Long-range pairs (sep ≥24)"),
    ):
        subset = (
            frame[frame.cohort.eq("eval-val") & frame["range"].eq(span)]
            .set_index("cut")
            .loc[cuts]
        )
        x = np.arange(3) + offset
        axes[0].errorbar(
            x,
            subset["mean"],
            yerr=np.stack(
                [subset["mean"] - subset.ci_low, subset.ci_high - subset["mean"]]
            ),
            fmt="o",
            capsize=4,
            markersize=7,
            color=colors[span],
            label=name,
        )
        for px, value in zip(x, subset["mean"]):
            axes[0].text(
                px,
                value + 0.059,
                f"{value:.1%}",
                ha="center",
                fontsize=10,
                color=colors[span],
            )
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=cuts,
        ylim=(0, 1),
        xlim=(-0.5, 2.5),
        title="Precision among the top-ranked pairs",
        ylabel="Mean per-protein precision",
        xlabel="Number of predicted contacts selected per protein",
    )
    axes[0].legend(loc="lower left", frameon=False, fontsize=10)
    groups = ["MSA_10_99", "MSA_100_999", "MSA_ge1000"]
    subset = (
        frame[frame["range"].eq("long") & frame.cut.eq("L/5")]
        .set_index("cohort")
        .loc[groups]
    )
    x = np.arange(3)
    axes[1].bar(x, subset["mean"], width=0.55, color=["#b8c8e5", "#7899cc", "#335bb5"])
    axes[1].errorbar(
        x,
        subset["mean"],
        yerr=np.stack(
            [subset["mean"] - subset.ci_low, subset.ci_high - subset["mean"]]
        ),
        fmt="none",
        capsize=4,
        color="#23334d",
    )
    for px, value in zip(x, subset["mean"]):
        axes[1].text(
            px,
            0.03,
            f"{value:.1%}",
            ha="center",
            color="#152d4c",
            fontsize=12,
            fontweight="bold",
        )
    axes[1].set(
        xticks=x,
        xticklabels=[
            f"10–99\n(n={subset.iloc[0]['n']})",
            f"100–999\n(n={subset.iloc[1]['n']})",
            f"≥1,000\n(n={subset.iloc[2]['n']})",
        ],
        ylim=(0, 1),
        title="Long-range P@L/5 depends on the target mix",
        xlabel="Recorded ColabFold MSA depth (MarinFold uses no MSA)",
    )
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
    fig.suptitle(
        "MarinFold exp277 · 97 natural eval-val monomers",
        fontsize=17,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.01,
        "Cβ <8 Å (Cα for glycine); 100 resampled rollouts/protein; protein-bootstrap 95% intervals. No eval-val proteins have MSA depth <10.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.93))
    save_plot_with_meta(
        fig,
        HERE / "plots/precision_and_depth.png",
        caption="CASP-style contacts on eval-val. Top-k precision and MSA-depth strata, with 20,000 protein-bootstrap draws. The depth strata describe the targets; MarinFold receives a single sequence.",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
