"""Plot the measured upper-bound survival through locally available training arms."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
STAGES = [
    ("quality_and_scope", "Scope + quality"),
    ("after_afcdb", "AFCDB"),
    ("after_pinder", "PINDER train"),
    ("after_afdb", "AFDB"),
    ("after_esm_atlas", "ESM-Atlas"),
]
COLORS = {"foldbench": "#2B6CB0", "pinder": "#DD6B20"}


def main() -> None:
    """Render the survival curves with exact counts annotated."""
    rows = list(csv.DictReader((HERE / "data/survival.csv").open()))
    counts = {
        (row["source"], row["stage"]): int(row["remaining"])
        for row in rows
        if row["complex_type"] == "all"
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = list(range(len(STAGES)))
    for source, label in [("foldbench", "FoldBench PPI"), ("pinder", "PINDER test")]:
        values = [counts[source, stage] for stage, _ in STAGES]
        plotted = [max(value, 0.5) for value in values]
        ax.plot(
            x,
            plotted,
            marker="o",
            linewidth=2.5,
            markersize=7,
            color=COLORS[source],
            label=label,
        )
        for index, (shown, value) in enumerate(zip(plotted, values, strict=True)):
            ax.annotate(
                str(value),
                (index, shown),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                color=COLORS[source],
                fontweight="bold",
            )
    ax.set_yscale("log")
    ax.set_ylim(0.35, 4000)
    ax.set_xticks(x, [label for _, label in STAGES])
    ax.set_ylabel("Candidate dimers remaining (log scale)")
    ax.set_title("30% identity / 50% shorter-sequence survival")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.02,
        "ProteinMPNN redesign arms are not searched; they can only reduce the final upper bound of one.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        HERE / "plots/survival.png",
        caption=(
            "Natural protein-only dimers after scope/quality filters and cumulative "
            "sequence exclusion against four locally available exp343 training arms. "
            "Every constituent chain must avoid a >=30% identity alignment covering "
            ">=50% of the shorter sequence. The remaining PINDER target is an upper "
            "bound because the ProteinMPNN redesign corpora were not searched."
        ),
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
