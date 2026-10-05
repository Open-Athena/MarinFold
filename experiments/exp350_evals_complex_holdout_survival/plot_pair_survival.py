"""Compare component-held-out and pair-held-out candidate yields."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
SOURCES = [
    ("foldbench", "FoldBench PPI", "#2B6CB0"),
    ("pinder", "PINDER test", "#DD6B20"),
]


def stage_count(path: Path, source: str, stage: str) -> int:
    """Read one all-complex count from a survival table."""
    for row in csv.DictReader(path.open()):
        if (
            row["source"] == source
            and row["complex_type"] == "all"
            and row["stage"] == stage
        ):
            return int(row["remaining"])
    raise ValueError(f"Missing {source}/{stage} in {path}")


def main() -> None:
    """Render the yield change caused by switching the holdout definition."""
    strict = HERE / "data/survival.csv"
    pair = HERE / "data/pair_survival.csv"
    stages = [
        ("Component-held-out", strict, "after_esm_atlas"),
        ("Pair-held-out\nMarinFold", pair, "after_pinder_pair"),
        ("Pair-held-out +\nHelico PDB screen", pair, "after_helico_finetune_pair"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    x = list(range(len(stages)))
    offsets = {"foldbench": -0.10, "pinder": 0.10}
    for source, label, color in SOURCES:
        values = [stage_count(path, source, stage) for _, path, stage in stages]
        shown = [max(value, 0.5) for value in values]
        ax.plot(
            [position + offsets[source] for position in x],
            shown,
            marker="o",
            linewidth=0,
            markersize=10,
            color=color,
            label=label,
        )
        for position, plot_value, value in zip(x, shown, values, strict=True):
            ax.annotate(
                str(value),
                (position + offsets[source], plot_value),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                color=color,
                fontweight="bold",
            )
    ax.set_yscale("log")
    ax.set_ylim(0.35, 400)
    ax.set_xticks(x, [label for label, _, _ in stages])
    ax.set_ylabel("Candidate dimers remaining (log scale)")
    ax.set_title("Holdout definition determines whether the benchmark is feasible")
    ax.grid(axis="y", which="both", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.01,
        0.02,
        "Pair-held-out means both candidate chains co-occur as homologs in one training document.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        HERE / "plots/pair_survival.png",
        caption=(
            "The strict component-held-out rule rejects a dimer when either chain has "
            "a training homolog. The pair-held-out rule rejects it only when one complex "
            "training document contains one-to-one homologs for both candidate chains. "
            "The final column additionally applies a conservative same-PDB pair screen "
            "against Helico's documented fine-tuning pool."
        ),
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
