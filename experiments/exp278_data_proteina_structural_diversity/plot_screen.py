"""Plot completed screening reports without rerunning inference or alignment."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
CONDITIONS = ["unconditional", "1.x.x.x", "2.x.x.x", "3.x.x.x"]
LABELS = ["Unconditional", "Alpha conditioned", "Beta conditioned", "Mixed conditioned"]
COLORS = ["#777777", "#d95f02", "#1b9e77", "#7570b3"]


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    reports = sorted(
        (HERE / "data").glob("l*-screen"),
        key=lambda path: int(path.name.split("-")[0][1:]),
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for condition, label, color in zip(CONDITIONS, LABELS, COLORS, strict=True):
        rows = [
            row
            for report in reports
            for row in read_csv(report / "quality-by-case.csv")
            if row["condition"] == condition
        ]
        axes[0].plot(
            [int(row["length"]) for row in rows],
            [float(row["quality_yield"]) for row in rows],
            "o-",
            label=label,
            color=color,
        )
        axes[1].scatter(
            [float(row["mean_alpha"]) for row in rows],
            [float(row["mean_beta"]) for row in rows],
            label=label,
            color=color,
            s=50,
        )
        for row in rows:
            axes[1].annotate(
                row["length"],
                (float(row["mean_alpha"]), float(row["mean_beta"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set(
        xlabel="Residues",
        ylabel="Quality pass fraction",
        ylim=(0, 1.03),
        title="Designability depends on length and requested class",
    )
    axes[1].set(
        xlabel="Mean alpha fraction (P-SEA)",
        ylabel="Mean beta fraction (P-SEA)",
        xlim=(0, 1),
        ylim=(0, 0.65),
        title="Measured composition follows conditioning",
    )
    axes[0].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    save_plot_with_meta(
        figure,
        HERE / "plots/quality-and-composition.png",
        caption="One ProteinMPNN/ESMFold attempt per backbone; final cis-aware quality checks. Composition averages all refolded candidates and is an independent P-SEA measurement, not a CATH assignment. Labels on the right denote sequence length.",
    )
    plt.close(figure)

    retained = [
        (
            int(report.name.split("-")[0][1:]),
            json.loads((report / "retention.json").read_text()),
        )
        for report in reports
        if (report / "retention.json").exists()
    ]
    figure, axis = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    for key, label in [
        ("quality_pass", "After quality"),
        ("after_decontamination", "After eval decontamination"),
        ("after_cluster_cap", "After fine-cluster cap"),
    ]:
        axis.plot(
            [length for length, _ in retained],
            [row[key] / row["candidates"] for _, row in retained],
            "o-",
            label=label,
        )
    axis.axhline(
        0.5, color="black", linestyle="--", alpha=0.5, label="50% planning threshold"
    )
    axis.set(
        xlabel="Residues",
        ylabel="Fraction of raw candidates retained",
        ylim=(0, 1.03),
        title="Usable yield includes every filter",
    )
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    save_plot_with_meta(
        figure,
        HERE / "plots/retention.png",
        caption="Screening-set retention. Duplicate growth at 100k/1M remains unmeasured; these are not production-yield forecasts.",
    )
    plt.close(figure)

    cost_path = HERE / "data/cost-by-length.csv"
    if not cost_path.exists():
        return
    costs = read_csv(cost_path)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(
        [int(row["length"]) for row in costs],
        [float(row["gpu_seconds_per_raw_candidate"]) * 1.2 for row in costs],
        "o-",
        label="Per raw candidate + 20%",
    )
    axes[0].plot(
        [int(row["length"]) for row in costs],
        [
            float(row["gpu_seconds_per_retained_document_20pct_overhead"])
            for row in costs
        ],
        "o-",
        label="Per retained document + 20%",
    )
    axes[0].set(
        xlabel="Residues",
        ylabel="H100 seconds",
        title="Low long-chain yield drives cost",
    )
    axes[0].legend(fontsize=8)
    comparisons = [
        (length, row["matched_diversity_after_decontamination"])
        for length, row in retained
    ]
    axes[1].plot(
        [length for length, _ in comparisons],
        [row["ratio_median"] for _, row in comparisons],
        "o-",
        label="Observed ratio (median)",
    )
    axes[1].plot(
        [length for length, _ in comparisons],
        [row["maximum_possible_ratio_median"] for _, row in comparisons],
        "o--",
        label="Ceiling for median control",
    )
    axes[1].axhline(1.5, color="black", linestyle=":", label="Proposed target: 1.5")
    axes[1].set(
        xlabel="Residues",
        ylabel="Conditioned / unconditional",
        title="Fine-cluster target is ceiling-limited",
        ylim=(0.65, 1.6),
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.2)
    save_plot_with_meta(
        figure,
        HERE / "plots/cost-and-diversity.png",
        caption="Reference-precision stage costs and final retained yield. Diversity uses equal-size length-matched samples after decontamination. The ceiling assumes every conditioned sample occupies a unique cluster; this exposes limited power of the proposed small-screen target.",
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
