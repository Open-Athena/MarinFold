"""Plot the target-balanced Helico decoy-ranking pilot."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"


def load_csv(path: Path) -> list[dict[str, str]]:
    """Load a CSV file."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_metric_comparison() -> None:
    """Compare pilot target-macro Spearman correlations."""
    rows = load_csv(DATA / "pilot_metric_summary.csv")
    wanted = [
        "Helico pTM",
        "Helico mean CA pLDDT",
        "Helico composite",
        "AF2Rank composite",
        "DeepAccNet",
        "Rosetta energy",
    ]
    by_method = {row["method"]: row for row in rows}
    values = [float(by_method[method]["mean_spearman_tmscore"]) for method in wanted]
    lower = [
        value - float(by_method[method]["spearman_tmscore_ci_low"])
        for method, value in zip(wanted, values, strict=True)
    ]
    upper = [
        float(by_method[method]["spearman_tmscore_ci_high"]) - value
        for method, value in zip(wanted, values, strict=True)
    ]

    fig, axis = plt.subplots(figsize=(9, 5.2))
    colors = ["#157f8c", "#4aa5ad", "#0f5964", "#e07a2d", "#8a70b3", "#777777"]
    bars = axis.bar(
        range(len(wanted)), values, color=colors, yerr=[lower, upper], capsize=4
    )
    axis.set_xticks(range(len(wanted)), wanted, rotation=24, ha="right")
    axis.set_ylabel("Mean target-wise Spearman correlation with TM-score")
    axis.set_ylim(0.7, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("AF2Rank Rosetta-decoy pilot: 9 targets x 24 TM-stratified decoys")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
        )
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "pilot_spearman_comparison.png",
        caption=(
            "Mean target-wise Spearman correlation with reference TM-score; bars are "
            "95% target-bootstrap intervals. Helico pTM is the primary score; the "
            "Helico composite additionally uses candidate/output TM-score."
        ),
    )
    plt.close(fig)


def plot_ptm_scatter() -> None:
    """Show Helico pTM across the reference TM-score range."""
    predictions = load_csv(DATA / "pilot_candidate_metrics.csv")
    truth = {
        (row["target"], row["decoy_id"]): row
        for row in load_csv(DATA / "pilot_candidates.csv")
    }
    targets = sorted({row["target"] for row in predictions})
    color_map = plt.get_cmap("tab10")

    fig, axis = plt.subplots(figsize=(8.2, 6.2))
    for target_index, target in enumerate(targets):
        rows = [row for row in predictions if row["target"] == target]
        decoys = [row for row in rows if row["decoy_id"] != "native"]
        native = next(row for row in rows if row["decoy_id"] == "native")
        color = color_map(target_index)
        axis.scatter(
            [
                float(truth[(row["target"], row["decoy_id"])]["tmscore"])
                for row in decoys
            ],
            [float(row["max_ptm"]) for row in decoys],
            s=24,
            alpha=0.68,
            color=color,
            label=target,
        )
        axis.scatter(
            [1.0],
            [float(native["max_ptm"])],
            s=105,
            marker="*",
            edgecolor="black",
            linewidth=0.6,
            color=color,
        )
    axis.set_xlabel("Candidate TM-score to native")
    axis.set_ylabel("Helico pTM (best of 3 diffusion samples)")
    axis.set_xlim(0.15, 1.02)
    axis.set_ylim(0.25, 1.0)
    axis.grid(alpha=0.22)
    axis.legend(title="Target", ncol=3, fontsize=8, title_fontsize=9)
    axis.set_title(
        "Helico confidence tracks candidate quality, but natives are not always top-1"
    )
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "pilot_ptm_vs_tmscore.png",
        caption=(
            "Each circle is a TM-stratified Rosetta decoy; stars are native structures. "
            "Colors denote the nine length/count-stratified targets."
        ),
    )
    plt.close(fig)


def main() -> None:
    """Generate all pilot figures."""
    PLOTS.mkdir(exist_ok=True)
    plot_metric_comparison()
    plot_ptm_scatter()


if __name__ == "__main__":
    main()
