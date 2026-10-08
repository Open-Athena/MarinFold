"""Plot the complete 133-target Helico decoy-ranking evaluation."""

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
    """Compare full-set target-macro rank correlations."""
    rows = load_csv(DATA / "full_metric_summary.csv")
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
    colors = ["#157f8c", "#4aa5ad", "#0f5964", "#e07a2d", "#8a70b3", "#777777"]

    fig, axis = plt.subplots(figsize=(9, 5.2))
    bars = axis.bar(
        range(len(wanted)), values, color=colors, yerr=[lower, upper], capsize=4
    )
    axis.set_xticks(range(len(wanted)), wanted, rotation=24, ha="right")
    axis.set_ylabel("Mean target-wise Spearman correlation with TM-score")
    axis.set_ylim(0.65, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.set_title("Complete AF2Rank Rosetta-decoy benchmark: 133 targets")
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.009,
            f"{value:.3f}",
            ha="center",
        )
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "full_metric_comparison.png",
        caption=(
            "Mean target-wise Spearman correlation across all 133 AF2Rank targets; "
            "error bars are 95% target-bootstrap intervals."
        ),
    )
    plt.close(fig)


def plot_native_selection() -> None:
    """Compare complete-set native recovery and native rank."""
    rows = load_csv(DATA / "full_native_summary.csv")
    wanted = [
        "Helico composite",
        "AF2Rank composite",
        "Helico mean CA pLDDT",
        "AF2 pTM",
        "Helico mean-sample pTM",
        "Helico pTM",
    ]
    by_method = {row["method"]: row for row in rows}
    top1 = [float(by_method[method]["mean_native_top1"]) for method in wanted]
    mean_rank = [float(by_method[method]["mean_native_rank"]) for method in wanted]
    top1_lower = [
        value - float(by_method[method]["native_top1_ci_low"])
        for method, value in zip(wanted, top1, strict=True)
    ]
    top1_upper = [
        float(by_method[method]["native_top1_ci_high"]) - value
        for method, value in zip(wanted, top1, strict=True)
    ]
    rank_lower = [
        value - float(by_method[method]["native_rank_ci_low"])
        for method, value in zip(wanted, mean_rank, strict=True)
    ]
    rank_upper = [
        float(by_method[method]["native_rank_ci_high"]) - value
        for method, value in zip(wanted, mean_rank, strict=True)
    ]
    colors = ["#0f5964", "#e07a2d", "#4aa5ad", "#efaa70", "#2c919c", "#157f8c"]
    positions = list(range(len(wanted)))

    fig, (top1_axis, rank_axis) = plt.subplots(
        1, 2, figsize=(11.5, 5.7), sharey=True, gridspec_kw={"wspace": 0.08}
    )
    top1_bars = top1_axis.barh(
        positions,
        [100 * value for value in top1],
        color=colors,
        xerr=[
            [100 * value for value in top1_lower],
            [100 * value for value in top1_upper],
        ],
        capsize=4,
    )
    top1_axis.set_yticks(positions, wanted)
    top1_axis.invert_yaxis()
    top1_axis.set_xlim(0, 104)
    top1_axis.set_xlabel("Targets with native ranked #1 (%)")
    top1_axis.grid(axis="x", alpha=0.22)
    for bar, value in zip(top1_bars, top1, strict=True):
        top1_axis.text(
            100 * value - 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{round(133 * value):d}/133",
            ha="right",
            va="center",
            color="white",
        )

    rank_limit = max(
        value + upper for value, upper in zip(mean_rank, rank_upper, strict=True)
    )
    rank_bars = rank_axis.barh(
        positions,
        mean_rank,
        color=colors,
        xerr=[rank_lower, rank_upper],
        capsize=4,
    )
    rank_axis.set_xlim(0, rank_limit * 1.12)
    rank_axis.set_xlabel("Mean native rank (lower is better)")
    rank_axis.grid(axis="x", alpha=0.22)
    rank_axis.tick_params(axis="y", left=False, labelleft=False)
    for bar, value, upper in zip(rank_bars, mean_rank, rank_upper, strict=True):
        rank_axis.text(
            value + upper + 0.01 * rank_limit,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
        )

    fig.suptitle("Native identification across all 133 AF2Rank targets")
    fig.text(
        0.5,
        0.01,
        "Error bars: 95% target-bootstrap interval. DeepAccNet and Rosetta omitted: "
        "their AF2Rank native rows contain -1 sentinel scores.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.16, top=0.88)
    save_plot_with_meta(
        fig,
        PLOTS / "full_native_selection.png",
        caption=(
            "Native top-1 recovery and mean native rank over the complete benchmark; "
            "error bars are 95% target-bootstrap intervals."
        ),
    )
    plt.close(fig)


def main() -> None:
    """Generate complete-benchmark figures."""
    PLOTS.mkdir(exist_ok=True)
    plot_metric_comparison()
    plot_native_selection()


if __name__ == "__main__":
    main()
