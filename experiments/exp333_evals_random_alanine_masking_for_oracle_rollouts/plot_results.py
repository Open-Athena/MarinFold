"""Plot the development oracle and useful-diversity results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"


def fraction_from_mode(mode: str) -> float:
    """Extract percent alanine masking from a constructed mode name."""
    raw = mode.split("mask_p")[-1]
    return int(raw) / 10


def plot_mutant_deltas() -> None:
    """Show matched all-mutant oracle deltas against the zero-mask worker."""
    deltas = pd.read_csv(DATA / "dev_paired_deltas_vs_zero.csv")
    deltas = deltas[
        deltas["mode"].str.startswith("mutant100_mask_")
        & (deltas["mode"] != "mutant100_mask_p0000")
    ].copy()
    deltas["fraction"] = deltas["mode"].map(fraction_from_mode)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = {"all": "#2b6cb0", "long": "#c53030"}
    offsets = {"all": -0.7, "long": 0.7}
    for region in ("all", "long"):
        frame = deltas[deltas["range"] == region].sort_values("fraction")
        x = frame.fraction.to_numpy() + offsets[region]
        y = frame.mean_delta.to_numpy()
        error = np.vstack([y - frame.ci95_low.to_numpy(), frame.ci95_high.to_numpy() - y])
        ax.errorbar(
            x,
            y,
            yerr=error,
            marker="o",
            capsize=4,
            linewidth=1.8,
            color=colors[region],
            label=f"{region}-range",
        )
    ax.axhline(0, color="#555555", linewidth=1)
    ax.set_xticks([5, 10, 20, 40, 100], labels=["5", "10", "20", "40", "100"])
    ax.set_xlabel("non-alanine residues replaced per rollout (%)")
    ax.set_ylabel("paired delta in validity-gated oracle best@100")
    ax.set_title("Random alanine masking reduces oracle accuracy at every tested rate")
    ax.grid(alpha=0.2)
    ax.legend()
    save_plot_with_meta(
        fig,
        PLOTS / "01_mutant_oracle_deltas.png",
        caption=(
            "Paired protein-bootstrap mean deltas against the zero-mutation vLLM worker on "
            "the frozen 16-protein eval-val development set. Bars are 95% intervals. Even "
            "5% masking significantly reduces all- and long-range oracle best@100."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_mixed_gate() -> None:
    """Show the fixed-budget mixed pools against the published iid100 pool."""
    deltas = pd.read_csv(DATA / "dev_paired_deltas.csv")
    modes = ["mix50_mask_p0050", "mix50_mask_p0100", "mix50_top2"]
    labels = {
        "mix50_mask_p0050": "50 iid + 50 at 5%",
        "mix50_mask_p0100": "50 iid + 50 at 10%",
        "mix50_top2": "50 iid + 25 each at 5%, 10%",
    }
    frame = deltas[deltas["mode"].isin(modes)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, region in zip(axes, ("all", "long"), strict=True):
        selected = frame[frame["range"] == region].set_index("mode").loc[modes]
        y = np.arange(len(modes))
        means = selected.mean_delta.to_numpy()
        errors = np.vstack(
            [means - selected.ci95_low.to_numpy(), selected.ci95_high.to_numpy() - means]
        )
        ax.errorbar(means, y, xerr=errors, fmt="o", capsize=4, color="#2f855a")
        ax.axvline(0, color="#555555", linewidth=1)
        threshold = 0.005 if region == "all" else -0.005
        ax.axvline(threshold, color="#d69e2e", linestyle="--", linewidth=1.2)
        ax.set_title(f"{region}-range")
        ax.set_xlabel("paired oracle best@100 delta vs iid100")
        ax.set_yticks(y, labels=[labels[mode] for mode in modes])
        ax.grid(alpha=0.2, axis="x")
    fig.suptitle("Mixed pools gain slightly all-range but fail the long-range safety gate")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "02_mixed_pool_gate.png",
        caption=(
            "Paired deltas against #321's published iid100 pool. Dashed lines mark the "
            "preregistered all-range improvement threshold (+0.005) and long-range safety "
            "threshold (-0.005). The two all-range-positive pools lose about two points "
            "long-range, so no policy advances."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_diversity_tradeoff() -> None:
    """Show that lower overlap did not translate into useful true-contact coverage."""
    summary = pd.read_csv(DATA / "dev_summary.csv")
    frame = summary[
        summary["mode"].str.startswith("mutant100_mask_")
        & (summary["range"] == "all")
    ].copy()
    frame["fraction"] = frame["mode"].map(fraction_from_mode)
    frame = frame.sort_values("fraction")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].plot(frame.fraction, frame.mean_pairwise_jaccard, marker="o", color="#805ad5")
    axes[0].set_ylabel("mean pairwise map Jaccard")
    axes[0].set_title("Raw diversity")
    axes[1].plot(frame.fraction, frame.true_union_recall, marker="o", color="#2b6cb0")
    axes[1].set_ylabel("true-contact union recall")
    axes[1].set_title("Useful coverage")
    for ax in axes:
        ax.set_xlabel("alanine masking (%)")
        ax.set_xticks([0, 5, 10, 20, 40, 100])
        ax.grid(alpha=0.2)
    fig.suptitle("Masking creates different maps, but their true-contact coverage shrinks")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "03_diversity_tradeoff.png",
        caption=(
            "All-range development means for 100 masked rollouts. Overlap falls through 40%, "
            "but true-contact union recall also falls. At 100% masking nearly every rollout "
            "ends immediately; its high empty-map Jaccard is a degenerate endpoint."
        ),
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    """Build all committed result plots."""
    PLOTS.mkdir(exist_ok=True)
    plot_mutant_deltas()
    plot_mixed_gate()
    plot_diversity_tradeoff()


if __name__ == "__main__":
    main()
