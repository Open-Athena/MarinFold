"""Plot preregistered accuracy deltas and clustering diagnostics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
COLORS = {
    "medoid_k8": "#3b82f6",
    "random_k8": "#93c5fd",
    "medoid_k16": "#ea580c",
    "random_k16": "#fdba74",
}


def plot_dev_deltas() -> None:
    """Plot development deltas with paired bootstrap intervals."""
    frame = pd.read_csv(DATA / "dev_paired_deltas.csv")
    panels = [
        ("validity_gated_oracle_r_precision", "all", "Oracle fixed-R (all)"),
        ("validity_gated_oracle_r_precision", "long", "Oracle fixed-R (long)"),
        ("consensus_r_precision", "all", "Consensus fixed-R (all)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    order = [
        ("medoid_k8", "iid100"),
        ("medoid_k8", "random_k8"),
        ("medoid_k16", "iid100"),
        ("medoid_k16", "random_k16"),
    ]
    labels = ["k8 vs iid", "k8 vs random", "k16 vs iid", "k16 vs random"]
    for axis, (metric, region, title) in zip(axes, panels, strict=True):
        subset = frame[(frame.metric == metric) & (frame["range"] == region)]
        rows = []
        for arm, comparator in order:
            row = subset[(subset.arm == arm) & (subset.comparator == comparator)]
            if len(row) != 1:
                raise ValueError(f"missing {metric}/{region}/{arm}/{comparator}")
            rows.append(row.iloc[0])
        values = np.asarray([row.mean_delta for row in rows], dtype=float)
        lower = values - np.asarray([row.ci95_low for row in rows], dtype=float)
        upper = np.asarray([row.ci95_high for row in rows], dtype=float) - values
        colors = [COLORS[row.arm] for row in rows]
        axis.barh(range(4), values, color=colors, alpha=0.9)
        axis.errorbar(values, range(4), xerr=[lower, upper], fmt="none", color="black")
        axis.axvline(0, color="black", linewidth=0.8)
        if metric.startswith("validity") and region == "all":
            axis.axvline(0.005, color="#15803d", linestyle="--", linewidth=1)
        if metric.startswith("validity") and region == "long":
            axis.axvline(-0.005, color="#b91c1c", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("paired mean delta")
        axis.set_yticks(range(4), labels if axis is axes[0] else [])
        axis.invert_yaxis()
        axis.grid(axis="x", alpha=0.2)
    fig.suptitle("Natural development: 50 iid + 50 seeded branches vs controls")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "01_dev_paired_deltas.png",
        caption=(
            "Protein-paired development deltas with 95% bootstrap intervals. "
            "Green/red dashed lines mark the preregistered all/long gate thresholds."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_cluster_diagnostics() -> None:
    """Show map-cluster separation versus sequence length."""
    frame = pd.read_csv(DATA / "natural-dev_plan_diagnostics.csv")
    frame = frame[frame.bundle_size == 8]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    scatter = axis.scatter(
        frame.L,
        frame.silhouette,
        c=frame.n_clusters,
        cmap="viridis",
        s=70,
        edgecolor="black",
        linewidth=0.4,
    )
    for row in frame.itertuples():
        axis.annotate(
            row.stem,
            (row.L, row.silhouette),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xlabel("sequence length")
    axis.set_ylabel("selected-clustering silhouette")
    axis.set_title("Whole-map basin separation on the frozen development set")
    axis.grid(alpha=0.2)
    colorbar = fig.colorbar(scatter, ax=axis)
    colorbar.set_label("selected k")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "02_cluster_diagnostics.png",
        caption=(
            "Silhouette of the deterministic k-medoids choice for each protein; "
            "the two seed sizes share the same clustering."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_foldswitch() -> None:
    """Plot fold-switch oracle mode coverage when the panel has been run."""
    path = DATA / "dev_foldswitch_summary.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path)
    frame = frame[frame.map_variant == "continuation"]
    order = ["iid100", "random_k8", "medoid_k8", "random_k16", "medoid_k16"]
    frame = frame.set_index("arm").loc[order].reset_index()
    x = np.arange(len(frame))
    width = 0.25
    fig, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.bar(x - width, frame.oracle_fold1, width, label="fold 1")
    axis.bar(x, frame.oracle_fold2, width, label="fold 2")
    axis.bar(x + width, frame.oracle_dual, width, label="both")
    axis.set_xticks(x, frame.arm, rotation=20, ha="right")
    axis.set_ylabel("pairs reached (of 15)")
    axis.set_title("Fold-switch development oracle mode coverage")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "03_foldswitch_modes.png",
        caption="Number of frozen fold-switch pairs for which each 100-map pool reaches each mode or both modes.",
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    """Regenerate every plot available from committed result tables."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_dev_deltas()
    plot_cluster_diagnostics()
    plot_foldswitch()
    choice_path = DATA / "frozen_choice.json"
    if choice_path.exists():
        print(json.loads(choice_path.read_text()))


if __name__ == "__main__":
    main()
