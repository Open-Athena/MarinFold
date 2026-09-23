"""Render the experiment's accuracy, coverage, and fold-mode figures."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"
COLORS = {"iid100": "#555555", "random_k5": "#e69f00", "cluster_k5": "#0072b2"}


def plot_paired_deltas() -> None:
    """Plot cluster-k5 oracle and consensus effects on both natural splits."""
    rows = []
    for split in ("dev", "heldout"):
        frame = pd.read_csv(DATA / f"{split}_paired_deltas.csv")
        frame = frame[
            (frame.arm == "cluster_k5")
            & (frame["range"] == "all")
            & frame.metric.isin(
                [
                    "validity_gated_oracle_r_precision",
                    "consensus_r_precision",
                ]
            )
        ].copy()
        frame["split"] = split
        rows.append(frame)
    data = pd.concat(rows, ignore_index=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    labels = [
        ("dev", "validity_gated_oracle_r_precision", "Dev oracle"),
        ("dev", "consensus_r_precision", "Dev consensus"),
        ("heldout", "validity_gated_oracle_r_precision", "Held-out oracle"),
        ("heldout", "consensus_r_precision", "Held-out consensus"),
    ]
    comparators = [
        ("iid100", "vs iid100", "#0072b2"),
        ("random_k5", "vs random k=5", "#d55e00"),
    ]
    for row_index, (split, metric, label) in enumerate(labels):
        for offset, (comparator, comparator_label, color) in zip(
            (-0.12, 0.12), comparators
        ):
            item = data[
                (data.split == split)
                & (data.metric == metric)
                & (data.comparator == comparator)
            ].iloc[0]
            y = len(labels) - 1 - row_index + offset
            ax.errorbar(
                item.mean_delta,
                y,
                xerr=[
                    [item.mean_delta - item.ci95_low],
                    [item.ci95_high - item.mean_delta],
                ],
                fmt="o",
                color=color,
                capsize=3,
                label=comparator_label if row_index == 0 else None,
            )
    ax.axvline(0, color="#777777", linewidth=1)
    ax.set_yticks(range(len(labels)), [item[2] for item in reversed(labels)])
    ax.set_xlabel("Paired mean delta in fixed-R precision")
    ax.set_title("Five-contact cluster branching: oracle and consensus effects")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "01_paired_accuracy_deltas.png",
        caption=(
            "Protein-paired bootstrap mean differences for the primary continuation-only map. "
            "The held-out split was opened only after k=5 cleared the development gate."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_per_protein() -> None:
    """Show which held-out proteins gain under the frozen policy."""
    scores = pd.read_csv(DATA / "heldout_natural.csv")
    scores = scores[
        (scores.map_variant == "continuation")
        & (scores["range"] == "all")
        & scores.arm.isin(["iid100", "cluster_k5"])
    ]
    diagnostic = pd.read_csv(DATA / "natural-heldout_plan_diagnostics.csv")
    diagnostic = diagnostic[diagnostic.bundle_size == 5].set_index("stem")
    pivot = scores.pivot(index="stem", columns="arm")
    fallback = diagnostic.loc[pivot.index, "fallback_fraction"].astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    specs = [
        ("validity_gated_oracle_r_precision", "Oracle best-of-100"),
        ("consensus_r_precision", "Consensus"),
    ]
    for ax, (metric, title) in zip(axes, specs, strict=True):
        x = pivot[metric]["iid100"]
        y = pivot[metric]["cluster_k5"]
        for is_fallback, label, color, marker in (
            (False, "stable k=5 cluster", "#0072b2", "o"),
            (True, "no k=5 cluster; fallback", "#999999", "x"),
        ):
            keep = fallback == is_fallback
            ax.scatter(
                x[keep],
                y[keep],
                s=32,
                color=color,
                marker=marker,
                label=label,
                alpha=0.85,
            )
        low = min(float(x.min()), float(y.min()))
        high = max(float(x.max()), float(y.max()))
        ax.plot([low, high], [low, high], color="#555555", linewidth=1)
        ax.set_xlabel("iid100")
        ax.set_ylabel("50 iid + 50 cluster-policy branches")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Frozen held-out split: per-protein effects")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "02_heldout_per_protein.png",
        caption=(
            "Each point is one of 81 held-out eval-val proteins. Blue points had at least one "
            "split-half-stable five-contact cluster; gray crosses used the preregistered coherent-random fallback."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_cluster_coverage() -> None:
    """Plot stable-cluster availability against sequence length."""
    dev = pd.read_csv(DATA / "natural-dev_plan_diagnostics.csv").assign(split="dev")
    heldout = pd.read_csv(DATA / "natural-heldout_plan_diagnostics.csv").assign(
        split="heldout"
    )
    data = pd.concat([dev, heldout], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    for bundle_size, color in ((3, "#56b4e9"), (5, "#0072b2")):
        frame = data[data.bundle_size == bundle_size]
        axes[0].scatter(
            frame.L,
            frame.n_stable_clusters,
            s=24,
            alpha=0.7,
            color=color,
            label=f"k={bundle_size}",
        )
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_xlabel("sequence length")
    axes[0].set_ylabel("eligible stable clusters")
    axes[0].legend(frameon=False)
    axes[0].grid(alpha=0.2)
    availability = (
        data.assign(available=data.n_stable_clusters > 0)
        .groupby(["split", "bundle_size"], as_index=False)
        .available.mean()
    )
    for index, split in enumerate(("dev", "heldout")):
        frame = availability[availability.split == split]
        axes[1].bar(
            np.arange(len(frame)) + (index - 0.5) * 0.35,
            frame.available,
            width=0.35,
            label=split,
            color=("#009e73" if split == "dev" else "#cc79a7"),
        )
    axes[1].set_xticks([0, 1], ["k=3", "k=5"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("fraction with an eligible cluster")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.suptitle("Truth-free split-half cluster coverage")
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "03_cluster_coverage.png",
        caption=(
            "Three-contact clusters are broadly available; the stricter five-contact arm falls back "
            "on about half the proteins, an important limitation of the frozen policy."
        ),
        dpi=180,
    )
    plt.close(fig)


def plot_foldswitch() -> None:
    """Plot fold-switch oracle mode coverage if that stress test is complete."""
    path = DATA / "dev_foldswitch_summary.csv"
    if not path.exists():
        return
    data = pd.read_csv(path)
    data = data[
        (data.map_variant == "continuation")
        & data.arm.isin(
            ["iid100", "random_k3", "cluster_k3", "random_k5", "cluster_k5"]
        )
    ].copy()
    order = [
        arm
        for arm in ["iid100", "random_k3", "cluster_k3", "random_k5", "cluster_k5"]
        if arm in set(data.arm)
    ]
    data = data.set_index("arm").loc[order]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(data))
    ax.bar(
        x - 0.2, data.oracle_fold1, width=0.2, label="fold 1 reached", color="#56b4e9"
    )
    ax.bar(x, data.oracle_fold2, width=0.2, label="fold 2 reached", color="#e69f00")
    ax.bar(x + 0.2, data.oracle_dual, width=0.2, label="both reached", color="#009e73")
    ax.set_xticks(x, order)
    ax.set_ylabel("pairs out of 15")
    ax.set_title("Fold-switch development stress test")
    ax.legend(frameon=False, ncol=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "04_foldswitch_modes.png",
        caption=(
            "Oracle contact-mode coverage on 15 frozen fold-switch development pairs, using the "
            "same total 100-map budget and excluding supplied contacts from branch continuations."
        ),
        dpi=180,
    )
    plt.close(fig)


def main() -> None:
    """Render every result figure from committed tables."""
    plot_paired_deltas()
    plot_per_protein()
    plot_cluster_coverage()
    plot_foldswitch()


if __name__ == "__main__":
    main()
