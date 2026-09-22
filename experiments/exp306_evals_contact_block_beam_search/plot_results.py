#!/usr/bin/env python
"""Plot paired R-precision, fold-mode coverage, and inference cost."""

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta


def main() -> None:
    """Render compact figures from the committed summary tables."""
    val = pd.read_csv("data/eval_val_beam4.csv")
    fold = pd.read_csv("data/foldswitch_beam4.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, region in zip(axes, ("all", "long")):
        group = val[val["range"] == region]
        ax.scatter(group.iid_r_precision, group.beam_r_precision, s=18, alpha=0.7)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_title(f"{region} range; mean Δ={group.paired_delta.mean():+.3f}")
        ax.set_xlabel("iid R-precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("beam width 4 R-precision")
    fig.tight_layout()
    save_plot_with_meta(fig, "plots/eval_val_paired.png",
                        caption="Paired 100-rollout eval-val R-precision, exp277 iid versus contact-block beam width 4.")
    plt.close(fig)

    primary = fold[fold.primary & (fold.split == "test")]
    labels = ["iid pool", "beam pool", "iid blind", "beam blind"]
    counts = [primary.iid_dual_pool.sum(), primary.dual_pool.sum(),
              primary.iid_dual_blind.sum(), primary.dual_blind.sum()]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color=["#7caeaa", "#e29563", "#7caeaa", "#e29563"])
    ax.bar_label(bars)
    ax.set_ylim(0, max(counts) + 3)
    ax.set_ylabel(f"Proteins with both contact modes (n={len(primary)})")
    ax.set_title("Held-out fold-switching coverage at 100 rollouts")
    fig.tight_layout()
    save_plot_with_meta(fig, "plots/foldswitch_dual.png",
                        caption="Oracle pool and reference-blind 16-map shortlist; contact-level fold evidence only.")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(fold.L, fold.time_ratio, s=25, alpha=0.75)
    ax.axhline(1, color="black", linewidth=1)
    ax.set_xlabel("Sequence length (residues)")
    ax.set_ylabel("Beam / iid H100 generation time")
    ax.set_title("Per-protein inference cost")
    fig.tight_layout()
    save_plot_with_meta(fig, "plots/foldswitch_time.png",
                        caption="Matched 100-rollout pure generation time from saved per-protein timing tables.")
    plt.close(fig)


if __name__ == "__main__":
    main()
