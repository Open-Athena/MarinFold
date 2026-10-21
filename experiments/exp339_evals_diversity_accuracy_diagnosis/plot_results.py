"""Build figures solely from committed diagnostic and source tables."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Render the report's diagnostic plots with command metadata."""
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    summary = pd.read_csv(HERE / "data/summary.csv")
    summary = summary.query("split == 'test' and range == 'all'")
    colors = ["#2171b5", "#d95f0e"]
    modes = ["full_iid_single", "full_g05_pa_all"]
    names = ["Ordinary iid", "Bounded sequence guidance"]
    fig, ax = plt.subplots(figsize=(11, 5.3))
    metrics = ["mean_emission_fixed_r", "best_emission_fixed_r",
               "best_frequency_fixed_r", "best_within_map_truth_ranking",
               "consensus_published", "union_recall"]
    labels = ["Mean sample", "Best sample\n(emission order)",
              "Best sample\n(independent ranks)*", "Best map\n(truth-ranked ceiling)",
              "Consensus", "Union coverage\n(truth ceiling)"]
    for index, (mode, name, color) in enumerate(zip(modes, names, colors, strict=True)):
        table = summary.query("mode == @mode").set_index("metric").loc[metrics]
        x = np.arange(len(metrics)) + (index - 0.5) * 0.36
        ax.bar(x, table['mean'], width=0.35, label=name, color=color)
        ax.errorbar(x, table['mean'], yerr=[table['mean'] - table.low, table.high - table['mean']],
                    fmt="none", color="#333333", capsize=3, linewidth=1)
    ax.set_xticks(np.arange(len(metrics)), labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fixed-R precision or truth coverage")
    ax.set_title("Most true contacts appear somewhere; no single map collects them")
    ax.legend(loc="upper left", frameon=False)
    fig.text(0.06, 0.01, "81 natural eval-val proteins; 100 candidate maps. *Ranks use 100 extra iid maps.\n"
             "95% protein-bootstrap intervals. Coverage ceilings use truth and are not folding accuracy.", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_plot_with_meta(fig, HERE / "plots/01_headroom.png", caption="New raw-rollout reanalysis; distinct metric definitions must not be treated as interchangeable model scores.")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for mode, name, color in zip(modes, names, colors, strict=True):
        table = summary.query("mode == @mode").set_index("metric")
        budgets = [1, 5, 10, 25, 50, 100]
        axes[0].plot(budgets, [table.loc[f"best_emission_{n}", "mean"] for n in budgets],
                     marker="o", color=color, label=name)
        axes[1].plot(budgets, [table.loc[f"best_map_recall_{n}", "mean"] for n in budgets],
                     marker="o", color=color, label=name)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(n) for n in budgets])
        ax.set_xlabel("Candidate maps")
        ax.set_ylim(0.35, 0.60)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Mean across 81 natural eval-val proteins")
    axes[0].set_title("Oracle emission-order fixed-R")
    axes[1].set_title("Best unordered-map recall ceiling")
    axes[0].legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save_plot_with_meta(fig, HERE / "plots/02_sampling_curve.png", caption="More samples still help, but doubling 50 to 100 iid gives about one percentage point; no extrapolation to infinite sampling is justified.")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    exposure = pd.read_csv(HERE / "data/training_exposure.csv")
    axes[0].barh(exposure.corpus, exposure.token_share * 100, color=["#2171b5", "#6baed6", "#d95f0e", "#fdae6b"])
    axes[0].set_xlabel("Percent of exp277 raw token exposure")
    axes[0].set_title("70% redesigned sequences; reused backbones")
    dedup = pd.read_csv(HERE / "data/redundancy_source.csv")
    axes[1].bar(dedup.min_sequence_identity.astype(str), dedup.removal_fraction * 100, color="#756bb1")
    axes[1].set_xlabel("Minimum sequence identity (both coverages ≥80%)")
    axes[1].set_ylabel("Percent of native documents removed")
    axes[1].set_title("Measured native redundancy in #336")
    fig.text(0.06, 0.01, "Deduplication is candidate- and representative-policy-limited; it is not an exhaustive redundancy bound.\n"
             "100% alignment identity does not mean exact whole-sequence duplication. Structure checks remain incomplete.", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save_plot_with_meta(fig, HERE / "plots/03_data_exposure.png", caption="Sources: exp277 epoch_corpus_counts.csv and exp336 direct_sequence_thresholds.csv at 9aaa1451.")
    plt.close(fig)

    inputs = ["per_protein.csv", "summary.csv", "paired_deltas.csv", "raw_manifest.csv",
              "training_exposure.csv", "redundancy_source.csv"]
    for name in inputs:
        if not (HERE / "data" / name).exists():
            raise FileNotFoundError(name)


if __name__ == "__main__":
    main()
