#!/usr/bin/env python
"""Plot contact-mode coverage as independent-rollout budget grows."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def main() -> None:
    report = pd.read_csv(DATA / "iid_mode_coverage_summary.csv")
    test = report[report.cohort == "primary_test"].sort_values("budget")
    fig, ax = plt.subplots(figsize=(7, 4.7))
    for field, label, color in [
        ("fold1", "at least one Fold1-like map", "#306b9b"),
        ("fold2", "at least one Fold2-like map", "#b25a2d"),
        ("both", "both modes in the pool", "#4f7660"),
    ]:
        ax.plot(test.budget, test[field] / test.n_proteins, marker="o", linewidth=2,
                color=color, label=label)
    ax.plot(test.budget, test.both_strict / test.n_proteins, marker="s", linestyle="--",
            linewidth=1.5, color="#4f7660", label="both, 50% recall cutoff")
    ax.set(xscale="log", xticks=test.budget.tolist(), xticklabels=test.budget.tolist(),
           ylim=(0, 0.8), xlabel="Independent rollouts per protein",
           ylabel="Fraction of 29 held-out proteins",
           title="Reference-aware contact-mode coverage of iid rollout pools")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    save_plot_with_meta(
        fig, HERE / "plots" / "iid_mode_coverage.png", dpi=180,
        caption="All curves inspect every map with reference contacts; none is a blind "
                "selection rate. Main hit cutoff: at least 10 switching-region predictions, "
                "25% unique-contact recall, and 0.10 enrichment versus the opposing fold. "
                "Dashed curve uses 50% recall with the other conditions unchanged. "
                "The 500-rollout run is a single sampling experiment; smaller budgets are prefixes.",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
