#!/usr/bin/env python
"""Plot development tradeoffs and held-out fold-mode coverage."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"


def pilot_tradeoff(pilot: pd.DataFrame) -> None:
    """Show the cost and fold coverage of each development variant."""
    group = pilot[pilot.cohort == "selection"].copy()
    labels = {
        "b16_e0p0_d0_w10": "width 16, no penalty",
        "b16_e0p05_d0_w10": "width 16, constant 0.05",
        "b16_e0p05_d20_w10": "width 16, early 0.05",
        "b32_e0p05_d20_w10": "width 32, early 0.05",
        "b16_e0p2_d20_w10": "width 16, early 0.2",
    }
    fig, ax = plt.subplots(figsize=(9, 5))
    for row in group.itertuples():
        x = float(row.total_time_ratio_vs_beam4)
        y = int(row.dual_pool)
        ax.scatter(x, y, s=130, zorder=3)
        ax.annotate(f"{labels[row.mode]}\nblind {int(row.dual_blind)}/7",
                    (x, y), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.axhline(int(group.iid_dual_pool.iloc[0]), color="0.45", linestyle="--",
               label=f"iid100 oracle: {int(group.iid_dual_pool.iloc[0])}/7")
    ax.axhline(int(group.beam4_dual_pool.iloc[0]), color="0.7", linestyle=":",
               label=f"width-4 oracle: {int(group.beam4_dual_pool.iloc[0])}/7")
    ax.set(xlabel="H100 inference time / unpenalized width-4 beam",
           ylabel="Development pairs with both contact modes (of 7)",
           title="Contact novelty pilot: discovery versus compute")
    ax.set_ylim(-0.2, 7.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    save_plot_with_meta(fig, PLOTS / "pilot_tradeoff.png",
                        caption="Seven development pairs, 100 rollouts per method. "
                        "Oracle coverage uses reference contacts after generation; "
                        "blind counts are from a presealed 16-map shortlist.")
    plt.close(fig)


def heldout_coverage(test: pd.DataFrame, mode: str) -> None:
    """Compare exactly matched 100-rollout pools and 16-map shortlists."""
    row = test[test.cohort == "selection"].iloc[0]
    labels = ["iid100", "width-4 beam", "selected novelty"]
    pool = [int(row.iid_dual_pool), int(row.beam4_dual_pool), int(row.dual_pool)]
    blind = [int(row.iid_dual_blind), int(row.beam4_dual_blind), int(row.dual_blind)]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.17, pool, width=0.34, label="oracle 100-map pool", color="#477da4")
    ax.bar(x + 0.17, blind, width=0.34, label="blind 16-map shortlist", color="#de8f43")
    for index, (oracle_count, blind_count) in enumerate(zip(pool, blind)):
        ax.text(index - 0.17, oracle_count + 0.25, str(oracle_count), ha="center")
        ax.text(index + 0.17, blind_count + 0.25, str(blind_count), ha="center")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(pool + blind + [1]) + 2)
    ax.set(ylabel="Primary held-out proteins with both contact modes",
           title=f"Held-out fold-mode coverage (n={int(row.n)}, {mode})")
    ax.legend()
    fig.tight_layout()
    save_plot_with_meta(fig, PLOTS / "heldout_coverage.png",
                        caption="The selected method was frozen after development scoring. "
                        "All methods use 100 rollouts per protein. Hits are contact-level "
                        "and do not establish 3D fold recovery.")
    plt.close(fig)


def main() -> None:
    """Regenerate small figures from committed result tables."""
    mode = (DATA / "frozen_choice.txt").read_text().strip()
    pilot = pd.read_csv(DATA / "pilot_comparison.csv")
    test = pd.read_csv(DATA / f"foldswitch_summary_test_{mode}.csv")
    PLOTS.mkdir(exist_ok=True)
    pilot_tradeoff(pilot)
    heldout_coverage(test, mode)
    print("saved pilot_tradeoff.png and heldout_coverage.png")


if __name__ == "__main__":
    main()
