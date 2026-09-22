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
    group = pilot[pilot.cohort == "selection"].set_index("mode")
    first = group.iloc[0]
    beam4_seconds = first.total_beam_seconds / first.total_time_ratio_vs_beam4
    iid_cost = float(first.total_iid_seconds / beam4_seconds)
    rows = [
        ("iid100", iid_cost, int(group.iid_dual_pool.iloc[0]),
         int(group.iid_dual_blind.iloc[0])),
        ("width 4", 1.0, int(group.beam4_dual_pool.iloc[0]),
         int(group.beam4_dual_blind.iloc[0])),
        ("width 16, no penalty", float(group.loc["b16_e0p0_d0_w10", "total_time_ratio_vs_beam4"]),
         int(group.loc["b16_e0p0_d0_w10", "dual_pool"]),
         int(group.loc["b16_e0p0_d0_w10", "dual_blind"])),
        ("width 16, constant 0.05", float(group.loc["b16_e0p05_d0_w10", "total_time_ratio_vs_beam4"]),
         int(group.loc["b16_e0p05_d0_w10", "dual_pool"]),
         int(group.loc["b16_e0p05_d0_w10", "dual_blind"])),
        ("width 16, early 0.05", float(group.loc["b16_e0p05_d20_w10", "total_time_ratio_vs_beam4"]),
         int(group.loc["b16_e0p05_d20_w10", "dual_pool"]),
         int(group.loc["b16_e0p05_d20_w10", "dual_blind"])),
        ("width 16, early 0.2", float(group.loc["b16_e0p2_d20_w10", "total_time_ratio_vs_beam4"]),
         int(group.loc["b16_e0p2_d20_w10", "dual_pool"]),
         int(group.loc["b16_e0p2_d20_w10", "dual_blind"])),
        ("width 32, early 0.05", float(group.loc["b32_e0p05_d20_w10", "total_time_ratio_vs_beam4"]),
         int(group.loc["b32_e0p05_d20_w10", "dual_pool"]),
         int(group.loc["b32_e0p05_d20_w10", "dual_blind"])),
    ]
    labels, seconds, oracle, blind = zip(*rows)
    y = np.arange(len(rows))
    colors = ["#777777", "#777777", "#477da4", "#477da4", "#477da4",
              "#6c61a5", "#477da4"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y, seconds, color=colors, height=0.62)
    for index, (cost, pool_hits, blind_hits) in enumerate(zip(seconds, oracle, blind)):
        ax.text(cost + 0.15, index,
                f"{pool_hits}/7 oracle  ·  {blind_hits}/7 blind", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(seconds) + 4)
    ax.set(xlabel="H100 inference time / unpenalized width-4 beam",
           title="Seven-pair development pilot: cost and fold-mode coverage")
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
           title=f"Held-out fold-mode coverage (n={int(row.n)})")
    ax.legend()
    fig.tight_layout()
    save_plot_with_meta(fig, PLOTS / "heldout_coverage.png",
                        caption=f"Selected method {mode} was frozen after development scoring. "
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
