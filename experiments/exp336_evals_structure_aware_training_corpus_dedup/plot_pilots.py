# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot sequence-dedup results, the overlap pilot, and structure calibration."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"


def plot_sequence() -> None:
    data = pd.read_csv(DATA / "cross_source_sequence_pilot.csv").sort_values(
        "min_sequence_identity"
    )
    x = data.min_sequence_identity * 100
    y = data.afdb_query_hit_fraction * 100
    low = data.afdb_query_hit_fraction_ci95_low * 100
    high = data.afdb_query_hit_fraction_ci95_high * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, y, marker="o", color="#2b6cb0", linewidth=2)
    ax.fill_between(x, low, high, color="#2b6cb0", alpha=0.18)
    ax.axvline(50, color="#c53030", linestyle="--", linewidth=1)
    point = data.loc[data.min_sequence_identity == 0.5].iloc[0]
    ax.annotate(
        f"50%: {100 * point.afdb_query_hit_fraction:.1f}%",
        (50, 100 * point.afdb_query_hit_fraction),
        xytext=(57, 27),
        arrowprops={"arrowstyle": "->", "color": "#555555"},
    )
    ax.set(
        xlabel="Minimum aligned sequence identity (%)",
        ylabel="Sampled AFDB queries with an ESM hit (%)",
        title="Cross-source overlap in a random 10k-AFDB pilot",
    )
    ax.text(
        0.01,
        0.02,
        ">=80% coverage of both chains; lower bound because the MMseqs prefilter saturated",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "cross_source_sequence_pilot.png",
        caption=(
            "Fraction of 10,000 randomly sampled AFDB rows with at least one current "
            "ESM-Atlas match. Counts are lower bounds because the 1,000-candidate "
            "prefilter saturated."
        ),
    )
    plt.close(fig)


def plot_structure() -> None:
    data = pd.read_csv(DATA / "structure_threshold_calibration.csv")
    data = data.loc[
        (data.min_sequence_identity == 0.5)
        & data.scope.isin(["current_anchor_anchor", "prospective_anchor_candidate"])
    ].copy()
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    styles = {
        ("afdb", "current_anchor_anchor"): ("AFDB current/current", "#2b6cb0", "o"),
        ("afdb", "prospective_anchor_candidate"): ("AFDB current/#292", "#dd6b20", "s"),
        ("esm_atlas", "prospective_anchor_candidate"): ("ESM current/#292", "#2f855a", "^"),
    }
    for key, (label, color, marker) in styles.items():
        subset = data.loc[(data.source == key[0]) & (data.scope == key[1])]
        if subset.sequence_candidate_pairs.max() == 0:
            continue
        ax.plot(
            subset.min_bidirectional_tm_for_redundancy,
            subset.rescue_fraction * 100,
            label=label,
            color=color,
            marker=marker,
            linewidth=2,
        )
    ax.axvline(0.7, color="#c53030", linestyle="--", linewidth=1)
    ax.set(
        xlabel="TM-score required in both directions",
        ylabel="Sequence-candidate pairs kept as structural modes (%)",
        title=">=50%-identity pairs are highly sensitive to the TM threshold",
        xticks=[0.5, 0.7, 0.8, 0.9],
    )
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.text(
        0.01,
        0.02,
        "Curated #292 galleries: threshold calibration, not a prevalence estimate",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "structure_threshold_calibration.png",
        caption=(
            "Share of >=50%-identity, >=80%-bidirectional-coverage measured pairs whose "
            "minimum directional TM-score falls below the active threshold."
        ),
    )
    plt.close(fig)


def plot_direct_sequence_thresholds() -> None:
    data = pd.read_csv(DATA / "direct_sequence_thresholds.csv").sort_values(
        "min_sequence_identity"
    )
    x = data.min_sequence_identity * 100
    y = data.removal_fraction * 100
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(x, y, marker="o", color="#2f855a", linewidth=2)
    ax.fill_between(x, 0, y, color="#2f855a", alpha=0.14)
    for _, row in data.iterrows():
        ax.annotate(
            f"{100 * row.removal_fraction:.2f}%",
            (100 * row.min_sequence_identity, 100 * row.removal_fraction),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.axvline(50, color="#c53030", linestyle="--", linewidth=1)
    ax.set(
        xlabel="Minimum aligned sequence identity (%)",
        ylabel="Training documents removed (%)",
        title="Direct-witness sequence-only deduplication ceiling",
        xticks=[50, 70, 90, 95, 100],
        ylim=(0, max(y) * 1.14),
    )
    ax.grid(alpha=0.2)
    ax.text(
        0.01,
        0.95,
        ">=80% coverage of both chains; structure rule not yet applied",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
        va="top",
    )
    fig.tight_layout()
    save_plot_with_meta(
        fig,
        PLOTS / "direct_sequence_thresholds.png",
        caption=(
            "Documents removed by a deterministic direct-witness selector within frozen "
            "50%-identity Linclust candidate neighborhoods. This is a sequence-only ceiling "
            "for joint sequence/structure removal, not the final structure-aware result."
        ),
    )
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(exist_ok=True)
    plot_direct_sequence_thresholds()
    plot_sequence()
    plot_structure()


if __name__ == "__main__":
    main()
