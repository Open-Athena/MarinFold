"""Plot curation outcomes from saved measurements, without rerunning alignment."""

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from build_summary import save_plot_with_meta
from structure_audit import read_csv


def main() -> None:
    """Write selection and confidence diagnostics with reproducible plot metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.sample_dir / "candidates.csv")
    tm = np.array([float(r["nearest_anchor_tm"]) for r in rows])
    core = np.array(
        [
            float(r["max_anchor_core_tm"]) if r["max_anchor_core_tm"] else np.nan
            for r in rows
        ]
    )
    chosen = np.array([int(r["selected_order"]) > 0 for r in rows])
    comparable = np.array(
        [
            float(r["min_coverage"]) >= 0.8 and float(r["min_length_ratio"]) >= 0.8
            for r in rows
        ]
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for mask, color, label in [
        (~comparable, "#9da9ae", "Coverage / length failure"),
        (comparable & ~chosen, "#238794", "Comparable, not selected"),
        (chosen, "#da852d", "Provisional addition"),
    ]:
        axes[0].hist(
            tm[mask], bins=np.linspace(0.2, 1, 25), color=color, alpha=0.7, label=label
        )
        axes[1].scatter(
            tm[mask],
            core[mask],
            s=20 if not mask is chosen else 35,
            color=color,
            alpha=0.75,
        )
    axes[0].axvline(0.8, color="#252525", linestyle="--", linewidth=1)
    axes[0].set(
        xlabel="TM to closest retained training anchor", ylabel="Omitted members"
    )
    axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot([0.2, 1], [0.2, 1], color="#999999", linewidth=1)
    axes[1].axhline(0.8, color="#333333", linestyle="--", linewidth=1)
    axes[1].set(
        xlabel="Max whole-chain TM vs all anchors",
        ylabel="Max masked-core TM vs all anchors",
        xlim=(0.2, 1.01),
        ylim=(0.2, 1.01),
    )
    counts = Counter(r["selection_reason"].split(";")[0] for r in rows)
    keys = [
        "coverage_or_length",
        "covered_by_training_anchor",
        "redundant_with_addition_or_cluster_cap",
        "provisional_diverse",
    ]
    axes[2].barh(
        ["Coverage / length", "Covered by training", "Redundant / cap", "Provisional"],
        [counts[k] for k in keys],
        color=["#9da9ae", "#238794", "#547680", "#da852d"],
    )
    axes[2].set(xlabel="Omitted members")
    axes[2].invert_yaxis()
    fig.suptitle(
        f"{args.source}: developmental curation sample ({len(rows)} omitted members)",
        fontsize=15,
    )
    save_plot_with_meta(
        fig,
        args.output / f"{args.source.lower()}_curation.png",
        caption="Small stratified development sample, not a population yield estimate. Provisional means whole-chain TM ≤0.8 versus all retained anchors, length ratio and both coverages ≥0.8, and greedy per-cluster selection. Core TM is a diagnostic with independently masked chains; it is not a training-clearance criterion. Candidate-level evaluation exclusion remains outstanding.",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
