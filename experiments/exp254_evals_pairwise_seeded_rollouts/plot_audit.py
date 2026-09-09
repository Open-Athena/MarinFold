# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot paired effects from saved audit CSVs without rerunning inference."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta


def unique_row(frame: pd.DataFrame, **filters: str | int) -> pd.Series:
    """Select exactly one reported contrast, failing on missing/duplicate data."""
    selection = frame
    for column, value in filters.items():
        selection = selection[selection[column] == value]
    if len(selection) != 1:
        raise ValueError(f"expected one row for {filters}, found {len(selection)}")
    return selection.iloc[0]


def panel(ax: plt.Axes, title: str, rows: list[tuple[str, float, float, float]],
          color: str, practical_band: bool = False) -> None:
    """Draw effects in percentage points on a common horizontal scale."""
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=13)
    if practical_band:
        ax.axvspan(-0.5, 0.5, color="#64748b", alpha=0.10, zorder=0)
    ax.axvline(0, color="#64748b", linewidth=1, zorder=1)
    positions = np.arange(len(rows))[::-1]
    for y, (label, mean, lower, upper) in zip(positions, rows):
        mean, lower, upper = np.array([mean, lower, upper]) * 100
        ax.errorbar(mean, y, xerr=[[mean - lower], [upper - mean]], fmt="o",
                    color=color, markersize=7, capsize=4, elinewidth=2, zorder=3)
        ax.text(1.035, y, f"{mean:+.2f}  [{lower:+.2f}, {upper:+.2f}]",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=10)
    ax.set_yticks(positions, [row[0] for row in rows], fontsize=10)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_ylim(-0.55, len(rows) - 0.45)
    ax.set_xlim(-6.2, 4.3)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> int:
    """Render the scientific audit figure and its regeneration metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("plots"))
    args = parser.parse_args()
    consensus = pd.read_csv(args.data / "exp254_paired_deltas.csv")
    oracle = pd.read_csv(args.data / "exp254_audit_oracle_summary.csv")
    cluster = pd.read_csv(args.data / "exp254_audit_cluster_summary.csv")
    consensus_rows = []
    for arm, label in (("top-100", "Top-100 seeds vs i.i.d."),
                       ("long-range", "Long-range seeds vs i.i.d."),
                       ("1/3 per range", "Equal-thirds seeds vs i.i.d.")):
        row = unique_row(consensus, range="all", a=f"seeded {arm} consensus",
                         b="i.i.d. consensus")
        consensus_rows.append((label, row.mean_delta, row.lo, row.hi))
    seed = unique_row(oracle, range="all", mode="continuation", metric="oracle_fixed_R",
                      arm="seeded", reference="iid")
    clustered = unique_row(cluster, K=10, comparison="cluster_oracle_vs_pooled")
    oracle_rows = [
        ("Top-100 vs i.i.d. single-rollout oracle\n(seed removed; fixed R)",
         seed["mean"], seed.ci_low, seed.ci_high),
        ("Best of 10 cluster maps vs pooled", clustered["mean"],
         clustered.ci_low, clustered.ci_high),
    ]
    selector_rows = []
    for comparison, label in (
        ("geometric_vs_pooled", "Most consistent of 5 vs pooled"),
        ("geometric_vs_blind", "Most consistent of 5 vs blind pick"),
    ):
        row = unique_row(cluster, K=5, comparison=comparison)
        selector_rows.append((label, row["mean"], row.ci_low, row.ci_high))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(3, 1, figsize=(13.6, 8.2), sharex=True,
                            gridspec_kw={"height_ratios": [3, 2, 2]})
    fig.subplots_adjust(left=0.31, right=0.77, bottom=0.16, top=0.83, hspace=0.63)
    panel(axes[0], "Consensus decoding", consensus_rows, "#0369a1", practical_band=True)
    panel(axes[1], "Oracle diagnostics — selection requires ground truth", oracle_rows, "#7c3aed")
    panel(axes[2], "Geometric selector — saved sets cut at ground-truth R", selector_rows, "#b45309")
    axes[-1].set_xticks([-6, -4, -2, 0, 2, 4])
    axes[-1].set_xlabel("Paired precision difference (percentage points)", labelpad=9, fontsize=11)
    fig.text(0.04, 0.955, "Seeding and selection on eval-val: paired effects", fontsize=19,
             fontweight="bold", ha="left")
    fig.text(0.04, 0.912, "97 proteins · all-range contacts · existing samples only", fontsize=12,
             color="#475569")
    fig.text(0.786, 0.86, "Effect [95% CI], pp", fontsize=10, fontweight="bold")
    fig.text(0.04, 0.065,
             "Intervals resample proteins; they do not measure fresh-run variability. "
             "Exploratory contrasts are not adjusted for multiple comparisons.",
             fontsize=9, color="#475569")
    fig.text(0.04, 0.037,
             "Gray band in consensus panel: prespecified ±0.5 pp practical threshold. "
             "Positive values favor the first method; each row names its reference.",
             fontsize=9, color="#475569")
    args.out.mkdir(parents=True, exist_ok=True)
    save_plot_with_meta(
        fig, args.out / "conclusions_audit.png", dpi=180,
        caption="Paired effects on 97 eval-val proteins. Consensus, fixed-R continuation "
                "oracle, cluster oracle, and geometric selection use explicitly labeled "
                "references. Error bars are protein-bootstrap 95% pointwise intervals; "
                "oracle results require ground truth and are not deployable improvements.",
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
