# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 4 — plot accuracy against MSA depth.

Three figures:

``rprecision_by_depth_tier.png``
    The requested table as a picture: mean all-range R-precision per depth tier,
    for all natural proteins and for the FoldBench / non-FoldBench halves, with
    bootstrap intervals and the bin size printed under each x tick. The
    intervals are the point — the shallow bins are small and the figure should
    say so without a caption.
``rprecision_vs_depth_scatter.png``
    Every natural protein, R-precision against depth, so the tier boundaries
    are visibly a choice rather than a finding.
``rprecision_by_neff_tier.png``
    The same cut on redundancy-weighted depth, which moves most proteins down a
    tier or two and fills the shallow bins.

    uv run python plot_depth.py
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd
import upstream as U
from build_summary import save_plot_with_meta

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

PREDICTOR_STYLE = {
    "MarinFold #232 m2-p06 (step 363k)": ("#1f77b4", "o", "-"),
    "Protenix-v2 + MSA": ("#d62728", "s", "-"),
    "Protenix-v2 single-seq": ("#ff7f0e", "^", "--"),
    "ESMFold2": ("#2ca02c", "D", "--"),
    "seq-KNN (decontaminated corpus)": ("#7f7f7f", "x", ":"),
}
POPULATIONS = (
    ("all_natural", "All natural (372)"),
    ("foldbench_natural", "FoldBench natural (314)"),
    ("nonfoldbench_natural", "CAMEO-hard + CASP-FM (58)"),
)
TIER_ORDER = [name for name, _, _ in U.DEPTH_TIERS]


def tier_figure(tiers: pd.DataFrame, *, axis: str, title: str):
    """One panel per population: mean R-precision against depth tier."""

    selected = tiers[
        (tiers.tier_axis == axis)
        & (tiers["range"] == "all")
        & (tiers["cut"] == "R")
        & (tiers.tier != "all")
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axis_handle, (population, label) in zip(axes, POPULATIONS, strict=True):
        panel = selected[selected.population == population]
        for predictor, (color, marker, line) in PREDICTOR_STYLE.items():
            rows = panel[panel.predictor == predictor].set_index("tier")
            rows = rows.reindex([t for t in TIER_ORDER if t in rows.index])
            if rows.empty:
                continue
            x = [TIER_ORDER.index(tier) for tier in rows.index]
            axis_handle.errorbar(
                x,
                rows["mean"],
                yerr=[
                    rows["mean"] - rows.ci_low,
                    rows.ci_high - rows["mean"],
                ],
                color=color,
                marker=marker,
                linestyle=line,
                capsize=3,
                label=predictor,
            )
        counts = (
            panel[panel.predictor.isin(PREDICTOR_STYLE)]
            .groupby("tier")["n"]
            .max()
            .reindex(TIER_ORDER)
        )
        axis_handle.set_xticks(range(len(TIER_ORDER)))
        axis_handle.set_xticklabels(
            [
                f"{tier}\nn={int(count) if pd.notna(count) else 0}"
                for tier, count in zip(TIER_ORDER, counts, strict=True)
            ]
        )
        axis_handle.set_title(label)
        axis_handle.grid(alpha=0.3)
        axis_handle.set_xlabel(
            "ColabFold MSA depth" if axis == "depth_tier" else "Neff (80 % identity)"
        )
    axes[0].set_ylabel("R-precision (all ranges)")
    axes[0].set_ylim(0, 1)
    figure.suptitle(title)
    # One legend under the panels: in-axes placement covered the sparse
    # bottom-right of the non-FoldBench panel, which is where its
    # single-sequence baseline sits.
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
        frameon=False, bbox_to_anchor=(0.5, -0.02),
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return figure


def scatter_figure(frame: pd.DataFrame):
    """Per-protein R-precision against depth, plus the depth distribution."""

    natural = frame[
        (frame.subset != "foldbench_designed")
        & (frame["range"] == "all")
        & (frame["cut"] == "R")
    ]
    figure, axes = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True, height_ratios=[3, 1]
    )
    jitter = np.random.default_rng(0)
    for predictor in ("MarinFold #232 m2-p06 (step 363k)", "Protenix-v2 + MSA"):
        rows = natural[natural.predictor == predictor]
        color, marker, _ = PREDICTOR_STYLE[predictor]
        x = rows.msa_depth.to_numpy(dtype=float)
        x = np.clip(x, 1, None) * np.exp(jitter.normal(0, 0.02, len(x)))
        axes[0].scatter(
            x, rows.precision, s=14, alpha=0.45, color=color, marker=marker,
            label=predictor, edgecolors="none",
        )
        edges = np.array([1, 10, 100, 1000, 10_000, 100_000], dtype=float)
        centers, means = [], []
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            window = rows[(rows.msa_depth >= low) & (rows.msa_depth < high)]
            if len(window) >= 3:
                centers.append(np.sqrt(low * high))
                means.append(window.precision.mean())
        axes[0].plot(centers, means, color=color, linewidth=2.5)
    axes[0].set_xscale("log")
    axes[0].set_ylabel("R-precision (all ranges)")
    axes[0].set_ylim(0, 1)
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_title(
        "Every natural eval protein: contact accuracy against the MSA depth an "
        "MSA-based method would have had"
    )

    proteins = natural.drop_duplicates(["dataset", "stem"])
    bins = np.logspace(0, 5, 26)
    for subset, label, color in (
        ("foldbench_natural", "FoldBench natural", "#4c72b0"),
        ("nonfoldbench_natural", "CAMEO-hard + CASP-FM", "#dd8452"),
    ):
        axes[1].hist(
            proteins[proteins.subset == subset].msa_depth.clip(lower=1),
            bins=bins, alpha=0.6, label=label, color=color,
        )
    for _, low, _ in U.DEPTH_TIERS[1:]:
        for axis_handle in axes:
            axis_handle.axvline(low, color="k", linewidth=0.8, alpha=0.35)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("ColabFold MSA depth (sequences, log scale)")
    axes[1].set_ylabel("proteins")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(U.PLOTS))
    args = parser.parse_args()
    U.PLOTS.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(U.DATA / "per_protein_depth.csv")
    tiers = pd.read_csv(U.DATA / "depth_tiers.csv")

    save_plot_with_meta(
        tier_figure(
            tiers,
            axis="depth_tier",
            title="Contact accuracy by ColabFold MSA depth",
        ),
        f"{args.out}/rprecision_by_depth_tier.png",
        caption=(
            "Mean all-range R-precision per MSA-depth tier, with 95 % bootstrap "
            "intervals over proteins and bin sizes on the x axis."
        ),
        dpi=160,
    )
    save_plot_with_meta(
        tier_figure(
            tiers,
            axis="neff_tier",
            title="Contact accuracy by effective MSA depth (Neff, 80 % identity)",
        ),
        f"{args.out}/rprecision_by_neff_tier.png",
        caption=(
            "The same cut on redundancy-weighted depth, which fills the shallow "
            "bins raw sequence count leaves nearly empty."
        ),
        dpi=160,
    )
    save_plot_with_meta(
        scatter_figure(frame),
        f"{args.out}/rprecision_vs_depth_scatter.png",
        caption=(
            "Every natural eval protein, accuracy against depth, with per-decade "
            "means; the histogram shows where each subset sits."
        ),
        dpi=160,
    )
    plt.close("all")
    print(f"wrote 3 figures to {args.out}")


if __name__ == "__main__":
    main()
