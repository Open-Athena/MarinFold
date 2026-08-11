# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 4 — the four figures.

1. ``overlap_profile.png`` — how much of the eval set has a training homolog,
   by identity stratum and by training arm. The measurement.
2. ``rprecision_vs_identity.png`` — every predictor's R-precision across the
   identity ladder. The exp94 figure, redone with the ESM-Atlas arm folded in
   and the current frontier model. seq-KNN is the positive control: it *must*
   rise with identity.
3. ``headline_homology_free.png`` — the pre-registered cut: R-precision on the
   proteins with no detectable training homolog, next to the full 554.
4. ``designed_vs_natural.png`` — the same cut split on designed vs natural,
   because the low-identity end of this eval set is mostly de novo designs and
   pooling them hides that.

    uv run python plot_overlap.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402
from overlap_lib import (  # noqa: E402
    ARM_AFDB,
    ARM_ESM,
    ARM_LABELS,
    STRATUM_LABELS,
    STRATUM_NO_HIT,
    STRATUM_ORDER,
)
from stratify_and_compare import (  # noqa: E402
    KNN_LABEL,
    MARINFOLD,
    PREDICTOR_ORDER,
)

HERE = Path(__file__).resolve().parent

COLORS = {
    MARINFOLD: "#c0392b",
    "Protenix-v2 single-seq": "#2980b9",
    "ESMFold": "#7f8c8d",
    "ESMFold2": "#16a085",
    "Protenix-v2 + MSA": "#8e44ad",
    KNN_LABEL: "#d68910",
}


def _present(frame: pd.DataFrame, column: str = "predictor") -> list[str]:
    return [p for p in PREDICTOR_ORDER if p in set(frame[column])]


def plot_overlap_profile(identity: pd.DataFrame, out: Path, args: list[str]) -> None:
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    counts = (identity.groupby(["stratum", "designed"], observed=False).size()
              .unstack(fill_value=0).reindex(STRATUM_ORDER, fill_value=0))
    x = np.arange(len(STRATUM_ORDER))
    natural = counts.get(0, pd.Series(0, index=counts.index)).to_numpy()
    designed = counts.get(1, pd.Series(0, index=counts.index)).to_numpy()
    ax_left.bar(x, natural, color="#34495e", label="natural")
    ax_left.bar(x, designed, bottom=natural, color="#95a5a6", label="de novo designed")
    for xi, total in zip(x, natural + designed):
        ax_left.text(xi, total + 3, str(total), ha="center", fontsize=9)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels([STRATUM_LABELS[s] for s in STRATUM_ORDER], fontsize=9)
    ax_left.set_ylabel("eval proteins")
    ax_left.set_xlabel("best sequence identity to anything in the training set")
    ax_left.set_title(f"Training-set proximity of the {len(identity)}-protein eval set")
    ax_left.legend(frameon=False, fontsize=9)

    # Right: what each arm contributes. "Only X" = a significant hit in that arm
    # and none in the other, so the bars partition the proteins that have a hit.
    has_afdb = identity["afdb_n_hits_significant"] > 0
    has_esm = identity["esm_atlas_n_hits_significant"] > 0
    groups = {
        "both arms": int((has_afdb & has_esm).sum()),
        f"only {ARM_LABELS[ARM_AFDB].split(' (')[0]}": int((has_afdb & ~has_esm).sum()),
        f"only {ARM_LABELS[ARM_ESM].split(' (')[0]}": int((~has_afdb & has_esm).sum()),
        "neither": int((~has_afdb & ~has_esm).sum()),
    }
    colors = ["#2c3e50", "#3498db", "#1abc9c", "#e74c3c"]
    bars = ax_right.bar(list(groups), list(groups.values()), color=colors)
    for bar, value in zip(bars, groups.values()):
        ax_right.text(bar.get_x() + bar.get_width() / 2, value + 3,
                      f"{value}\n{value / len(identity):.0%}", ha="center", fontsize=9)
    ax_right.set_ylabel("eval proteins")
    ax_right.set_title("Which training arm supplies the homolog")
    ax_right.tick_params(axis="x", labelsize=9)
    ax_right.set_ylim(0, max(groups.values()) * 1.25)

    fig.tight_layout()
    save_plot_with_meta(
        fig, out,
        caption=("Left: eval proteins by best sequence identity to exp199's training "
                 "set (MMseqs2 -s 7.5; identity measured over alignments covering "
                 "≥50% of the query), split on de novo designed vs natural. Right: "
                 "which of the two training corpora contributes the homolog. "
                 "'neither' is the homology-free subset the re-eval uses."),
        script="plot_overlap.py", args=args,
    )
    plt.close(fig)


def plot_rprecision_vs_identity(metrics: pd.DataFrame, out: Path,
                                args: list[str]) -> None:
    data = metrics[(metrics["range"] == "all") & (metrics["cut"] == "R")
                   & (metrics["split"] == "all") & (metrics["stratum"] != "ALL")]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(STRATUM_ORDER))
    for predictor in _present(data):
        rows = (data[data["predictor"] == predictor]
                .set_index("stratum").reindex(STRATUM_ORDER))
        ax.errorbar(x, rows["mean"], yerr=rows["sem"], marker="o", capsize=3,
                    label=predictor, color=COLORS.get(predictor),
                    lw=2.2 if predictor == MARINFOLD else 1.4,
                    zorder=3 if predictor == MARINFOLD else 2)
    counts = (data[data["predictor"] == MARINFOLD]
              .set_index("stratum").reindex(STRATUM_ORDER)["n"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{STRATUM_LABELS[s]}\nn={int(counts[s])}" for s in STRATUM_ORDER],
                       fontsize=9)
    ax.set_xlabel("best sequence identity to exp199's training set")
    ax.set_ylabel("R-precision (all ranges)")
    ax.set_title("Does contact accuracy depend on training-set sequence proximity?")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot_with_meta(
        fig, out,
        caption=("R-precision (all ranges, ±SEM) across training-identity strata, "
                 "all six predictors on the same proteins. seq-KNN is the positive "
                 "control — a pure copy-the-neighbour model must track identity. A "
                 "flat MarinFold curve means its skill is not homology retrieval."),
        script="plot_overlap.py", args=args,
    )
    plt.close(fig)


def plot_headline(headline: pd.DataFrame, out: Path, args: list[str]) -> None:
    data = headline[(headline["range"] == "all") & (headline["cut"] == "R")]
    subsets = ["all_554", "no_homolog"]
    titles = {"all_554": "All 554 eval proteins",
              "no_homolog": "No detectable training homolog"}
    predictors = _present(data)
    fig, axes = plt.subplots(1, len(subsets), figsize=(12.5, 4.6), sharey=True)
    for ax, subset in zip(np.atleast_1d(axes), subsets):
        rows = (data[data["subset"] == subset].set_index("predictor")
                .reindex(predictors))
        x = np.arange(len(predictors))
        ax.bar(x, rows["mean"], yerr=rows["sem"], capsize=3,
               color=[COLORS.get(p, "#777") for p in predictors])
        for xi, (mean, value) in enumerate(zip(rows["mean"], rows["mean"])):
            if np.isfinite(value):
                ax.text(xi, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace(" (", "\n(") for p in predictors],
                           rotation=30, ha="right", fontsize=8.5)
        n = int(rows["n"].iloc[0]) if len(rows) else 0
        ax.set_title(f"{titles[subset]}  (n={n})")
        ax.grid(axis="y", alpha=0.25)
    np.atleast_1d(axes)[0].set_ylabel("R-precision (all ranges)")
    fig.tight_layout()
    save_plot_with_meta(
        fig, out,
        caption=("The pre-registered cut. Left: the published benchmark. Right: only "
                 "eval proteins with no MMseqs2 hit (E ≤ 1e-3, -s 7.5) against either "
                 "training corpus. Error bars are ±SEM over proteins; the paired "
                 "MarinFold-minus-baseline CIs are in data/headline.csv."),
        script="plot_overlap.py", args=args,
    )
    plt.close(fig)


def plot_designed_vs_natural(wide: pd.DataFrame, out: Path, args: list[str]) -> None:
    subset = wide[(wide["range"] == "all") & (wide["cut"] == "R")
                  & (wide["stratum"] == STRATUM_NO_HIT)]
    predictors = [p for p in PREDICTOR_ORDER if p in subset.columns]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.38
    x = np.arange(len(predictors))
    for offset, (flag, label, hatch) in enumerate(
        [(0, "natural", ""), (1, "de novo designed", "//")]
    ):
        group = subset[subset["designed"] == flag]
        means = [group[p].mean() for p in predictors]
        sems = [group[p].std(ddof=1) / np.sqrt(max(group[p].notna().sum(), 1))
                for p in predictors]
        ax.bar(x + (offset - 0.5) * width, means, width, yerr=sems, capsize=3,
               label=f"{label} (n={len(group)})", hatch=hatch,
               color=[COLORS.get(p, "#777") for p in predictors],
               edgecolor="white", alpha=1.0 if flag == 0 else 0.65)
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace(" (", "\n(") for p in predictors],
                       rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("R-precision (all ranges)")
    ax.set_title("Homology-free subset, split by designed vs natural")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot_with_meta(
        fig, out,
        caption=("The homology-free subset is not homogeneous: de novo designed "
                 "proteins have no homologs anywhere by construction, and structure "
                 "predictors find their idealised backbones easy. Splitting the "
                 "subset keeps that confound visible."),
        script="plot_overlap.py", args=args,
    )
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=HERE / "data")
    ap.add_argument("--out-dir", type=Path, default=HERE / "plots")
    args = ap.parse_args()
    argv = [f"--data-dir={args.data_dir}", f"--out-dir={args.out_dir}"]

    identity = pd.read_csv(args.data_dir / "eval_train_identity.csv")
    metrics = pd.read_csv(args.data_dir / "strata_metrics.csv")
    headline = pd.read_csv(args.data_dir / "headline.csv")
    wide = pd.read_csv(args.data_dir / "per_protein_wide.csv.gz")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    plot_overlap_profile(identity, args.out_dir / "overlap_profile.png", argv)
    plot_rprecision_vs_identity(metrics, args.out_dir / "rprecision_vs_identity.png", argv)
    plot_headline(headline, args.out_dir / "headline_homology_free.png", argv)
    plot_designed_vs_natural(wide, args.out_dir / "designed_vs_natural.png", argv)
    print(f"wrote 4 figures -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
