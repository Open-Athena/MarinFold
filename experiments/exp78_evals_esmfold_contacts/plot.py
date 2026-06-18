# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plots for the ESMFold / ESMFold2 vs Protenix-v2 contact eval (issue #78).

Consumes the combined ``contact_precision_all.csv`` (from
``combine_scores.py``): tidy long form, one row per
``(model, mode, predictor, dataset, stem, range, cut)``. The headline
comparison is the **four structure-config bars** the issue asks for:

    protenix-v2 · single_seq · structure
    protenix-v2 · msa        · structure
    esmfold     · single_seq · structure
    esmfold2    · single_seq · structure

All four rank candidate pairs by pyconfind contact degree on the
*predicted* structure, scored against the same pyconfind ground truth.
Cuts are L / L/2 / L/5 (CASP top-L/k precision) and **R** (R-precision:
cutoff = the bin's ground-truth contact count, ceiling 1.0 for every
protein — the density-robust cut, right for comparing short vs long).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

# The four structure configs the issue asks to compare, in display order.
# (model, mode, predictor, display label, color).
CONFIGS = [
    ("protenix-v2", "single_seq", "structure", "Protenix-v2 · SS", "#9ecae1"),
    ("protenix-v2", "msa", "structure", "Protenix-v2 · MSA", "#2171b5"),
    ("esmfold", "single_seq", "structure", "ESMFold", "#74c476"),
    ("esmfold2", "single_seq", "structure", "ESMFold2", "#238b45"),
]
RANGE_ORDER = ["all", "short", "medium", "long"]
RANGE_TITLE = {"all": "aggregate (sep≥6)", "short": "short [6,11]",
               "medium": "medium [12,23]", "long": "long [≥24]"}
NEFF_ORDER = ["orphan", "low", "marginal", "deep"]
FOLD_ORDER = ["novel_fold", "same_fold", "redundant"]
CUT_FILE = {"L": "L", "L/2": "L_2", "L/5": "L_5", "R": "R"}


def _axis_label(cut: str) -> str:
    return "R-precision" if cut == "R" else f"precision @ {cut}"


def _title(cut: str) -> str:
    return "R-precision" if cut == "R" else f"Contacts @ {cut}"


def _select(df: pd.DataFrame, model: str, mode: str, predictor: str) -> pd.DataFrame:
    return df[(df["model"] == model) & (df["mode"] == mode) & (df["predictor"] == predictor)]


def plot_by_config_and_range(df, out, *, cut, script_args):
    """Headline: mean precision@cut per model-config, one panel per range."""
    sub = df[df["cut"] == cut]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), sharey=True)
    for ax, rng in zip(axes, RANGE_ORDER):
        vals, colors, ticks = [], [], []
        for model, mode, predictor, disp, color in CONFIGS:
            rows = _select(sub[sub["range"] == rng], model, mode, predictor)
            vals.append(float(rows["precision"].mean()) if len(rows) else np.nan)
            colors.append(color)
            ticks.append(disp)
        x = np.arange(len(CONFIGS))
        ax.bar(x, vals, color=colors)
        for xi, v in zip(x, vals):
            if v == v:
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, rotation=30, ha="right", fontsize=8)
        ax.set_title(RANGE_TITLE[rng], fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"mean {_axis_label(cut)}")
    note = "  (ceiling 1.0 for every protein)" if cut == "R" else ""
    fig.suptitle(f"{_title(cut)}: ESMFold / ESMFold2 vs Protenix-v2 (structure configs)" + note, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot_with_meta(
        fig, out, script="plot.py", args=script_args,
        caption=(f"Mean {_axis_label(cut)} vs pyconfind contacts (degree≥0.001, sep≥6), per model, "
                 f"aggregate and by range. All configs rank candidate pairs by pyconfind contact "
                 f"degree on the predicted structure."
                 + (" R-precision cutoff = #true contacts, so a perfect ranker = 1.0." if cut == "R" else "")),
    )
    plt.close(fig)


def _grouped_by_stratum(df, out, *, stratum, order, cut, rng, script_args, title):
    """Grouped bars: x=stratum level, groups=model-config, y=mean precision@cut."""
    sub = df[(df["cut"] == cut) & (df["range"] == rng) & df[stratum].isin(order)]
    if sub.empty:
        return
    levels = [lv for lv in order if lv in set(sub[stratum])]
    counts = sub[(sub["model"] == "esmfold2")].groupby(stratum)["stem"].nunique()
    x = np.arange(len(levels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(levels) + 3), 4.4))
    for ci, (model, mode, predictor, disp, color) in enumerate(CONFIGS):
        vals = []
        for lv in levels:
            rows = _select(sub[sub[stratum] == lv], model, mode, predictor)
            vals.append(float(rows["precision"].mean()) if len(rows) else np.nan)
        ax.bar(x + (ci - 1.5) * w, vals, width=w, color=color, label=disp)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lv}\n(n={int(counts.get(lv, 0))})" for lv in levels], fontsize=9)
    ax.set_ylabel(f"mean {_axis_label(cut)}")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    save_plot_with_meta(fig, out, script="plot.py", args=script_args, caption=title)
    plt.close(fig)


def main(precision_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(precision_csv)
    if "model" not in df.columns:
        raise ValueError(f"{precision_csv} has no `model` column — run combine_scores.py first.")
    sa = ["--precision-csv", str(precision_csv), "--out", str(out_dir)]

    # Headline: precision @ L / L2 / L5 / R-precision, by model-config and range.
    for cut in ["L", "L/2", "L/5", "R"]:
        plot_by_config_and_range(df, out_dir / f"contacts_at_{CUT_FILE[cut]}_by_model_and_range.png",
                                 cut=cut, script_args=sa)

    # Stratified (exp65 axes) — long-range, at both L and R (R is the fair cut).
    for cut in ["L", "R"]:
        lab = CUT_FILE[cut]
        for stratum, order, fname in [
            ("neff_tier", NEFF_ORDER, f"contacts_at_{lab}_by_neff_tier"),
            ("fold_verdict", FOLD_ORDER, f"contacts_at_{lab}_by_fold_verdict"),
        ]:
            if stratum in df.columns:
                axis = "MSA-depth tier" if stratum == "neff_tier" else "fold novelty"
                _grouped_by_stratum(df, out_dir / f"{fname}.png", stratum=stratum, order=order,
                                    cut=cut, rng="long", script_args=sa,
                                    title=f"{_title(cut)} by {axis} (long-range)")

    # foldbench vs exp65 (R-precision, aggregate), when both present.
    if df["dataset"].nunique() > 1:
        df2 = df.copy()
        df2["group"] = np.where(df2["dataset"] == "foldbench100", "foldbench100", "exp65")
        _grouped_by_stratum(df2, out_dir / "contacts_at_R_foldbench_vs_exp65.png",
                            stratum="group", order=["foldbench100", "exp65"], cut="R", rng="all",
                            script_args=sa, title="R-precision: FoldBench-100 vs exp65 (aggregate)")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precision-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args()
    main(args.precision_csv, args.out)
