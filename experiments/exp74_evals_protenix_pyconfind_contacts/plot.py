# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plots for the Protenix-vs-pyconfind contact eval.

Consumes ``contact_precision.csv`` (tidy long form, one row per
``stem x mode x predictor x range x cut``) + ``contact_eval_meta.csv``.
Cuts are L / L/2 / L/5 (CASP top-L/k precision) and **R** (R-precision:
cutoff = the bin's ground-truth contact count, so the ceiling is 1.0 for
every protein — density-robust, the right cut for comparing short vs long).

The four configs are {single_seq, msa} x {distogram, structure}:
  - distogram: rank pairs by P(rep atoms within 8 Å) from the distogram.
  - structure: rank pairs by pyconfind contact degree on the predicted CIF.
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

# Config order + display + color. Hue = predictor, shade = mode.
CONFIGS = [
    ("single_seq", "distogram", "SS · distogram", "#9ecae1"),
    ("msa", "distogram", "MSA · distogram", "#2171b5"),
    ("single_seq", "structure", "SS · structure", "#fdae6b"),
    ("msa", "structure", "MSA · structure", "#d94801"),
]
RANGE_ORDER = ["all", "short", "medium", "long"]
RANGE_TITLE = {"all": "aggregate (sep≥6)", "short": "short [6,11]",
               "medium": "medium [12,23]", "long": "long [≥24]"}
NEFF_ORDER = ["orphan", "low", "marginal", "deep"]
FOLD_ORDER = ["novel_fold", "same_fold", "redundant"]
# cut id (as stored in the CSV) -> filename-safe label
CUT_FILE = {"L": "L", "L/2": "L_2", "L/5": "L_5", "R": "R"}


def _axis_label(cut: str) -> str:
    return "R-precision" if cut == "R" else f"precision @ {cut}"


def _title(cut: str) -> str:
    return "R-precision" if cut == "R" else f"Contacts @ {cut}"


def _mean_precision(df: pd.DataFrame, *, cut: str) -> pd.DataFrame:
    sub = df[df["cut"] == cut]
    return sub.groupby(["mode", "predictor", "range"])["precision"].mean().reset_index()


def plot_by_config_and_range(df, out, *, cut, script_args):
    """Headline: precision@cut per config, one panel per range."""
    means = _mean_precision(df, cut=cut)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, rng in zip(axes, RANGE_ORDER):
        vals, colors, ticks = [], [], []
        for mode, predictor, disp, color in CONFIGS:
            row = means[(means["mode"] == mode) & (means["predictor"] == predictor) & (means["range"] == rng)]
            vals.append(float(row["precision"].iloc[0]) if len(row) else np.nan)
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
    fig.suptitle(f"{_title(cut)}: Protenix vs pyconfind ground truth (4 configs)" + note, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot_with_meta(
        fig, out, script="plot.py", args=script_args,
        caption=(f"Mean {_axis_label(cut)} vs pyconfind contacts (degree≥0.001, sep≥6), per config, "
                 f"aggregate and by range. distogram=rank by P(CB-CB≤8Å); structure=rank by pyconfind "
                 f"degree on the predicted CIF."
                 + (" R-precision cutoff = #true contacts, so a perfect ranker = 1.0." if cut == "R" else "")),
    )
    plt.close(fig)


def _grouped_by_stratum(df, out, *, stratum, order, cut, rng, script_args, title):
    """Grouped bars: x=stratum level, groups=config, y=mean precision@cut (one range)."""
    sub = df[(df["cut"] == cut) & (df["range"] == rng) & df[stratum].isin(order)]
    if sub.empty:
        return
    means = sub.groupby([stratum, "mode", "predictor"])["precision"].mean().reset_index()
    counts = sub[sub["predictor"] == "structure"].groupby(stratum)["stem"].nunique()
    levels = [lv for lv in order if lv in set(means[stratum])]
    x = np.arange(len(levels))
    w = 0.2
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(levels) + 3), 4.4))
    for ci, (mode, predictor, disp, color) in enumerate(CONFIGS):
        vals = []
        for lv in levels:
            row = means[(means[stratum] == lv) & (means["mode"] == mode) & (means["predictor"] == predictor)]
            vals.append(float(row["precision"].iloc[0]) if len(row) else np.nan)
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


def plot_precision_vs_neff(df, out, *, cut, rng, script_args):
    """Scatter precision@cut(range) vs MSA Neff (log x), per predictor, colored by mode."""
    if "msa_neff" not in df.columns:
        return
    sub = df[(df["cut"] == cut) & (df["range"] == rng)].copy()
    sub["msa_neff"] = pd.to_numeric(sub["msa_neff"], errors="coerce")
    sub = sub[sub["msa_neff"] > 0]
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    for ax, predictor in zip(axes, ["distogram", "structure"]):
        for mode, color in [("single_seq", "#9ecae1"), ("msa", "#08519c")]:
            s = sub[(sub["predictor"] == predictor) & (sub["mode"] == mode)]
            ax.scatter(s["msa_neff"], s["precision"], s=14, alpha=0.5, color=color, label=mode)
        ax.set_xscale("log")
        ax.set_xlabel("MSA Neff (log)")
        ax.set_title(f"{predictor}", fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"{_axis_label(cut)} ({RANGE_TITLE[rng]})")
    fig.suptitle(f"{_title(cut)} vs MSA depth — {RANGE_TITLE[rng]}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot_with_meta(fig, out, script="plot.py", args=script_args,
                        caption=f"Per-protein {_axis_label(cut)} ({rng} range) vs MSA Neff.")
    plt.close(fig)


def main(precision_csv: Path, meta_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(precision_csv)
    sa = ["--precision-csv", str(precision_csv), "--meta-csv", str(meta_csv), "--out", str(out_dir)]

    # Headline: precision @ L / L2 / L5 / R-precision, by config and range.
    for cut in ["L", "L/2", "L/5", "R"]:
        plot_by_config_and_range(df, out_dir / f"contacts_at_{CUT_FILE[cut]}_by_config_and_range.png",
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

    # Precision vs MSA depth (R-precision, long-range).
    plot_precision_vs_neff(df, out_dir / "rprecision_vs_neff_long.png", cut="R", rng="long", script_args=sa)

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
    ap.add_argument("--meta-csv", type=Path, default=Path("data/contact_eval_meta.csv"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args()
    main(args.precision_csv, args.meta_csv, args.out)
