# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plots for the Protenix-vs-pyconfind contact eval.

Consumes ``contact_precision.csv`` (tidy long form, one row per
``stem x mode x predictor x range x k``) + ``contact_eval_meta.csv``.
All plots show **contacts @ L accuracy** (precision among the top-L
predicted pairs); the headline is aggregate + split by short/medium/long
range across the four configs, the rest stratify by the exp65 novelty /
MSA-depth axes.

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


def _mean_precision(df: pd.DataFrame, *, k: int) -> pd.DataFrame:
    """Mean precision by (mode, predictor, range) for a given top-L/k."""
    sub = df[df["k"] == k]
    return sub.groupby(["mode", "predictor", "range"])["precision"].mean().reset_index()


def plot_by_config_and_range(df: pd.DataFrame, out: Path, *, k: int, label: str, script_args: list[str]) -> None:
    """Headline: contacts@L per config, one panel per range."""
    means = _mean_precision(df, k=k)
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
    axes[0].set_ylabel(f"mean precision @ top-{label}")
    fig.suptitle(f"Contacts @ {label}: Protenix vs pyconfind ground truth (4 configs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot_with_meta(
        fig, out, script="plot.py", args=script_args,
        caption=(f"Mean precision among top-{label} ranked pairs vs pyconfind contacts "
                 f"(degree≥0.001, sep≥6), per config, aggregate and by range. "
                 f"distogram=rank by P(CB-CB≤8Å); structure=rank by pyconfind degree on the predicted CIF."),
    )
    plt.close(fig)


def _grouped_by_stratum(df, out, *, stratum, order, k, rng, label, script_args, title):
    """Grouped bars: x=stratum level, groups=config, y=mean precision@L (one range)."""
    sub = df[(df["k"] == k) & (df["range"] == rng) & df[stratum].isin(order)]
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
    ax.set_ylabel(f"mean precision @ top-{label}")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    save_plot_with_meta(fig, out, script="plot.py", args=script_args, caption=title)
    plt.close(fig)


def plot_precision_vs_neff(df, meta, out, *, k, rng, label, script_args):
    """Scatter precision@L(range) vs MSA Neff (log x), per predictor, colored by mode."""
    if "msa_neff" not in df.columns:
        return
    sub = df[(df["k"] == k) & (df["range"] == rng)].copy()
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
    axes[0].set_ylabel(f"precision @ top-{label} ({RANGE_TITLE[rng]})")
    fig.suptitle(f"Contacts @ {label} vs MSA depth — {RANGE_TITLE[rng]}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_plot_with_meta(fig, out, script="plot.py", args=script_args,
                        caption=f"Per-protein precision@top-{label} ({rng} range) vs MSA Neff.")
    plt.close(fig)


def main(precision_csv: Path, meta_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(precision_csv)
    meta = pd.read_csv(meta_csv) if Path(meta_csv).exists() else pd.DataFrame()
    sa = ["--precision-csv", str(precision_csv), "--meta-csv", str(meta_csv), "--out", str(out_dir)]

    # Headline: contacts @ L / L2 / L5 by config and range.
    for k, lab in [(1, "L"), (2, "L_2"), (5, "L_5")]:
        plot_by_config_and_range(df, out_dir / f"contacts_at_{lab}_by_config_and_range.png",
                                 k=k, label=lab, script_args=sa)

    # Stratified (exp65 axes). Only fire when the strata columns exist + populate.
    for stratum, order, fname, title in [
        ("neff_tier", NEFF_ORDER, "contacts_at_L_by_neff_tier", "Contacts @ L by MSA-depth tier (long-range)"),
        ("fold_verdict", FOLD_ORDER, "contacts_at_L_by_fold_verdict", "Contacts @ L by fold novelty (long-range)"),
    ]:
        if stratum in df.columns:
            _grouped_by_stratum(df, out_dir / f"{fname}.png", stratum=stratum, order=order,
                                k=1, rng="long", label="L", script_args=sa, title=title)

    # Precision vs MSA depth.
    plot_precision_vs_neff(df, meta, out_dir / "precision_vs_neff_long.png",
                           k=1, rng="long", label="L", script_args=sa)

    # foldbench vs exp65 (only when both present).
    if df["dataset"].nunique() > 1:
        df2 = df.copy()
        df2["group"] = np.where(df2["dataset"] == "foldbench100", "foldbench100", "exp65")
        _grouped_by_stratum(df2, out_dir / "contacts_at_L_foldbench_vs_exp65.png",
                            stratum="group", order=["foldbench100", "exp65"], k=1, rng="all",
                            label="L", script_args=sa, title="Contacts @ L: FoldBench-100 vs exp65 (aggregate)")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precision-csv", type=Path, required=True)
    ap.add_argument("--meta-csv", type=Path, default=Path("data/contact_eval_meta.csv"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args()
    main(args.precision_csv, args.meta_csv, args.out)
