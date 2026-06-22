# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp89 plots — the contacts-v1 1.5B model (#61/#75 best, eval loss 2.7566)
against every other predictor on the shared eval set.

Reads the unified ``contact_precision_all.csv`` (from ``compute_metrics.py``):
``(model, mode, predictor, dataset, stem, range, cut, precision)``. The
headline is MarinFold next to Protenix-v2 (single_seq / msa · structure),
ESMFold and ESMFold2, for every metric the issue asks for — contacts @
{L, L/2, L/5}, R-precision, and **AUC** — aggregate and split by
short / medium / long range.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Display order. MarinFold (the experiment's subject) first, then the
# structure-config baselines. (model, mode, predictor, label, color).
CONFIGS = [
    ("marinfold-contacts-v1", "single_seq", "lm", "MarinFold-cv1 1.5B (seq)", "#e6550d"),
    ("protenix-v2", "single_seq", "structure", "Protenix-v2 · SS", "#9ecae1"),
    ("protenix-v2", "msa", "structure", "Protenix-v2 · MSA", "#2171b5"),
    ("esmfold", "single_seq", "structure", "ESMFold", "#74c476"),
    ("esmfold2", "single_seq", "structure", "ESMFold2", "#238b45"),
]
CUTS = ["L", "L/2", "L/5", "R", "AUC"]
CUT_FILE = {"L": "L", "L/2": "L_2", "L/5": "L_5", "R": "R", "AUC": "AUC"}
RANGE_ORDER = ["all", "short", "medium", "long"]
# exp65 strata labels (low MSA-depth → high; most novel fold → least).
NEFF_ORDER = ["orphan", "marginal", "low", "deep"]
FOLD_ORDER = ["novel_fold", "same_fold", "redundant"]


def _axis_label(cut: str) -> str:
    return {"R": "R-precision", "AUC": "ranking AUC"}.get(cut, f"precision @ {cut}")


def _title(cut: str) -> str:
    return {"R": "R-precision", "AUC": "Ranking AUC"}.get(cut, f"Contacts @ {cut}")


def _save(fig, out: Path, *, script_args, caption: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps({"script": "plot.py", "args": script_args, "caption": caption}, indent=2))
    plt.close(fig)


def _select(df, model, mode, predictor):
    return df[(df["model"] == model) & (df["mode"] == mode) & (df["predictor"] == predictor)]


def plot_by_config_and_range(df, out, *, cut, script_args):
    sub = df[df["cut"] == cut]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), sharey=True)
    for ax, rng in zip(axes, RANGE_ORDER):
        vals = []
        for model, mode, predictor, disp, color in CONFIGS:
            rows = _select(sub[sub["range"] == rng], model, mode, predictor)
            vals.append(float(rows["precision"].mean()) if len(rows) else np.nan)
        x = np.arange(len(CONFIGS))
        ax.bar(x, vals, color=[c[4] for c in CONFIGS])
        for xi, v in zip(x, vals):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
        if cut == "AUC":
            ax.axhline(0.5, ls="--", lw=1, color="grey")
        ax.set_title(rng, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([c[3] for c in CONFIGS], rotation=30, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"mean {_axis_label(cut)}")
    axes[0].set_ylim(0.4 if cut == "AUC" else 0, 1.0)
    fig.suptitle(f"{_title(cut)}: MarinFold-cv1 vs Protenix-v2 / ESMFold / ESMFold2", fontsize=13)
    fig.tight_layout()
    _save(fig, out, script_args=script_args,
          caption=(f"Mean {_axis_label(cut)} vs pyconfind contacts (degree>=0.001, sep>=6), per "
                   f"predictor, aggregate and by range. MarinFold ranks pairs by its pairwise "
                   f"contact-statement log-prob; the structure models rank by pyconfind degree on "
                   f"the predicted structure. n=554 proteins."))


def _grouped_by_stratum(df, out, *, stratum, order, cut, rng, script_args, title):
    sub = df[(df["cut"] == cut) & (df["range"] == rng) & df[stratum].isin(order)]
    if sub.empty:
        return
    counts = sub.drop_duplicates(["stem", stratum]).groupby(stratum).size()
    levels = [lv for lv in order if lv in set(sub[stratum])]
    x = np.arange(len(levels))
    w = 0.8 / len(CONFIGS)
    fig, ax = plt.subplots(figsize=(max(7, 1.7 * len(levels) + 3), 4.6))
    for ci, (model, mode, predictor, disp, color) in enumerate(CONFIGS):
        vals = []
        for lv in levels:
            rows = _select(sub[sub[stratum] == lv], model, mode, predictor)
            vals.append(float(rows["precision"].mean()) if len(rows) else np.nan)
        ax.bar(x + (ci - (len(CONFIGS) - 1) / 2) * w, vals, width=w, color=color, label=disp)
    if cut == "AUC":
        ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lv}\n(n={int(counts.get(lv, 0))})" for lv in levels], fontsize=9)
    ax.set_ylabel(f"mean {_axis_label(cut)}")
    ax.set_ylim(0.4 if cut == "AUC" else 0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    _save(fig, out, script_args=script_args, caption=title)


def main(precision_csv: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(precision_csv)
    sa = ["--precision-csv", str(precision_csv), "--out", str(out_dir)]

    for cut in CUTS:
        plot_by_config_and_range(df, out_dir / f"contacts_at_{CUT_FILE[cut]}_by_config_and_range.png",
                                 cut=cut, script_args=sa)

    for cut in ["L", "R", "AUC"]:
        lab = CUT_FILE[cut]
        for stratum, order, axis in [("neff_tier", NEFF_ORDER, "MSA-depth tier"),
                                     ("fold_verdict", FOLD_ORDER, "fold novelty")]:
            if stratum in df.columns:
                _grouped_by_stratum(df, out_dir / f"contacts_at_{lab}_by_{stratum}.png",
                                    stratum=stratum, order=order, cut=cut, rng="long",
                                    script_args=sa, title=f"{_title(cut)} by {axis} (long-range)")

    if df["dataset"].nunique() > 1:
        df2 = df.copy()
        df2["group"] = np.where(df2["dataset"] == "foldbench100", "foldbench100", "exp65")
        for cut in ["R", "AUC"]:
            _grouped_by_stratum(df2, out_dir / f"contacts_at_{CUT_FILE[cut]}_foldbench_vs_exp65.png",
                                stratum="group", order=["foldbench100", "exp65"], cut=cut, rng="all",
                                script_args=sa, title=f"{_title(cut)}: FoldBench-100 vs exp65 (aggregate)")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precision-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args()
    main(args.precision_csv, args.out)
