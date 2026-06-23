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

The headline accuracy panels are **boxplots** over the 554 proteins (box =
median + IQR, whiskers = 1.5×IQR, points = outliers, ◆ = mean). The
stratified-by-stratum plots stay grouped bars with 95% bootstrap-CI error bars.
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
import seaborn as sns

# Display order. MarinFold (the experiment's subject) first, then the
# structure-config baselines. (model, mode, predictor, label, color).
CONFIGS = [
    ("marinfold-cv1-ens10", "single_seq", "lm", "MarinFold #61 ×10 ens", "#a63603"),
    ("marinfold-contacts-v1", "single_seq", "lm", "MarinFold #61 (2.76)", "#e6550d"),
    ("marinfold-cv1-exp67", "single_seq", "lm", "MarinFold #67 (2.98)", "#fdae6b"),
    ("protenix-v2", "single_seq", "structure", "Protenix-v2 · SS", "#9ecae1"),
    ("protenix-v2", "msa", "structure", "Protenix-v2 · MSA", "#2171b5"),
    ("esmfold", "single_seq", "structure", "ESMFold", "#74c476"),
    ("esmfold2", "single_seq", "structure", "ESMFold2", "#238b45"),
]
CUTS = ["L", "L/2", "L/5", "R", "AUC"]
CUT_FILE = {"L": "L", "L/2": "L_2", "L/5": "L_5", "R": "R", "AUC": "AUC"}
RANGE_ORDER = ["all", "short", "medium", "long"]
NEFF_ORDER = ["orphan", "marginal", "low", "deep"]
FOLD_ORDER = ["novel_fold", "same_fold", "redundant"]

N_BOOT = 2000
BOXNOTE = ("box = median & IQR (Q1–Q3)   ·   whiskers = 1.5×IQR   ·   ◆ = mean   ·   "
           "points = outliers   ·   n=554 proteins")


def _axis_label(cut: str) -> str:
    return {"R": "R-precision", "AUC": "ranking AUC"}.get(cut, f"precision @ {cut}")


def _title(cut: str) -> str:
    return {"R": "R-precision", "AUC": "Ranking AUC"}.get(cut, f"Contacts @ {cut}")


def boot_ci(vals: np.ndarray, *, n_boot: int = N_BOOT, ci: float = 95.0, seed: int = 0):
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - ci) / 2, 100 - (100 - ci) / 2])
    return float(vals.mean()), float(lo), float(hi)


def _save(fig, out: Path, *, script_args, caption: str) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps({"script": "plot.py", "args": script_args, "caption": caption}, indent=2))
    plt.close(fig)


def _vals(df, model, mode, predictor) -> np.ndarray:
    s = df[(df["model"] == model) & (df["mode"] == mode) & (df["predictor"] == predictor)]
    return s["precision"].to_numpy(dtype=float)


def plot_by_config_and_range(df, out, *, cut, script_args, configs=None, title=None):
    configs = configs or CONFIGS
    sub = df[df["cut"] == cut]
    labels = [c[3] for c in configs]
    palette = {c[3]: c[4] for c in configs}
    meanprops = dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=4.5)
    flierprops = dict(marker=".", markersize=2, markerfacecolor="0.4", markeredgecolor="none", alpha=0.35)
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), sharey=True)
    for ax, rng in zip(axes, RANGE_ORDER):
        rsub = sub[sub["range"] == rng]
        rows, means = [], {}
        for model, mode, pred, disp, _ in configs:
            v = _vals(rsub, model, mode, pred)
            v = v[np.isfinite(v)]
            rows += [(disp, x) for x in v]
            means[disp] = float(v.mean()) if v.size else float("nan")
        bdf = pd.DataFrame(rows, columns=["cfg", "precision"])
        sns.boxplot(data=bdf, x="cfg", y="precision", order=labels, hue="cfg", palette=palette,
                    ax=ax, width=0.62, legend=False, showmeans=True, meanprops=meanprops,
                    flierprops=flierprops, medianprops=dict(color="black", linewidth=1.3),
                    boxprops=dict(alpha=0.85), linewidth=1.0)
        for xi, disp in enumerate(labels):
            if not np.isnan(means[disp]):
                ax.text(xi, 1.03, f"{means[disp]:.2f}", ha="center", va="bottom", fontsize=7)
        if cut == "AUC":
            ax.axhline(0.5, ls="--", lw=1, color="grey", zorder=0)
        ax.set_title(rng, fontsize=10)
        for t in ax.get_xticklabels():
            t.set_rotation(30); t.set_horizontalalignment("right"); t.set_fontsize(8)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(_axis_label(cut))
    axes[0].set_ylim(-0.02, 1.08)
    fig.suptitle(title or f"{_title(cut)}: MarinFold-cv1 vs Protenix-v2 / ESMFold / ESMFold2  (n=554)",
                 fontsize=13)
    fig.text(0.5, 0.005, BOXNOTE, ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out, script_args=script_args,
          caption=(f"{_axis_label(cut)} vs pyconfind contacts (degree>=0.001, sep>=6), per "
                   f"predictor, aggregate and by range, over 554 proteins. Boxplots: box = median "
                   f"& IQR, whiskers = 1.5×IQR, points = outliers, white diamond = mean (labelled). "
                   f"MarinFold ranks pairs by its pairwise contact-statement log-prob; the "
                   f"structure models rank by pyconfind degree on the predicted structure."))


def _grouped_by_stratum(df, out, *, stratum, order, cut, rng, script_args, title):
    sub = df[(df["cut"] == cut) & (df["range"] == rng) & df[stratum].isin(order)]
    if sub.empty:
        return
    counts = sub.drop_duplicates(["stem", stratum]).groupby(stratum).size()
    levels = [lv for lv in order if lv in set(sub[stratum])]
    x = np.arange(len(levels))
    w = 0.8 / len(CONFIGS)
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * len(levels) + 3), 4.8))
    for ci, (model, mode, predictor, disp, color) in enumerate(CONFIGS):
        means, los, his = [], [], []
        for lv in levels:
            v = _vals(sub[sub[stratum] == lv], model, mode, predictor)
            m, lo, hi = boot_ci(v, seed=ci)
            means.append(m); los.append(lo); his.append(hi)
        off = (ci - (len(CONFIGS) - 1) / 2) * w
        yerr = np.array([[m - lo if np.isfinite(lo) else 0 for m, lo in zip(means, los)],
                         [hi - m if np.isfinite(hi) else 0 for m, hi in zip(means, his)]])
        ax.bar(x + off, means, width=w, color=color, label=disp,
               yerr=yerr, ecolor="0.2", capsize=2, error_kw={"elinewidth": 1.0})
    if cut == "AUC":
        ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lv}\n(n={int(counts.get(lv, 0))})" for lv in levels], fontsize=9)
    ax.set_ylabel(f"mean {_axis_label(cut)}")
    ax.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_title(title, fontsize=11)
    fig.text(0.5, 0.005, f"error bars = 95% bootstrap CI of the mean ({N_BOOT} resamples)",
             ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out, script_args=script_args,
          caption=f"{title}. Bars = mean; error bars = 95% bootstrap CI of the mean.")


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

    # Headline "where do we stand" figure for the main repo README: contact
    # R-precision, excluding the near-chance #67 baseline.
    plot_by_config_and_range(
        df, out_dir / "where_we_stand_rprecision.png", cut="R", script_args=sa,
        configs=[c for c in CONFIGS if c[0] != "marinfold-cv1-exp67"],
        title="Where we stand — contact R-precision vs Protenix-v2 / ESMFold / ESMFold2  (n=554)")
    print(f"wrote plots to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--precision-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args()
    main(args.precision_csv, args.out)
