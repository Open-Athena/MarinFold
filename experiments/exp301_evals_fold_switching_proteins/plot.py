#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp301 figures. Every panel reads a CSV `analyze.py` wrote; nothing is recomputed here.

    uv run python plot.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"

FOLD1 = "#1f77b4"
FOLD2 = "#d62728"
NEUTRAL = "#7f7f7f"
LABEL_COLOR = {"fold1": FOLD1, "fold2": FOLD2, "neither": NEUTRAL,
               "ambiguous": "#bcbd22", "none": "#cccccc"}


def _finish(fig, name: str) -> None:
    PLOTS.mkdir(exist_ok=True)
    fig.tight_layout()
    out = PLOTS / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out.relative_to(HERE)}")


def plot_fold_preference(pref: pd.DataFrame) -> None:
    """The headline: where each pair's fold score lands, and how lopsided it is."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    order = pref.sort_values("phi").reset_index(drop=True)
    colors = [FOLD1 if v > 0 else FOLD2 for v in order["phi"]]
    err = order["phi_seed_sd"].fillna(0.0)
    ax.barh(np.arange(len(order)), order["phi"], color=colors, height=0.8,
            xerr=err, error_kw=dict(lw=0.6, ecolor="0.35"))
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel(r"fold score  $\varphi$ = recall(fold1-unique) $-$ recall(fold2-unique)")
    ax.set_ylabel(f"fold-switching pair (n={len(order)}, sorted)")
    ax.set_yticks([])
    n1 = int((order["phi"] > 0).sum())
    ax.set_title(f"Prefers fold1 in {n1}/{len(order)} pairs "
                 f"(mean $\\varphi$ = {order['phi'].mean():+.3f})")

    ax = axes[1]
    lim = max(pref[["fold1_R_all", "fold2_R_all"]].to_numpy().max() * 1.08, 0.05)
    ax.plot([0, lim], [0, lim], color="0.6", lw=1, ls="--", zorder=1)
    ax.scatter(pref["fold2_R_all"], pref["fold1_R_all"], s=26,
               c=[FOLD1 if v > 0 else FOLD2 for v in pref["phi"]],
               alpha=0.85, edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlabel("R-precision against fold2")
    ax.set_ylabel("R-precision against fold1")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_title("Above the diagonal = fold1 scored better")
    _finish(fig, "fold_preference.png")


def plot_bimodality(bim: pd.DataFrame) -> None:
    """Is the rollout spread more than sampling noise? The null is the point."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.scatter(bim["null_sd"], bim["phi_sd"], s=26, c=NEUTRAL, alpha=0.8,
               edgecolor="white", lw=0.5, zorder=3)
    lim = max(bim[["null_sd", "phi_sd"]].to_numpy().max() * 1.1, 1e-3)
    ax.plot([0, lim], [0, lim], color="0.6", lw=1, ls="--",
            label="independent-sampling null")
    ax.set_xlabel(r"null sd of $\varphi$ (binomial, from this pair's own recalls)")
    ax.set_ylabel(r"observed sd of $\varphi$ across rollouts")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Points on the line = one fold, sampled repeatedly")

    ax = axes[1]
    ax.hist(bim["dispersion"].dropna(), bins=24, color=NEUTRAL, edgecolor="white")
    ax.axvline(1.0, color="k", lw=1.2, label="no excess spread")
    ax.set_xlabel(r"dispersion = var($\varphi$) / null var")
    ax.set_ylabel("pairs")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"median {bim['dispersion'].median():.2f}; "
                 f"BIC favours 2 components in {int(bim['bic_favours_2'].sum())}/{len(bim)}")
    _finish(fig, "bimodality.png")


def plot_memorization(mem: pd.DataFrame) -> None:
    """Does the model's preference track what its training documents encoded?"""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    groups = [g for g in ("fold1", "fold2", "neither", "ambiguous")
              if (mem["training_fold"] == g).any()]
    data = [mem.loc[mem["training_fold"] == g, "phi"].dropna().to_numpy() for g in groups]
    parts = ax.boxplot(data, tick_labels=[f"{g}\n(n={len(d)})" for g, d in zip(groups, data)],
                       patch_artist=True, widths=0.6, showmeans=True)
    for patch, g in zip(parts["boxes"], groups):
        patch.set_facecolor(LABEL_COLOR.get(g, NEUTRAL))
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for index, values in enumerate(data, start=1):
        ax.scatter(rng.normal(index, 0.06, len(values)), values, s=12, c="0.2", alpha=0.6, zorder=3)
    ax.axhline(0, color="k", lw=1)
    ax.set_ylabel(r"model fold score $\varphi$")
    ax.set_xlabel("fold encoded by the training document (M3)")
    ax.set_title("Model preference vs training-set fold")

    ax = axes[1]
    sub = mem.dropna(subset=["hit_identity"])
    ax.scatter(sub["hit_identity"], sub["phi"], s=26, alpha=0.85, edgecolor="white", lw=0.5,
               c=[LABEL_COLOR.get(g, NEUTRAL) for g in sub["training_fold"]], zorder=3)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("identity to the nearest training sequence")
    ax.set_ylabel(r"model fold score $\varphi$")
    ax.set_title("Preference vs training exposure")
    handles = [plt.Line2D([], [], marker="o", ls="", color=LABEL_COLOR[g], label=g)
               for g in groups]
    ax.legend(handles=handles, frameon=False, fontsize=8, title="training fold", title_fontsize=8)
    _finish(fig, "memorization.png")


def plot_nll_vs_phi(pref: pd.DataFrame, dn: pd.DataFrame) -> None:
    """Two independent readouts of the same question - do they agree?"""
    matched = dn[dn["variant"] == "matched"]
    merged = pref.merge(matched[["pair_id", "delta_nll_tok"]], on="pair_id", how="inner")
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.scatter(merged["delta_nll_tok"], merged["phi"], s=28, alpha=0.85,
               c=[FOLD1 if v > 0 else FOLD2 for v in merged["phi"]],
               edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0, color="k", lw=1)
    ax.axvline(0, color="k", lw=1)
    rho = merged["delta_nll_tok"].corr(merged["phi"], method="spearman")
    ax.set_xlabel(r"$\Delta$NLL/token  (fold1 $-$ fold2; negative = fold1 more likely)")
    ax.set_ylabel(r"sampled fold score $\varphi$  (positive = fold1 sampled more)")
    ax.set_title(f"Teacher forcing vs sampling  (Spearman $\\rho$ = {rho:.2f}, n={len(merged)})")
    ax.text(0.02, 0.02, "agreement puts points in the\nupper-left and lower-right quadrants",
            transform=ax.transAxes, fontsize=8, color="0.35", va="bottom")
    _finish(fig, "nll_vs_phi.png")


def plot_calibration(cal: pd.DataFrame) -> None:
    """The gate: this worker against exp277's published per-protein numbers."""
    sub = cal.dropna(subset=["delta"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 5))
    lim = max(sub[["ours_R_all", "published_R_all"]].to_numpy().max() * 1.08, 0.05)
    ax.plot([0, lim], [0, lim], color="0.6", lw=1, ls="--")
    ax.scatter(sub["published_R_all"], sub["ours_R_all"], s=30, c="#2ca02c",
               alpha=0.85, edgecolor="white", lw=0.5, zorder=3)
    ax.set_xlabel("published exp277 R-precision (#245 / #277)")
    ax.set_ylabel("this worker's R-precision")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    mean_delta = sub["delta"].mean()
    se = sub["delta"].std(ddof=1) / np.sqrt(len(sub))
    ax.set_title(f"Calibration gate, n={len(sub)} eval-val\n"
                 f"mean paired $\\Delta$ = {mean_delta:+.4f} $\\pm$ {se:.4f}")
    _finish(fig, "calibration.png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    pref = pd.read_csv(DATA / "fold_preference.csv")
    plot_fold_preference(pref)

    for name, fn in (("bimodality.csv", plot_bimodality), ("calibration.csv", plot_calibration)):
        path = DATA / name
        if path.exists():
            fn(pd.read_csv(path))

    mem_path = DATA / "memorization.csv"
    if mem_path.exists():
        plot_memorization(pd.read_csv(mem_path))

    nll_path = DATA / "delta_nll.csv"
    if nll_path.exists():
        plot_nll_vs_phi(pref, pd.read_csv(nll_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
