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

from build_summary import save_plot_with_meta  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
PLOTS = HERE / "plots"

FOLD1 = "#1f77b4"
FOLD2 = "#d62728"
NEUTRAL = "#7f7f7f"
LABEL_COLOR = {"fold1": FOLD1, "fold2": FOLD2, "neither": NEUTRAL,
               "ambiguous": "#bcbd22", "none": "#cccccc"}


#: One-line caption per figure, carried into plots/summary.pdf by
#: build_summary.py's sidecar convention so a slide says what it shows.
#: Kept to a single line each — #224's summary.pdf overlaps the plot when a
#: caption runs past two.
CAPTIONS = {
    "fold_preference.png":
        "Per-pair fold score and per-fold R-precision: MarinFold prefers Fold1 in 52/68 pairs.",
    "bimodality.png":
        "Per-rollout phi spread against a binomial null: the ensemble is one mode, not two.",
    "calibration.png":
        "The gate: this worker against exp277's published per-protein R-precision on eval-val.",
    "memorization.png":
        "Model preference against the fold its own training document encoded, and against exposure.",
    "nll_vs_phi.png":
        "Teacher-forced likelihood against sampled preference: the two readouts agree per protein.",
    "conditioning.png":
        "Dose-response of prompting with fold2 contacts, against the symmetric fold1 control.",
}


def _finish(fig, name: str) -> None:
    PLOTS.mkdir(exist_ok=True)
    fig.tight_layout()
    out = save_plot_with_meta(
        fig, PLOTS / name,
        caption=CAPTIONS.get(name, name),
        script="plot.py", args=[], dpi=150)
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


def plot_conditioning(curve: pd.DataFrame, ks: pd.DataFrame | None) -> None:
    """The dose-response, and what it costs to move the model off its fold."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    has_panel = "panel" in curve.columns
    allp = curve[curve["panel"] == "all"] if has_panel else curve
    balp = curve[curve["panel"] == "balanced"] if has_panel else curve.iloc[0:0]

    ax = axes[0]
    style = {"seed_b": (FOLD2, "o-", "seeded with fold2 contacts"),
             "seed_a": (FOLD1, "s--", "seeded with fold1 contacts (control)")}
    for arm, (color, marker, label) in style.items():
        sub = allp[allp["arm"] == arm].sort_values("k")
        if sub.empty:
            continue
        n_lo, n_hi = int(sub["n_pairs"].min()), int(sub["n_pairs"].max())
        span = f"n={n_hi}" if n_lo == n_hi else f"n={n_hi}\u2192{n_lo}"
        ax.plot(sub["k"], sub["phi"], marker, color=color, lw=1.8, ms=5,
                label=f"{label}  ({span})")
        ax.fill_between(sub["k"], sub["phi_lo"], sub["phi_hi"], color=color, alpha=0.15, lw=0)
        # The balanced panel -- pairs present at every dose -- shown faintly, so
        # the reader can see how much of the high-k slope is composition change.
        bsub = balp[balp["arm"] == arm].sort_values("k")
        if not bsub.empty:
            ax.plot(bsub["k"], bsub["phi"], ":", color=color, lw=1.4, alpha=0.75,
                    label=f"   same, balanced panel (n={int(bsub['n_pairs'].iloc[0])})")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlim(-1, allp["k"].max() * 1.03)
    ax.set_xticks(sorted(allp["k"].unique()))
    ax.set_xlabel("k = contacts placed in the prompt")
    ax.set_ylabel(r"fold score $\varphi$ on the REMAINING sets")
    ax.set_title("Conditioning moves the model, and the arms diverge")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    below = allp[(allp["arm"] == "seed_b") & (allp["phi"] < 0)].sort_values("k")
    if not below.empty:
        k = int(below["k"].iloc[0])
        ax.axvline(k, color="0.4", lw=1, ls=":")
        ax.annotate(f"k* = {k}", xy=(k, 0), xytext=(4, 6), textcoords="offset points",
                    fontsize=9, color="0.3")

    ax = axes[1]
    if ks is not None and ks["flipped"].any():
        # Only pairs that actually needed flipping; the rest preferred fold2
        # before any conditioning and carry no k*.
        ks = ks[ks["needs_flip"]]
        flipped = ks[ks["flipped"]]
        ax.scatter(flipped["n_b"], flipped["k_star"], s=30, c=FOLD2, alpha=0.85,
                   edgecolor="white", lw=0.5, zorder=3, label="flipped")
        never = ks[~ks["flipped"]]
        if not never.empty:
            ax.scatter(never["n_b"], never["max_k_tested"], s=30, c="0.7", marker="^",
                       alpha=0.8, edgecolor="white", lw=0.5, zorder=3,
                       label="never flipped (dose tested)")
        # y = frac * x is a straight line on log-log, but it must be drawn over
        # the real x range: a segment starting at x=0 is invalid on a log axis
        # and matplotlib clips it into a near-horizontal stub.
        xs = np.logspace(np.log10(ks["n_b"].min() * 0.9),
                         np.log10(ks["n_b"].max() * 1.1), 50)
        for frac, ls in ((0.05, ":"), (0.25, "--")):
            ax.plot(xs, frac * xs, color="0.6", lw=1, ls=ls, label=f"{frac:.0%} of |B|")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.8, max(ks["k_star"].max(), ks["max_k_tested"].max()) * 1.6)
        ax.set_xlabel("|B| — contacts unique to fold2")
        ax.set_ylabel("k* for that pair")
        ax.set_title("What each pair needed, against its own size")
        ax.legend(frameon=False, fontsize=8)
    _finish(fig, "conditioning.png")


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

    curve_path = DATA / "conditioning_curve.csv"
    if curve_path.exists():
        ks_path = DATA / "conditioning_k_star.csv"
        plot_conditioning(pd.read_csv(curve_path),
                          pd.read_csv(ks_path) if ks_path.exists() else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
