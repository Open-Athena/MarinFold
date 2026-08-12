# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Figures for issue #211. Writes plots/*.png with build_summary sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

METRIC = "contact_excess_per_contact"
ARM_LABEL = {
    "gt": "ground truth", "gt_subsampled": "GT, size-matched",
    "rollout": "rollout (treatment)", "chimera_marginal": "marginal chimera (null)",
    "chimera_splice": "splice chimera", "random": "sep-matched random",
    "decoy": "decoy protein",
}
ORDER = ["gt", "gt_subsampled", "rollout", "chimera_marginal", "chimera_splice",
         "decoy", "random"]
C_TREAT, C_NULL, C_REF = "#1f77b4", "#d62728", "#999999"


def fig_arms(df, out):
    w = df.pivot_table(index="record_id", columns="arm", values=METRIC, aggfunc="mean")
    arms = [a for a in ORDER if a in w]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [C_TREAT if a == "rollout" else C_NULL if a.startswith("chimera")
              else C_REF for a in arms]
    bp = ax.boxplot([w[a].dropna() for a in arms],
                tick_labels=[ARM_LABEL[a] for a in arms],
                    showfliers=False, patch_artist=True, medianprops=dict(color="k"))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    ax.set_ylabel("contact excess per contact (lower = more 3D-consistent)")
    ax.set_title("Geometric self-consistency by arm")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_plot_with_meta(fig, out / "arms.png", script="plot_results.py",
                        caption="Per-protein mean consistency by arm. Blue = the "
                                "rollout; red = the size- and marginal-matched nulls.")
    plt.close(fig)


def fig_paired(df, out):
    w = df.pivot_table(index="record_id", columns="arm", values=METRIC, aggfunc="mean")
    if "rollout" not in w or "chimera_marginal" not in w:
        return
    d = (w["chimera_marginal"] - w["rollout"]).dropna()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(d, bins=45, color=C_TREAT, alpha=0.8)
    ax.axvline(0, color="k", lw=1.2)
    ax.axvline(d.mean(), color=C_NULL, lw=2,
               label=f"mean {d.mean():+.4f}  ({100 * (d > 0).mean():.0f}% > 0)")
    ax.set_xlabel("chimera − rollout   (> 0 ⇒ the rollout is more self-consistent)")
    ax.set_ylabel("proteins")
    ax.set_title("Paired effect: rollout vs marginal-matched chimera")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_plot_with_meta(fig, out / "paired_effect.png", script="plot_results.py",
                        caption="Per-protein paired difference. Positive means the "
                                "jointly-generated set embeds better than its "
                                "marginal-matched null.")
    plt.close(fig)


def fig_selection(out):
    p = Path("data/per_protein_rho.csv")
    if not p.exists():
        return
    rho = pd.read_csv(p)
    sel = pd.read_csv("data/per_protein_selection.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.hist(rho["rho"], bins=40, color=C_TREAT, alpha=0.8)
    a1.axvline(0, color="k", lw=1.2)
    a1.axvline(rho["rho"].mean(), color=C_NULL, lw=2,
               label=f"mean {rho['rho'].mean():+.3f}")
    a1.set_xlabel("Spearman ρ(excess, precision) within a protein")
    a1.set_ylabel("proteins"); a1.legend(); a1.grid(alpha=0.3)
    a1.set_title("Does consistency rank rollouts?")

    gains = {"most-consistent": (sel["selected"] - sel["mean"]).mean(),
             "oracle-best": (sel["oracle"] - sel["mean"]).mean(),
             "worst": (sel["worst"] - sel["mean"]).mean()}
    a2.bar(list(gains), list(gains.values()),
           color=[C_TREAT, "#2ca02c", C_NULL], alpha=0.85)
    a2.axhline(0, color="k", lw=1)
    a2.set_ylabel("precision vs a random rollout")
    a2.set_title("Selection gain"); a2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_plot_with_meta(fig, out / "selection.png", script="plot_results.py",
                        caption="Left: per-protein rank correlation (negative = "
                                "useful). Right: precision gained by selecting on "
                                "consistency, against the oracle headroom.")
    plt.close(fig)


def fig_by_length(df, out):
    w = df.pivot_table(index="record_id", columns="arm", values=METRIC, aggfunc="mean")
    w = w.join(df.groupby("record_id")["L"].first())
    if "rollout" not in w or "chimera_marginal" not in w:
        return
    w["delta"] = w["chimera_marginal"] - w["rollout"]
    bins = [(100, 200), (200, 350), (350, 800)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    xs, ys, es, ns = [], [], [], []
    for i, (lo, hi) in enumerate(bins):
        s = w[(w["L"] >= lo) & (w["L"] < hi)]["delta"].dropna()
        if len(s) < 8:
            continue
        xs.append(i); ys.append(s.mean()); es.append(s.sem() * 1.96); ns.append(len(s))
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color=C_TREAT, capsize=4, lw=2)
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lo}–{hi}\nn={n}" for (lo, hi), n in zip(bins, ns)])
    ax.set_xlabel("protein length (residues)")
    ax.set_ylabel("chimera − rollout")
    ax.set_title("Paired effect by protein length")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_plot_with_meta(fig, out / "by_length.png", script="plot_results.py",
                        caption="Paired effect stratified by length, 95% CI. The "
                                "metric is near-blind below L≈100, so that bin is "
                                "excluded.")
    plt.close(fig)


def fig_gate(out):
    p = Path("data/gt_gate.csv")
    if not p.exists():
        return
    g = pd.read_csv(p)
    w = g.pivot_table(index="record_id", columns="arm", values=METRIC)
    w = w.join(g.groupby("record_id")[["L", "has_chain_break"]].first())
    w = w[~w["has_chain_break"]]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for col, lab, c in (("gt", "ground truth", "#2ca02c"),
                        ("decoy", "decoy protein", C_REF),
                        ("random_0", "sep-matched random", C_NULL)):
        if col in w:
            ax.hist(w[col].dropna(), bins=50, alpha=0.55, label=lab, color=c)
    ax.set_xlabel("contact excess per contact")
    ax.set_ylabel("proteins"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_xlim(0, np.nanpercentile(w["random_0"], 97))
    ax.set_title("Calibration gate: the metric separates real folds from non-folds")
    fig.tight_layout()
    save_plot_with_meta(fig, out / "calibration_gate.png", script="plot_results.py",
                        caption="Ground truth sits 5.6× below random. A decoy "
                                "protein ties with the truth — the score is "
                                "sequence-blind.")
    plt.close(fig)


def fig_power(out):
    p = Path("data/power_check.csv")
    if not p.exists():
        return
    d = pd.read_csv(p)
    w = d.pivot_table(index="record_id", columns="frac_corrupt", values=METRIC)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    fr = sorted(w.columns)
    m = [w[f].mean() for f in fr]
    e = [w[f].sem() * 1.96 for f in fr]
    ax.errorbar(fr, m, yerr=e, fmt="o-", color=C_TREAT, capsize=4, lw=2)
    ax.axvspan(0.35, 0.45, color=C_NULL, alpha=0.25, label="#199 operating band")
    ax.set_xlabel("fraction of true contacts replaced by random ones")
    ax.set_ylabel("contact excess per contact")
    ax.set_title("Dose–response: the score is graded, not binary")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    save_plot_with_meta(fig, out / "dose_response.png", script="plot_results.py",
                        caption="Score vs corruption, 95% CI. Grey band marks where "
                                "#199 actually generates (R-precision ≈ 0.59).")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, default=Path("data/arm_scores.csv"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    ap.add_argument("--min-length", type=int, default=100)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    fig_gate(a.out)
    fig_power(a.out)
    if a.scores.exists():
        df = pd.read_csv(a.scores)
        m = df[(df["L"] >= a.min_length) & (~df["has_chain_break"])]
        fig_arms(m, a.out); fig_paired(m, a.out)
        fig_by_length(m, a.out); fig_selection(a.out)
    print(f"[plot] wrote {len(list(a.out.glob('*.png')))} figures to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
