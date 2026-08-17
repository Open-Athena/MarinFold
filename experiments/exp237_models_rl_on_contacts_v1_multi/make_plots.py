# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp237's two plots — issue #237.

**Figure 1 — the diversity gates over training.** #237's kill criteria are
quantities, not verdicts, and the only honest way to report them is as curves
with the thresholds drawn on. Four panels: sections per rollout, union pairs per
rollout, mean pairwise Jaccard, and within-rollout consensus R-precision. The
arms are on one axis with the zero-LR control, so "the reward did this" is
separable from "the harness did this" by eye.

**Figure 2 — coverage against KL.** #208's central result was that vote coverage
ordered its eleven-run results table almost perfectly, and that the damage
tracked how far the policy moved. Plotting the same two axes here is what says
whether moving the *unit* of the reward changed that relationship or merely
reproduced it.

    python make_plots.py --steps data/training_steps.csv.gz --out plots/
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

#: Preregistered kill criteria, drawn as horizontal lines.
GATES = {
    "sections_per_rollout": (12.0, "kill below 12"),
    "mean_jaccard": (0.45, "kill above 0.45"),
}
#: #230's eval-time readings on the warm start, for context.
EXP230 = {"sections_per_rollout": 22.0, "union_pairs": 658.0, "mean_jaccard": 0.304}

PANELS = [
    ("sections_per_rollout", "contact sets per rollout"),
    ("union_pairs", "distinct pairs covered per rollout"),
    ("mean_jaccard", "mean pairwise Jaccard between sections"),
    ("consensus_rprec", "within-rollout consensus R-precision (train)"),
]
GATE_WINDOW = 6
COLORS = {"M-C": "#1f77b4", "M-F": "#d62728", "M-B": "#2ca02c", "M-0": "#7f7f7f"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="data/training_steps.csv.gz")
    ap.add_argument("--out", type=Path, default=Path("plots"))
    a = ap.parse_args()

    df = pd.read_csv(a.steps)
    a.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
    for ax, (col, title) in zip(axes.ravel(), PANELS):
        if col not in df:
            ax.set_visible(False)
            continue
        for arm, g in df.groupby("arm"):
            g = g.dropna(subset=[col]).sort_values("step")
            if g.empty:
                continue
            # RAW, faint. Every arm draws the same 8 proteins in the same order
            # at the same step, so these curves share their jaggedness: the
            # batch-to-batch swing is the protein draw, and it is 2-4x larger
            # than anything the reward does (see the M-0 control, which produced
            # the same swing with a policy that did not change at all).
            ax.plot(g["step"], g[col], color=COLORS.get(arm), lw=0.6, alpha=0.28)
            # ROLLING MEDIAN, window 6 -- the quantity the kill criteria are
            # evaluated on, and the only one in which the arms are separable.
            ax.plot(g["step"], g[col].rolling(GATE_WINDOW, min_periods=3).median(),
                    label=arm, color=COLORS.get(arm),
                    lw=2.2 if arm != "M-0" else 1.4,
                    ls="--" if arm == "M-0" else "-")
        if col in EXP230:
            ax.axhline(EXP230[col], color="k", lw=0.8, ls=":",
                       label="#230, at eval" if ax is axes[0, 0] else "_nolegend_")
        if col in GATES:
            y, txt = GATES[col]
            ax.axhline(y, color="crimson", lw=1.0, ls="-.")
            ax.text(0.99, y, txt, color="crimson", fontsize=7, ha="right",
                    va="bottom", transform=ax.get_yaxis_transform())
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    for ax in axes[1]:
        ax.set_xlabel("training step")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("exp237 — the diversity gates, rolling median over 6 batches\n"
                 "faint lines are raw batches: that swing is the protein draw, and the "
                 "zero-LR control produced it with an unchanged policy", fontsize=11)
    fig.tight_layout()
    save_plot_with_meta(
        fig, a.out / "gates_over_training.png", dpi=150,
        caption=(
            "The three preregistered diversity gates, per training batch, rolling median "
            "over 6 batches. Jaccard is the panel that separates the arms: M-B (best-section "
            "reward) rises past #230's 0.304 -- it pays to emit your best guess repeatedly -- "
            "while M-C and M-F fall, i.e. their sections became MORE complementary. All three "
            "nonetheless hit the union-coverage floor and were stopped. Faint lines are raw "
            "batches; that swing is the protein draw, and the zero-LR control reproduced it "
            "with a policy that did not change."))
    print(f"wrote {a.out}/gates_over_training.png")

    if {"policy_kl", "union_pairs"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for arm, g in df.groupby("arm"):
            g = g.dropna(subset=["policy_kl", "union_pairs"])
            if g.empty or g["union_pairs"].isna().all():
                continue
            g = g.sort_values("step")
            med = g["union_pairs"].rolling(GATE_WINDOW, min_periods=3).median()
            base = med.dropna().iloc[0] if med.notna().any() else g["union_pairs"].iloc[0]
            # SCATTER, not a line. policy_kl is not monotone in the step index --
            # it dips and recovers batch to batch -- so joining the points in step
            # order draws a zigzag that reads as structure and is not.
            ax.plot(g["policy_kl"], 100 * med / base, "o", ms=4,
                    label=arm, color=COLORS.get(arm), alpha=0.75)
        ax.axhline(80, color="crimson", lw=1.0, ls="-.")
        ax.text(0.99, 80, "kill below 80 %", color="crimson", fontsize=7,
                ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.set_xlabel("policy KL from the warm start")
        ax.set_ylabel("union pairs per rollout, % of opening")
        ax.set_title("exp237 — does coverage still fall with distance moved?\n"
                     "#208's finding, re-asked with the reward's unit changed", fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        save_plot_with_meta(
            fig, a.out / "coverage_vs_kl.png", dpi=150,
            caption=(
                "Union coverage against distance moved. #208 fitted, and then refuted, a "
                "model in which diversity loss depends only on how far the policy travels. "
                "It does not hold here either: M-C crosses the 80% floor by KL 0.013, M-B "
                "around 0.016, and M-F not until 0.036 -- so per unit of KL the three "
                "rewards cost very different amounts of coverage. They differ in HOW they "
                "lose it (volume for M-C and M-F, redundancy for M-B) and in how fast, but "
                "not in WHETHER."))
        print(f"wrote {a.out}/coverage_vs_kl.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
