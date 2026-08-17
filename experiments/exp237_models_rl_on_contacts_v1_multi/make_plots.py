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
            g = g.dropna(subset=[col])
            if g.empty:
                continue
            ax.plot(g["step"], g[col], label=arm, color=COLORS.get(arm),
                    lw=2.0 if arm != "M-0" else 1.2,
                    ls="--" if arm == "M-0" else "-")
        if col in EXP230:
            ax.axhline(EXP230[col], color="k", lw=0.8, ls=":",
                       label="_#230 eval" if ax is not axes[0, 0] else "#230, at eval")
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
    fig.suptitle("exp237 — the diversity gates, measured every batch\n"
                 "dashed grey is the zero-LR control: whatever it does is the harness, "
                 "not the reward", fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out / "gates_over_training.png", dpi=150)
    print(f"wrote {a.out}/gates_over_training.png")

    if {"policy_kl", "union_pairs"} <= set(df.columns):
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for arm, g in df.groupby("arm"):
            g = g.dropna(subset=["policy_kl", "union_pairs"])
            if g.empty or g["union_pairs"].isna().all():
                continue
            base = g["union_pairs"].iloc[0]
            ax.plot(g["policy_kl"], 100 * g["union_pairs"] / base, "o-", ms=3,
                    label=arm, color=COLORS.get(arm), alpha=0.85)
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
        fig.savefig(a.out / "coverage_vs_kl.png", dpi=150)
        print(f"wrote {a.out}/coverage_vs_kl.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
