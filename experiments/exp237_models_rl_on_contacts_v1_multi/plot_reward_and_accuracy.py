# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Reward and accuracy, on the same x-axis, as two separate figures — issue #237.

Kept apart deliberately, and both keyed on **training step**, because the whole
point is to read one against the other. Arm M-F is the case that makes it worth
doing: its reward climbs for 50 steps while its accuracy is already falling.

* `curves_reward.png` — each arm against **its own** reward. The arms do not
  share one, so a single axis would be a category error; the y-label says which
  quantity each is.
* `curves_accuracy.png` — **consensus R-precision on the legacy 554**, the metric
  of record, at every checkpoint that was scored. Reference lines are the #230
  warm start and the budget-matched plain-22 bar.

Reward is a rolling median over 6 training batches (8 proteins each, so the raw
series is dominated by the protein draw — the lr-0 control swings 2-4x with an
unchanged policy). Accuracy is measured on 554 proteins x 8 rollouts and needs no
smoothing.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402
from plot_reward_curves import parse  # noqa: E402

COLOR = {"M-C": "#1f77b4", "M-F": "#d62728", "M-B": "#1a7f4b",
         "M-BC": "#9467bd", "M-FC": "#e08214", "M-K": "#111111", "M-0": "#9aa5b1"}
REWARD_COL = {"M-C": "consensus_rprec", "M-K": "consensus_rprec", "M-0": "consensus_rprec",
              "M-B": "best_f1", "M-BC": "best_f1", "M-F": "last_f1", "M-FC": "last_f1"}
REWARD_NAME = {"consensus_rprec": "the rollout's own consensus",
               "best_f1": "max$_k$ F1(section $k$)", "last_f1": "F1(last section)"}

#: Every scored checkpoint: arm -> {step: consensus R-precision, legacy 554}.
ACC = {
    "M-C":  {18: 0.5750, 20: 0.5739, 24: 0.5484, 32: 0.4576},
    "M-F":  {18: 0.5647, 36: 0.5529, 60: 0.5157, 84: 0.3974, 120: 0.3758},
    "M-B":  {18: 0.5763, 36: 0.5741, 54: 0.5376, 72: 0.5129, 80: 0.3969},
    "M-BC": {12: 0.5735, 24: 0.5646, 36: 0.5616, 48: 0.5504},
    "M-FC": {12: 0.5728, 18: 0.5732, 24: 0.5717, 36: 0.4818},
    "M-K":  {12: 0.5739, 18: 0.5764, 24: 0.5787, 36: 0.5806},
    "M-0":  {8: 0.5678},
}
WARM, BAR = 0.5673, 0.5896


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", type=Path, default=Path("/tmp/claude-1000/logs_final"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    a = ap.parse_args()

    runs = {}
    frame = pd.read_csv("data/training_steps.csv.gz")
    for arm, g in frame.groupby("arm"):
        runs[arm] = g.sort_values("step").reset_index(drop=True)
    for arm, name in [("M-BC", "exp237_m_bc_lr1e-5.log"), ("M-FC", "exp237_m_fc_lr1e-5.log"),
                      ("M-K", "exp237_m_k_lr1e-5.log")]:
        f = a.logs / name
        if f.exists():
            runs[arm] = parse(f)
    # M-F's continuation, spliced at its true offset.
    f = a.logs / "exp237_m_f_lr1e-5.log"
    if f.exists() and "M-F" in runs:
        cont = parse(f)
        cont["step"] = cont["step"] + 36
        runs["M-F"] = (pd.concat([runs["M-F"][runs["M-F"].step <= 36], cont], ignore_index=True)
                       .drop_duplicates("step", keep="last").sort_values("step"))

    # ---------------- reward ----------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for arm in ["M-0", "M-C", "M-F", "M-B", "M-BC", "M-FC", "M-K"]:
        d = runs.get(arm)
        col = REWARD_COL[arm]
        if d is None or col not in d:
            continue
        g = d.dropna(subset=[col]).sort_values("step")
        ax.plot(g["step"], g[col].rolling(6, min_periods=3).median(), color=COLOR[arm],
                lw=1.5 if arm == "M-0" else 2.4, ls="--" if arm == "M-0" else "-",
                label=f"{arm} · {REWARD_NAME[col]}")
    ax.set_ylim(0.15, 0.63)
    ax.annotate("M-F exits here → 0.006 by step 120", xy=(62, 0.17), fontsize=8, color=COLOR["M-F"])
    ax.set_xlabel("training step"); ax.set_ylabel("the arm's own reward (rolling median of 6)")
    ax.set_title("exp237 — reward during training", fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    save_plot_with_meta(fig, a.out / "curves_reward.png", dpi=150,
        caption=("Each arm against its OWN reward — the arms do not share one. Rolling median of "
                 "6 training batches; the raw per-batch series is dominated by the protein draw."))
    print(f"wrote {a.out}/curves_reward.png")

    # ---------------- accuracy ----------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(BAR, color="#9a6f16", lw=1.6, ls="-.")
    ax.text(0.995, BAR, "plain, 22 rollouts — the bar  ", color="#9a6f16", fontsize=8,
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.axhline(WARM, color="0.35", lw=1.0, ls=":")
    ax.text(0.005, WARM, "  #230 warm start", color="0.35", fontsize=8, va="bottom",
            transform=ax.get_yaxis_transform())
    for arm in ["M-0", "M-C", "M-F", "M-B", "M-BC", "M-FC", "M-K"]:
        pts = sorted(ACC[arm].items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o-", color=COLOR[arm], lw=2.2 if arm != "M-0" else 0,
                ms=7 if arm == "M-K" else 5.5, label=arm,
                mec="white", mew=1.2, zorder=4 if arm == "M-K" else 3)
    ax.set_xlabel("training step")
    ax.set_ylabel("consensus R-precision  (legacy 554, 8 rollouts/protein)")
    ax.set_title("exp237 — accuracy at every scored checkpoint", fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=8.5, loc="lower left", ncol=2)
    fig.tight_layout()
    save_plot_with_meta(fig, a.out / "curves_accuracy.png", dpi=150,
        caption=("Consensus R-precision on the 554-protein exp89 benchmark at every checkpoint "
                 "that was scored. M-K is the only arm still rising at its last scored point."))
    print(f"wrote {a.out}/curves_accuracy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
