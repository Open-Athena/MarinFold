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
         "M-BC": "#9467bd", "M-FC": "#e08214", "M-K": "#111111", "M-0": "#9aa5b1",
         "M-KS2": "#be123c", "M-KB": "#7c3aed"}
REWARD_COL = {"M-C": "consensus_rprec", "M-K": "consensus_rprec", "M-0": "consensus_rprec",
              "M-B": "best_f1", "M-BC": "best_f1", "M-F": "last_f1", "M-FC": "last_f1"}
REWARD_NAME = {"consensus_rprec": "the rollout's own consensus",
               "best_f1": "max$_k$ F1(section $k$)", "last_f1": "F1(last section)"}

#: Every scored checkpoint: arm -> {step: consensus R-precision, legacy 554}.
ACC = {
    "M-C":  {18: 0.5750, 20: 0.5739, 24: 0.5484, 28: 0.4990, 32: 0.4576},
    "M-F":  {18: 0.5647, 36: 0.5529, 48: 0.5237, 60: 0.5157, 84: 0.3974, 120: 0.3758},
    "M-B":  {18: 0.5763, 36: 0.5741, 54: 0.5376, 72: 0.5129, 80: 0.3969},
    "M-BC": {12: 0.5735, 24: 0.5646, 36: 0.5616, 48: 0.5504},
    "M-FC": {12: 0.5728, 18: 0.5732, 24: 0.5717, 36: 0.4818},
    "M-K":  {12: 0.5739, 18: 0.5764, 24: 0.5787, 30: 0.5803, 36: 0.5806,
             42: 0.5776, 48: 0.5762},
    "M-0":  {8: 0.5678},
    #: Arm M-KS2 — M-K's base plus the positionally-corrected shaping term. Its
    #: step-36 is a dip (17.8 sections at eval against 20.9 either side), which
    #: is why the whole curve is drawn rather than its best point quoted.
    "M-KS2": {12: 0.5769, 24: 0.5799, 36: 0.5658, 48: 0.5466},
    #: Arm M-KB — M-K at 4x the batch. Killed at step 42; step-36 sits at KL 0.18.
    "M-KB": {12: 0.5749, 24: 0.5475},
}
#: M-B was also run at lr 3e-6 -- a different schedule over the same reward, so it
#: is a separate trace rather than more points on the 1e-5 one. Extended to step
#: 180, where it was killed by the section-count gate at 11.0 sections/rollout:
#: the flat stretch from 60 to 120 is a plateau in STEPS while the policy kept
#: travelling in KL (0.0087 -> 0.0397), and it then failed exactly as the lr 1e-5
#: run did.
ACC_SLOW = {"M-B": {30: 0.5713, 60: 0.5754, 75: 0.5760, 90: 0.5775, 120: 0.5739,
                    150: 0.5575}}
#: Arm M-BP — the same lr-3e-6 run resumed from step 90 with a candidate-count
#: floor on its reward. Drawn from step 90 so it visibly BRANCHES from the trace
#: it is the A/B against: same policy, same lr, same data, one term added.
ACC_PEN = {"M-B": {90: 0.5775, 120: 0.5757, 150: 0.5753, 180: 0.5649}}

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
    slow = {}
    f = a.logs / "exp237_m_b_lr3e-6.log"
    if f.exists():
        slow["M-B"] = parse(f)
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
        # Same reward, slower schedule -- drawn here so the reward panel carries
        # the same set of runs as the accuracy panel.
        if arm in slow:
            gs = slow[arm].dropna(subset=[col]).sort_values("step")
            ax.plot(gs["step"], gs[col].rolling(6, min_periods=3).median(), color=COLOR[arm],
                    lw=2.0, ls="-.", alpha=0.75, label=f"{arm} (lr 3e-6)")
    ax.set_ylim(0.15, 0.63)
    ax.annotate("M-F exits here → 0.006 by step 120", xy=(88, 0.20), fontsize=8, color=COLOR["M-F"])
    ax.set_xlabel("training step"); ax.set_ylabel("the arm's own reward (rolling median of 6)")
    ax.set_title("exp237 — reward during training", fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc="lower left", ncol=2, framealpha=0.95)
    fig.tight_layout()
    save_plot_with_meta(fig, a.out / "curves_reward.png", dpi=150,
        caption=("Each arm against its OWN reward — the arms do not share one. Rolling median of "
                 "6 training batches; the raw per-batch series is dominated by the protein draw."))
    print(f"wrote {a.out}/curves_reward.png")

    # ---------------- accuracy ----------------
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(BAR, color="#9a6f16", lw=1.6, ls="-.")
    ax.text(0.012, BAR, "plain, 22 rollouts — the bar", color="#9a6f16", fontsize=8,
            ha="left", va="bottom", bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.6), zorder=6, transform=ax.get_yaxis_transform())
    ax.axhline(WARM, color="0.35", lw=1.0, ls=":")
    # BELOW the line on the right: M-BP's trace runs just above it there.
    ax.text(0.60, WARM, "#230 warm start", color="0.35", fontsize=8, va="top",
            ha="left", bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.6), zorder=6, transform=ax.get_yaxis_transform())
    for arm in ["M-0", "M-C", "M-F", "M-B", "M-BC", "M-FC", "M-KB", "M-KS2", "M-K"]:
        pts = sorted(ACC[arm].items())
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "o", color=COLOR[arm], lw=2.2 if arm != "M-0" else 0,
                ms=7 if arm == "M-K" else 5.5, label=arm,
                ls="--" if arm in ("M-KS2", "M-KB") else "-",
                mec="white", mew=1.2, zorder=4 if arm == "M-K" else 3)
        if arm in ACC_PEN:
            xs3, ys3 = zip(*sorted(ACC_PEN[arm].items()))
            ax.plot(xs3, ys3, "D--", color="#c2410c", lw=2.2, ms=5.5, mec="white", mew=1.2,
                    label=f"{arm}P (lr 3e-6 + count floor)", zorder=4)
        if arm in ACC_SLOW:
            xs2, ys2 = zip(*sorted(ACC_SLOW[arm].items()))
            ax.plot(xs2, ys2, "o-.", color=COLOR[arm], lw=2.0, ms=5.5, alpha=0.75,
                    mec="white", mew=1.2, label=f"{arm} (lr 3e-6)", zorder=3)
    ax.set_xlabel("training step")
    ax.set_ylabel("consensus R-precision  (legacy 554, 8 rollouts/protein)")
    # No arrow: every path from a free area to (36, 0.5806) crosses three other
    # curves. The label sits in the gap between M-K's tail and the bar instead.
    ax.annotate("M-K 0.5806 · M-KS2 0.5799\n(the two best)", xy=(30, 0.5802),
                xytext=(62, 0.472), fontsize=8.5, fontweight="600", color=COLOR["M-K"], bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.6), zorder=6,
                arrowprops=dict(arrowstyle="->", color=COLOR["M-K"], lw=1.1))
    ax.annotate("M-B lr3e-6 + count floor:\nthe decline is delayed, not prevented",
                xy=(168, 0.5735), xytext=(103, 0.505), fontsize=8, color="#c2410c", bbox=dict(facecolor="white", alpha=0.88, edgecolor="none", pad=1.6), zorder=6,
                arrowprops=dict(arrowstyle="->", color="#c2410c", lw=1.0))
    ax.set_title("exp237 — accuracy at every scored checkpoint  (57 in total)", fontsize=11)
    ax.grid(alpha=.25); ax.legend(fontsize=8, loc="lower left", ncol=3, framealpha=0.95)
    fig.tight_layout()
    save_plot_with_meta(fig, a.out / "curves_accuracy.png", dpi=150,
        caption=("Consensus R-precision on the 554-protein exp89 benchmark at all 37 scored "
                 "checkpoints. Five arms peak by step ~20 and fall away; M-K peaks later, at "
                 "step 30-36, and highest."))
    print(f"wrote {a.out}/curves_accuracy.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
