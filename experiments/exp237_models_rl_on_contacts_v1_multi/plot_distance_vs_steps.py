# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Distance travelled against steps taken — issue #237.

The figure the long-trajectory runs are for. Every accuracy result in this
experiment is ordered by **distance moved** (`policy_kl` against the frozen #230
reference), not by steps, so "train it longer" is only a distinct proposal to the
extent that steps and distance can be pulled apart. This plots exactly that.

Three regimes, and the figure's finding is that **all three are the same road**:

* **lr 1e-5** — crosses the useful window (KL ≲ 0.02) in about a dozen steps;
* **lr 3e-6** — 3.3x smaller, so it takes ~5x the steps to cross the same window
  and then collapses in the same way. Slower, not further.
* **kl_loss_coef 0.05** — intended as a trust region, to hold distance fixed while
  optimisation continued. It does not do that. From step ~50 to ~130 it lies
  almost exactly on top of the lr-3e-6 trace: **a 50x KL penalty bought about what
  a 3x learning-rate cut bought.** It slows travel; it does not stop it. And it
  ends worse — lr 3e-6 was killed by a gate at KL 0.040, while the leashed run
  ran to KL 0.5 with sections per rollout at 54.

Read at 30 batches the leash looks like it is binding (KL 0.0037 against the
unleashed arm's 0.032 by step 24, ~9x). Read at 130 it is not a leash at all.
**That is the finding, and it means "many steps at a genuinely fixed distance"
has still not been run** — the knob reached for does not produce that regime.

The shaded band is where every arm in this experiment peaked.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

#: (label, csv, colour, dash, the eval-confirmed peak step or None)
SERIES = [
    ("M-B · lr 1e-5",              "M-B",       "#1a7f4b", "-",   18),
    ("M-B · lr 3e-6",              None,        "#5fb98a", "-.",  90),
    ("M-K · lr 1e-5",              "M-K",       "#111111", "-",   36),
    ("M-K · lr 1e-5, KL leash ×50", None,       "#c2410c", "-",   None),
]
WINDOW = (0.005, 0.02)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("plots"))
    a = ap.parse_args()

    steps = pd.read_csv("data/training_steps.csv.gz")
    frames = {
        "M-B · lr 1e-5": steps[steps.arm == "M-B"],
        "M-K · lr 1e-5": _mk(),
        "M-B · lr 3e-6": pd.read_csv("data/training_steps_mb_lowlr.csv.gz"),
        "M-K · lr 1e-5, KL leash ×50": pd.read_csv("data/training_steps_mk_leash.csv.gz"),
    }

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhspan(*WINDOW, color="#f0c674", alpha=0.28, lw=0, zorder=0)
    ax.text(2, WINDOW[1] * 0.92, "  the window every arm peaked in", fontsize=8.5,
            color="#7a5c10", va="top")

    for label, _, colour, dash, peak in SERIES:
        d = frames[label].dropna(subset=["policy_kl"]).sort_values("step")
        kl = d["policy_kl"].rolling(10, min_periods=4).median()
        ax.plot(d["step"], kl, color=colour, ls=dash, lw=2.3, label=label)
        if peak is not None:
            row = d[d.step == peak]
            if len(row):
                ax.plot(peak, kl.loc[row.index[0]], "o", color=colour, ms=8,
                        mec="white", mew=1.4, zorder=5)

    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("policy KL from the #230 warm start  (rolling median of 10)")
    ax.set_title("exp237 — nothing tried here holds distance fixed: a 50× KL penalty "
                 "buys about what a 3× smaller learning rate buys", fontsize=10.5)
    ax.annotate("M-B lr3e-6 killed here\n(11.0 sections/rollout)", xy=(180, 0.0397),
                xytext=(196, 0.010), fontsize=8, color="#5fb98a",
                arrowprops=dict(arrowstyle="->", color="#5fb98a", lw=1.1))
    ax.annotate("leash breaks:\nsections 25 → 54", xy=(158, 0.30),
                xytext=(178, 0.62), fontsize=8, color="#c2410c",
                arrowprops=dict(arrowstyle="->", color="#c2410c", lw=1.1))
    ax.annotate("steps 50–130: the KL penalty and the\n3× lower learning rate travel at "
                "the same rate", xy=(95, 0.0082), xytext=(104, 0.0016), fontsize=8,
                color="#444", arrowprops=dict(arrowstyle="->", color="#444", lw=1.0))
    ax.text(0.985, 0.05, "filled markers = each run's best-scoring checkpoint",
            transform=ax.transAxes, fontsize=8, ha="right", color="0.35")
    ax.grid(alpha=.25, which="both")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    save_plot_with_meta(fig, a.out / "distance_vs_steps.png", dpi=150,
        caption=("Distance moved against steps taken. A 3.3x smaller learning rate takes ~5x the "
                 "steps through the same KL window and then fails the same way. Raising "
                 "kl_loss_coef 50x does NOT hold distance fixed — from step 50 to 130 it tracks "
                 "the lr-3e-6 run, then runs away to KL 0.5."))
    print(f"wrote {a.out}/distance_vs_steps.png")
    return 0


def _mk():
    """Arm M-K's own log, which is not in training_steps.csv.gz."""
    from plot_reward_curves import parse
    return parse(Path("/tmp/claude-1000/logs_final/exp237_m_k_lr1e-5.log"))


if __name__ == "__main__":
    raise SystemExit(main())
