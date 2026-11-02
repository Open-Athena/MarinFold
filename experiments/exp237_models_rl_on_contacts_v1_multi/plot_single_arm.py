# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""One arm, one reward curve — issue #237.

The combined figure carries seven runs and four different objectives; these are
the two single-arm views worth having on their own. M-K because its reward IS
the quantity plotted, and M-F because its whole run is the clearest picture of
an unconstrained reward direction being found and followed.

The bold line is a ROLLING MEDIAN over 6 batches, and annotations must describe
it rather than the raw series: a first version of the M-F figure labelled 0.608,
which is the highest single batch, on a curve whose median peaks at 0.502 -- and
the lr-0 control established that single batches swing 2-4x on their own.

    python plot_single_arm.py
"""

import sys
sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, numpy as np
from pathlib import Path
from plot_reward_curves import parse
from build_summary import save_plot_with_meta
L = Path("/tmp/claude-1000/logs_final")

def draw(df, col, title, ylab, colour, out, caption, notes=(), vline=None):
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    g = df.dropna(subset=[col]).sort_values("step")
    ax.plot(g["step"], g[col], color=colour, lw=0.7, alpha=0.28, label="per batch (8 proteins)")
    ax.plot(g["step"], g[col].rolling(6, min_periods=3).median(), color=colour, lw=2.6,
            label="rolling median of 6")
    if vline:
        ax.axvline(vline, color="0.35", lw=1.0, ls="--")
        ax.text(vline + 1, ax.get_ylim()[1], " resumed here", fontsize=8, va="top", color="0.35")
    for x, y, txt, ha in notes:
        ax.annotate(txt, xy=(x, y), fontsize=8.5, color=colour, ha=ha, fontweight="600")
    ax.margins(y=0.13)
    ax.set_xlabel("training step"); ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=11); ax.grid(alpha=.25); ax.legend(fontsize=8.5, loc="best")
    fig.tight_layout()
    save_plot_with_meta(fig, out, dpi=150, caption=caption)
    print("wrote", out, f"({len(g)} batches, last median "
          f"{g[col].rolling(6, min_periods=3).median().iloc[-1]:.4f})")

# ---- M-K --------------------------------------------------------------------
mk = parse(L / "exp237_m_k_lr1e-5.log")
draw(mk, "consensus_rprec",
     "Arm M-K — the rollout's own consensus R-precision IS the reward",
     "consensus R-precision (training batches)", "#111111",
     Path("plots/reward_curve_m_k.png"),
     "M-K's reward is exactly the quantity plotted: the rollout's own consensus "
     "R-precision, one scalar per rollout with a GRPO baseline.")

# ---- M-F: original + continuation, spliced ---------------------------------
a = parse(L / "exp237_m_f_lr1e-5.part1.log")
b = parse(L / "exp237_m_f_lr1e-5.log"); b["step"] = b["step"] + 36
mf = pd.concat([a[a.step <= 36], b], ignore_index=True).drop_duplicates("step", keep="last")
draw(mf, "last_f1",
     "Arm M-F — F1 of the last section, its own reward, over the whole run",
     "F1(last section) (training batches)", "#d62728",
     Path("plots/reward_curve_m_f.png"),
     "M-F's reward over both runs. It peaks at step 48 and is at 0.006 by step 120: "
     "the policy found the unconstrained direction (259 sections of 1.4 contacts each) "
     "and took the reward down with it.",
     # The bold line is the ROLLING MEDIAN, so annotate the median's peak.
     # A previous version labelled 0.608 here, which is the highest SINGLE
     # BATCH (step 40) -- a per-batch number on a per-median curve, and the
     # lr-0 control showed single batches swing 2-4x on their own.
     notes=[(53, 0.545, "median peaks 0.502 at step 53", "center"),
            (100, 0.085, "0.006 by step 120", "center")],
     vline=36)
