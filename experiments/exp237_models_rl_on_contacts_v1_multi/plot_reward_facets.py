# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw per-batch reward, one panel per arm — issue #237.

The overlaid version (`curves_reward_bare.png`) is honest but hard to read:
eight noisy series, and for the first ~40 steps they sit almost on top of each
other because every arm walks the SAME prompt pool in the SAME order, so the
batch-to-batch swing is the protein draw rather than the policy. Splitting them
apart shows each arm's own series without that interference.

No smoothing, no annotations. Each panel's y-axis is shared, but the arms do not
share a reward — the panel subtitle names the quantity being plotted.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402
from plot_reward_and_accuracy import COLOR, REWARD_NAME  # noqa: E402
from plot_reward_curves import parse  # noqa: E402

LOGS = Path("/tmp/claude-1000/logs_final")
#: (panel label, reward column, log file or None for the packed CSV, note)
#: Runs that were resumed have their continuation in a rotated `.partN` log with
#: the generator's batch counter restarted at 1, so those are spliced by hand.
PANELS = [
    ("M-C",   "consensus_rprec", None,                         "lr 1e-5"),
    ("M-F",   "last_f1",         "exp237_m_f_lr1e-5.log",      "lr 1e-5"),
    ("M-B",   "best_f1",         None,                         "lr 1e-5"),
    ("M-B",   "best_f1",         "exp237_m_b_lr3e-6.log",      "lr 3e-6"),
    ("M-BC",  "best_f1",         "exp237_m_bc_lr1e-5.log",     "lr 1e-5"),
    ("M-FC",  "last_f1",         "exp237_m_fc_lr1e-5.log",     "lr 1e-5"),
    ("M-K",   "consensus_rprec", "exp237_m_k_lr1e-5.log",      "lr 1e-5"),
    ("M-KB",  "consensus_rprec", "exp237_m_k_bb.log",          "lr 1e-5, 4x batch"),
    ("M-BP",  "best_f1",         "exp237_m_b_pen.log",         "lr 3e-6 + count floor"),
    ("M-KS2", "consensus_rprec", "exp237_m_ks2.log",           "lr 1e-5 + shaping"),
    ("M-KS3", "consensus_rprec", "exp237_m_ks3.log",           "lr 1e-5 + novelty"),
    ("M-KP",  "consensus_rprec", "exp237_m_kp.log",            "lr 1e-5 + per-pair"),
]
COLOR = dict(COLOR, **{"M-KS2": "#be123c", "M-KB": "#7c3aed",
                       "M-KS3": "#0e7490", "M-KP": "#a16207", "M-BP": "#c2410c"})
#: Runs whose panel needs a rotated continuation spliced on, and its step offset.
CONT = {"exp237_m_f_lr1e-5.log": ("packed:M-F", 36),
        "exp237_m_b_lr3e-6.log": ("exp237_m_b_lr3e-6.part2.log", 120),
        "exp237_m_b_pen.log":    (None, 90)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("plots"))
    a = ap.parse_args()

    packed = pd.read_csv("data/training_steps.csv.gz")
    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.4), sharex=True, sharey=True)
    for ax, (label, col, log, note) in zip(axes.ravel(), PANELS):
        if log and (LOGS / log).exists():
            d = parse(LOGS / log)
            src, off = CONT.get(log, (None, 0))
            if src == "packed:M-F":                 # M-F resumed; its head is in the CSV
                base = packed[packed.arm == "M-F"]
                d = pd.concat([base[base.step <= off], d.assign(step=d.step + off)],
                              ignore_index=True).drop_duplicates("step", keep="last")
            elif src and (LOGS / src).exists():     # continuation in a rotated log
                cont = parse(LOGS / src)
                d = pd.concat([d, cont.assign(step=cont.step + off)],
                              ignore_index=True).drop_duplicates("step", keep="last")
            elif off:                               # resumed run: counter restarts at 1
                d = d.assign(step=d.step + off)
        else:
            d = packed[packed.arm == label]
        d = d.dropna(subset=[col]).sort_values("step")
        ax.plot(d["step"], d[col], color=COLOR[label], lw=1.2)
        ax.set_title(f"{label} · {note}", fontsize=10, color=COLOR[label], fontweight="600")
        ax.text(0.97, 0.94, REWARD_NAME[col], transform=ax.transAxes, fontsize=8.5,
                ha="right", va="top", color="0.35")
        ax.grid(alpha=.25)
    axes[0][0].set_ylim(0, 0.78)
    for ax in axes[-1]:
        ax.set_xlabel("training step")
    for row in axes:
        row[0].set_ylabel("the arm's own reward")
    fig.tight_layout()
    save_plot_with_meta(fig, a.out / "curves_reward_facets.png", dpi=150,
        caption=("Raw per-batch reward, one panel per run, no smoothing. Each arm is plotted "
                 "against its OWN reward — the quantity is named in each panel — so the shared "
                 "y-axis is a common range, not a common scale."))
    print(f"wrote {a.out}/curves_reward_facets.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
