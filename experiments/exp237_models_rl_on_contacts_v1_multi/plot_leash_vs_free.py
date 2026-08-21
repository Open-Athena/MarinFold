# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Accuracy against distance, with and without the KL leash — issue #237.

The controlled comparison the leash run was built for. Both traces are **arm
M-K, lr 1e-5, the same reward and the same prompt pool**; the only difference is
`kl_loss_coef`, 0.001 against 0.05. Plotting against `policy_kl` rather than
against the step number is the whole point: it puts the two runs on the axis
that orders every other result in this experiment, so "did the extra steps buy
anything?" becomes a vertical comparison at fixed x.

The answer read off the figure:

* the leashed trace sits **below** the unleashed one at every matched distance,
  not merely at the end;
* and it is **flat** — 0.5728 / 0.5718 / 0.5678 / 0.5712 across steps 30-120,
  no trend — where the unleashed run climbed 0.5739 -> 0.5806 in 36 steps.

So 120 steps under the leash produced nothing over its own step-30 checkpoint.
Distance is not merely what orders the outcomes; **the path taken to a given
distance is not interchangeable**, and the KL penalty's path is the worse one.
That is a stronger statement than "outcome tracks distance", and it is the one
that closes the long-trajectory question.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

#: step -> policy_kl, rolling median of 10 training batches at the checkpoint.
LEASH = {30: 0.0036, 60: 0.0073, 90: 0.0088, 120: 0.0153}
FREE = {12: 0.0054, 18: 0.0095, 36: 0.0162, 42: 0.0199, 30: 0.0287, 24: 0.0317, 48: 0.0344}
WARM, BAR = 0.5673, 0.5896


def read(pattern, steps):
    out = {}
    for st in steps:
        f = Path(f"data/agg_modes_{pattern}{st}.json")
        if f.exists():
            out[st] = json.load(f.open())["legacy554/consensus"]["r_prec"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("plots"))
    a = ap.parse_args()

    free, leash = read("mk_step", FREE), read("leash_step", LEASH)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.axhline(BAR, color="#9a6f16", lw=1.5, ls="-.")
    ax.text(0.995, BAR, "plain, 22 rollouts — the bar  ", color="#9a6f16", fontsize=8,
            ha="right", va="bottom", transform=ax.get_yaxis_transform())
    ax.axhline(WARM, color="0.35", lw=1.0, ls=":")
    ax.text(0.005, WARM, "  #230 warm start", color="0.35", fontsize=8, va="bottom",
            transform=ax.get_yaxis_transform())

    for label, kl, acc, colour, mark in [
        ("M-K · kl_loss_coef 0.001 (as run)", FREE, free, "#111111", "o"),
        ("M-K · kl_loss_coef 0.05 (leashed)", LEASH, leash, "#c2410c", "s"),
    ]:
        pts = sorted((kl[s], acc[s], s) for s in acc)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], mark + "-", color=colour,
                lw=2.2, ms=7, mec="white", mew=1.2, label=label)
        for x, y, s in pts:
            ax.annotate(f"{s}", (x, y), textcoords="offset points", xytext=(0, -13),
                        ha="center", fontsize=7.5, color=colour)

    ax.set_xscale("log")
    ax.set_xlabel("policy KL from the #230 warm start  (point labels are training steps)")
    ax.set_ylabel("consensus R-precision  (legacy 554)")
    ax.set_title("exp237 — same reward, same lr, 3× the steps to the same distance:\n"
                 "the leashed run is uniformly worse and never improves", fontsize=10.5)
    ax.annotate("matched distance:\n120 steps vs 36, and 0.0094 worse",
                xy=(0.0153, 0.5712), xytext=(0.0035, 0.5735), fontsize=8.5,
                color="#c2410c", fontweight="600",
                arrowprops=dict(arrowstyle="->", color="#c2410c", lw=1.1))
    ax.grid(alpha=.25, which="both")
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout()
    save_plot_with_meta(fig, a.out / "leash_vs_free.png", dpi=150,
        caption=("Arm M-K with and without a 50x KL penalty, plotted against distance moved so "
                 "the extra steps are a vertical comparison at fixed x. The leashed trace is "
                 "below the unleashed one at every matched distance and is flat across steps "
                 "30-120 — the path to a distance is not interchangeable."))
    print(f"wrote {a.out}/leash_vs_free.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
