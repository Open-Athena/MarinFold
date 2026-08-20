# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Four arms on one axis: what the mode marker did (#175).

Concatenates #160's and #175's per-protein metric rows. They are directly
comparable because both were scored by the same code on the same 554 proteins
and both contain the shared ``exp120-base`` anchor — which lands at 0.4352 here
against 0.4357 there, a 0.0005 gap that is just independent rollout sampling and
sets the scale for reading everything else.

The story the figure has to carry is that the marker did two opposite things:
clean mode recovers most of #160's emission-level regression, and backtracking
mode is *worse* than #160's unconditioned model. Both follow from the same
cause — the model now knows which mode it is in — so they belong on one axis.

    uv run --no-project --with pandas --with matplotlib python plot_modes.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ANCHOR = "exp120-base"
SERIES = [
    ("exp120-base", "base model\n(exp120)", "#8c8c8c"),
    ("exp175-clean", "clean mode\n<contacts-v1>", "#3b6ea5"),
    ("exp175-backtracking", "retraction mode\n<contacts-v1.backtracking>", "#7a4e8c"),
]
# The same two arms trained on the SORTED-flush corpus, for the before/after.
V1 = {"exp175-clean": -0.0068, "exp175-backtracking": -0.0414}
V1_LONG = {"exp175-clean": -0.0095, "exp175-backtracking": -0.0462}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp175", type=Path, default=Path("data_v2/exp160_rows.csv.gz"))
    ap.add_argument("--exp160", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("plots_v2/mode_comparison.png"))
    a = ap.parse_args()

    rows = pd.read_csv(a.exp175)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, rng in zip(axes, ("all", "long")):
        sub = rows[(rows["cut"] == "R") & (rows["range"] == rng)]
        # Anchor #160's arm on ITS OWN exp120-base run, so each delta is paired
        # within the experiment that produced it rather than across two
        # independent rollout samples.
        v1ref = V1 if rng == "all" else V1_LONG
        means, errs, labels, colors = [], [], [], []
        for key, label, color in SERIES:
            v = sub[sub.model == key].precision
            means.append(v.mean())
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(v.size))
            labels.append(label)
            colors.append(color)

        x = np.arange(len(SERIES))
        ax.bar(x, means, width=0.62, color=colors, yerr=errs, capsize=5)
        base = means[0]
        for k, mval in enumerate(means):
            ax.text(k, mval + errs[k] + 0.004, f"{mval:.4f}", ha="center",
                    fontsize=9.5, fontweight="bold")
            if k == 0:
                continue
            s = rows[(rows["cut"] == "R") & (rows["range"] == rng)]
            w = s.pivot_table(index=["dataset", "stem"], columns="model",
                              values="precision").dropna()
            d = (w[SERIES[k][0]] - w[ANCHOR]).to_numpy()
            ci = 1.96 * d.std(ddof=1) / np.sqrt(d.size)
            old = v1ref.get(SERIES[k][0])
            lbl = f"{d.mean():+.4f}\n±{ci:.4f}"
            if old is not None:
                lbl += f"\n(was {old:+.4f})"
            ax.text(k, 0.012, lbl, ha="center", fontsize=8, color="white",
                    fontweight="bold")
        ax.axhline(base, color="#333", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, max(means) + max(errs) + 0.05)
        ax.set_ylabel("R-precision" if rng == "all" else "")
        ax.set_title(f"{rng}-range contacts (n=554; paired Δ vs base in bars)")

    fig.suptitle("Trained on the FIXED corpus: retraction mode's cost falls from "
                 "-0.0414 to -0.0153", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(a.out, dpi=150)
    print(f"[plot] wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
