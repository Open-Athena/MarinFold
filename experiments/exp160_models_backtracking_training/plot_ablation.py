# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""R-precision before/after fine-tuning, with retraction enabled vs disabled (#160).

The headline −0.020 hides two effects pulling in opposite directions, and this
figure separates them. Scored on the **same rollouts**, so the only thing that
changes between the two right-hand bars is whether a ``<retract>`` is obeyed:

* fine-tuning on the 50:50 mix *cost* R-precision (left → middle), and
* obeying the model's retractions *recovers* part of it (middle → right).

The middle bar — the fine-tuned model read out as if ``<retract>`` did not exist
— is what a pre-#158 regex readout would have measured, and it is the only
honest baseline for "is the retraction mechanism worth obeying".

    uv run --no-project --with pandas --with matplotlib python plot_ablation.py \\
        --rows data/exp160_ablation_rows.csv.gz --out-dir plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = "exp120-base (retraction on)"          # identical to "off": it never retracts
OFF = "exp160-bt50 (retraction off)"
ON = "exp160-bt50 (retraction on)"
SERIES = [
    (BASE, "before fine-tune\n(exp120 base)", "#8c8c8c"),
    (OFF, "after fine-tune\nretraction DISABLED", "#c07a4e"),
    (ON, "after fine-tune\nretraction ENABLED", "#3b6ea5"),
]


def paired(rows: pd.DataFrame, a: str, b: str, rng: str):
    s = rows[(rows["cut"] == "R") & (rows["range"] == rng)]
    w = s.pivot_table(index=["dataset", "stem"], columns="model",
                      values="precision").dropna()
    d = (w[a] - w[b]).to_numpy()
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(d.size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    rows = pd.read_csv(a.rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, rng in zip(axes, ("all", "long")):
        sub = rows[(rows["cut"] == "R") & (rows["range"] == rng)]
        means, errs, labels, colors = [], [], [], []
        for key, label, color in SERIES:
            v = sub[sub.model == key].precision
            means.append(v.mean())
            errs.append(1.96 * v.std(ddof=1) / np.sqrt(v.size))
            labels.append(label)
            colors.append(color)

        x = np.arange(3)
        ax.bar(x, means, width=0.6, color=colors, yerr=errs, capsize=5)
        for k, m in enumerate(means):
            ax.text(k, m + errs[k] + 0.004, f"{m:.4f}", ha="center", fontsize=10,
                    fontweight="bold")

        # The two paired differences, which are what the figure is actually about.
        # Unpaired error bars overlap heavily here — the between-protein spread of
        # R-precision is ~0.3 — so the arrows carry the paired CIs instead.
        d_ft, e_ft = paired(rows, OFF, BASE, rng)
        d_re, e_re = paired(rows, ON, OFF, rng)
        top = max(means) + max(errs) + 0.045
        ax.annotate("", xy=(1, top), xytext=(0, top),
                    arrowprops=dict(arrowstyle="->", color="#b03030", lw=1.6))
        ax.text(0.5, top + 0.004, f"fine-tune {d_ft:+.4f} ±{e_ft:.4f}",
                ha="center", color="#b03030", fontsize=9)
        ax.annotate("", xy=(2, top + 0.028), xytext=(1, top + 0.028),
                    arrowprops=dict(arrowstyle="->", color="#2e8b57", lw=1.6))
        ax.text(1.5, top + 0.032, f"retraction {d_re:+.4f} ±{e_re:.4f}",
                ha="center", color="#2e8b57", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, top + 0.075)
        ax.set_ylabel("R-precision" if rng == "all" else "")
        ax.set_title(f"{rng}-range contacts (n=554 proteins)")

    fig.suptitle("Fine-tuning costs R-precision; obeying retraction wins ~21% of it back",
                 fontsize=12)
    fig.tight_layout()
    out = a.out_dir / "ablation_retraction.png"
    fig.savefig(out, dpi=150)
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
