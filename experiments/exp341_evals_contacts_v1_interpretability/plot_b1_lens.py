# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the B1 lens curves into ``plots/b1_lens.png``.

From ``analyze_b1_lens.py``'s ``data/b1_lens_curve.csv``. Left: all-pairs
R-precision decoded at each layer boundary by the logit and tuned lens, all
116 proteins, protein-bootstrap 95% CI; boundary 24 is the model. Right:
long-range R-precision minus the model's, per length tercile, both lenses.

    uv run python plot_b1_lens.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
STYLES = {"logit": ("o-", "logit lens"), "tuned": ("s--", "tuned lens")}


def main() -> int:
    curve = pd.read_csv(HERE / "data" / "b1_lens_curve.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    rows = curve[(curve.split == "all") & (curve.band == "all")]
    for lens, (style, label) in STYLES.items():
        r = rows[rows.lens == lens].sort_values("boundary")
        ax.errorbar(r.boundary, r.rp, yerr=[r.rp - r.lo, r.hi - r.rp], fmt=style, ms=3, capsize=2, label=label)
    ax.set_title("all pairs (separation >= 6), 116 proteins", fontsize=10)
    ax.set_ylabel("R-precision decoded at boundary")
    ax.legend(fontsize=8)

    ax = axes[1]
    rows = curve[(curve.band == "long") & curve.split.str.startswith("length") & (curve.boundary >= 12)]
    for color, (split, group) in zip(("C2", "C4", "C3"), rows.groupby("split"), strict=True):
        for lens, (style, label) in STYLES.items():
            r = group[group.lens == lens].sort_values("boundary")
            ax.plot(r.boundary, r.gap, style, color=color, ms=3, label=f"{label}, {split.split('_', 2)[2]} residues")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title("long range (separation >= 24), minus the model, by length tercile", fontsize=10)
    ax.set_ylabel("R-precision gap to boundary 24")
    ax.legend(fontsize=7, ncol=2)

    for ax in axes:
        ax.set_xlabel("layer boundary (0 = embeddings, k = after layer k - 1)")
    axes[0].set_xticks(range(0, 25, 2))
    axes[1].set_xticks(range(12, 25, 2))
    fig.suptitle("B1: exp277 contact readout decoded from intermediate layers", fontsize=10)
    fig.tight_layout()
    out = HERE / "plots" / "b1_lens.png"
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
