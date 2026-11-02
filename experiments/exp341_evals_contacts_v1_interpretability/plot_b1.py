# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the B1 layer ablation into ``plots/b1_layer_ablation.png``.

From ``analyze_b1.py``'s ``data/b1_deltas.csv``. Left: R-precision lost
(all pairs, separation >= 6) when each layer's attention or MLP output is
mean-ablated, all 116 proteins, protein-bootstrap 95% CI. Right: the same
for long-range pairs (>= 24) by length tercile. The dotted line is minus the
clean R-precision, where an ablation that leaves no contact signal would sit.

    uv run python plot_b1.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
SITES = {"attn": "attention", "mlp": "MLP"}


def main() -> int:
    deltas = pd.read_csv(HERE / "data" / "b1_deltas.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    ax = axes[0]
    rows = deltas[(deltas.split == "all") & (deltas.band == "all")]
    for offset, (site, label) in zip((-0.15, 0.15), SITES.items(), strict=True):
        r = rows[rows.site == site].sort_values("layer")
        ax.errorbar(r.layer + offset, r.delta, yerr=[r.delta - r.lo, r.hi - r.delta], fmt="o-", ms=3, capsize=2,
                    label=label)
    ax.axhline(-rows.clean_rp.iloc[0], color="grey", linestyle=":", linewidth=0.8)
    ax.set_title("all pairs (separation >= 6), 116 proteins", fontsize=10)
    ax.set_ylabel("ΔR-precision (ablated - clean)")

    ax = axes[1]
    rows = deltas[(deltas.band == "long") & deltas.split.str.startswith("length")]
    for color, (split, group) in zip(("C2", "C4", "C3"), rows.groupby("split"), strict=True):
        label = split.split("_", 2)[2] + " residues"
        for site, style in (("attn", "o--"), ("mlp", "s-")):
            r = group[group.site == site].sort_values("layer")
            ax.plot(r.layer, r.delta, style, color=color, ms=3, label=f"{SITES[site]}, {label}")
    ax.set_title("long range (separation >= 24) by length tercile", fontsize=10)
    ax.legend(fontsize=7, ncol=2)

    for ax in axes:
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("layer mean-ablated")
        ax.set_xticks(range(0, 24, 2))
    axes[0].legend(fontsize=8)
    fig.suptitle("B1: exp277 contact readout with one sublayer output mean-ablated", fontsize=10)
    fig.tight_layout()
    out = HERE / "plots" / "b1_layer_ablation.png"
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
