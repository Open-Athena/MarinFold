# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the A4-panel A2 difference in differences into ``plots/a2_panel_did.png``.

Per relative-count bin and pooled, the DiD (exp199 - exp232 on removed,
minus the same on kept) with its bootstrap 95% CI, from
``analyze_a2_panel.py``'s ``data/a2_panel_did.csv``. Left: precision at L.
Right: locality on the contacts in both models' top L, both effect scales.

    uv run python plot_a2_panel.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BINS = ["0", "1-3", "4-15", "16-63", "64+", "all"]
PANELS = {
    "precision at L": {"precision_at_l": "precision at L"},
    "locality, contacts in both models' top L": {"locality_shared_raw": "|ΔP|", "locality_shared_log": "log ratio"},
}


def main() -> int:
    did = pd.read_csv(HERE / "data" / "a2_panel_did.csv", dtype={"bin": str}).set_index(["metric", "bin"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    x = np.arange(len(BINS))
    for ax, (title, metrics) in zip(axes, PANELS.items(), strict=True):
        offsets = np.linspace(-0.12, 0.12, len(metrics)) if len(metrics) > 1 else [0.0]
        for offset, (metric, label) in zip(offsets, metrics.items(), strict=True):
            rows = did.loc[metric].loc[BINS]
            ax.errorbar(x + offset, rows["did"], yerr=[rows["did"] - rows["did_lo"], rows["did_hi"] - rows["did"]],
                        fmt="o", capsize=3, label=label)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(len(BINS) - 1.5, color="grey", linewidth=0.5, linestyle=":")
        ax.set_xticks(x, BINS)
        ax.set_xlabel("surviving relatives in exp232's training data")
        ax.set_ylabel("difference in differences")
        ax.set_title(title, fontsize=10)
        if len(metrics) > 1:
            ax.legend(fontsize=8)
    fig.suptitle("A2 on the A4 panel: exp199 (trained on removed) vs exp232, 20 matched pairs per bin", fontsize=10)
    fig.tight_layout()
    out = HERE / "plots" / "a2_panel_did.png"
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
