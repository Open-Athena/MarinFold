# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot A3 and A4 from ``analyze_panel.py``'s CSVs into ``plots/panel_a3_a4.png``.

Left: exp232's R-precision on removed structures and their matched kept
controls per relative-count bin (A3). Right: exp199 - exp232 on each set and
the difference in differences (A4). Error bars are the bootstrap 95% CIs.

    uv run python plot_panel.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BINS = ["0", "1-3", "4-15", "16-63", "64+"]


def errorbar(ax, x, frame, column, **kwargs) -> None:
    mean = frame[column].to_numpy()
    err = np.vstack([mean - frame[f"{column}_lo"].to_numpy(), frame[f"{column}_hi"].to_numpy() - mean])
    ax.errorbar(x, mean, yerr=err, marker="o", capsize=3, **kwargs)


def main() -> int:
    dose = pd.read_csv(HERE / "data" / "relative_dose_response.csv", dtype={"bin": str})
    did = pd.read_csv(HERE / "data" / "contamination_did.csv", dtype={"bin": str})
    dose = dose[dose["metric"] == "r_precision"].set_index("bin").loc[BINS]
    did = did[did["metric"] == "r_precision"].set_index("bin").loc[BINS]
    x = np.arange(len(BINS))

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    errorbar(left, x - 0.05, dose, "exp232_removed", label="removed (never seen by exp232)")
    errorbar(left, x + 0.05, dose, "exp232_kept", label="matched kept (in exp232's training)")
    left.set_title("A3: exp232 R-precision by surviving relatives")
    left.set_ylabel("R-precision (separation >= 6)")

    errorbar(right, x - 0.08, did, "removed_delta", label="exp199 - exp232, removed")
    errorbar(right, x, did, "kept_delta", label="exp199 - exp232, kept")
    errorbar(right, x + 0.08, did, "did", label="difference in differences", color="black")
    right.axhline(0, color="grey", linewidth=0.8)
    right.set_title("A4: exposure effect (exp199 saw the removed structures)")
    right.set_ylabel("R-precision difference")

    for ax in (left, right):
        ax.set_xticks(x, BINS)
        ax.set_xlabel("kept relatives in training (identity >= 0.30, coverage >= 0.50)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("200 removed structures per bin, each with a matched kept control; bootstrap 95% CIs", fontsize=9)
    fig.tight_layout()
    out = HERE / "plots" / "panel_a3_a4.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150)
    return 0


if __name__ == "__main__":
    sys.exit(main())
