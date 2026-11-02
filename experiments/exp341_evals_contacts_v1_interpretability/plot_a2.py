# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot A2 from ``analyze_a2.py``'s outputs into ``plots/a2_locality.png``.

Left: mutation effect per residue against sequence distance from the nearer
endpoint of a predicted contact, as a multiple of the contact's mean effect
per residue (mean over proteins, interquartile band). Right: per-protein
locality (log-ratio scale) against length, coloured by set, with the
uniform-spread null.

    uv run python plot_a2.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCALE_LABELS = {"raw": "|ΔP| (plan definition)", "log": "|log P'/P| minus the mutation's median"}


def main() -> int:
    profile = pd.read_csv(HERE / "data" / "a2_distance_profile.csv", dtype={"distance": str})
    per_protein = pd.read_csv(HERE / "data" / "a2_per_protein.csv")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    for scale, label in SCALE_LABELS.items():
        rows = profile[profile["scale"] == scale]
        x = np.arange(len(rows))
        left.plot(x, rows["mean"], marker="o", label=label)
        left.fill_between(x, rows["q25"], rows["q75"], alpha=0.2)
        left.set_xticks(x, rows["distance"])
    left.axhline(1, color="grey", linewidth=0.8)
    left.set_yscale("log")
    left.set_xlabel("sequence distance of mutated residue from nearer contact endpoint")
    left.set_ylabel("effect per residue / mean over chain")
    left.set_title("A2: where mutations move a predicted contact")
    left.legend(fontsize=8)

    for name, frame in per_protein.groupby("set"):
        right.scatter(frame["L"], frame["locality_log"], s=10, alpha=0.7, label=name)
    lengths = np.linspace(per_protein["L"].min(), per_protein["L"].max(), 200)
    order = np.argsort(per_protein["L"].to_numpy())
    right.plot(per_protein["L"].to_numpy()[order], per_protein["null"].to_numpy()[order], color="black",
               linewidth=0.8, label="null (neighbourhood share of chain)")
    right.set_xscale("log")
    right.set_xlim(lengths[0] * 0.9, lengths[-1] * 1.1)
    ticks = [30, 50, 100, 200, 400, 700]
    right.set_xticks(ticks, [str(t) for t in ticks])
    right.minorticks_off()
    right.set_xlabel("protein length")
    right.set_ylabel("locality (share of effect within ±3 of i or j)")
    right.set_title("A2 locality per protein (log-ratio scale)")
    right.legend(fontsize=7)

    fig.tight_layout()
    out = HERE / "plots" / "a2_locality.png"
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
