# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot preliminary R-precision results for exp299."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta


ORDER = [
    "delta-stream mean_geomean",
    "delta-stream max_either_side",
    "contacts-v1-r60 contacts_v1_logprob",
]
RANGES = ["all", "short", "medium", "long"]
LABELS = {
    "delta-stream mean_geomean": "delta mean/geomean",
    "delta-stream max_either_side": "delta max/either-side",
    "contacts-v1-r60 contacts_v1_logprob": "contacts-v1 r60",
}


def main() -> None:
    here = Path(__file__).resolve().parent
    df = pd.read_csv(here / "data" / "rprecision_preliminary.csv")
    df["series"] = df["model"] + " " + df["score_readout"]
    pivot = df.pivot(index="range", columns="series", values="mean_r_precision").loc[RANGES, ORDER]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = range(len(RANGES))
    width = 0.24
    offsets = [-width, 0, width]
    for offset, series in zip(offsets, ORDER, strict=True):
        ax.bar([v + offset for v in x], pivot[series], width=width, label=LABELS[series])
    ax.set_xticks(list(x), RANGES)
    ax.set_ylabel("mean R-precision")
    ax.set_title("Preliminary contact R-precision, common 533-protein subset")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    save_plot_with_meta(
        fig,
        here / "plots" / "rprecision_preliminary.png",
        caption=(
            "Mean R-precision for delta-stream step-4000 readouts vs the closest "
            "early contacts-v1 r60 step-3567 baseline on the common 533 proteins."
        ),
    )


if __name__ == "__main__":
    main()
