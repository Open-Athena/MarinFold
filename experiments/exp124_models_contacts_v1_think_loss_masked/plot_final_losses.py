# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot exp124 final validation losses against contacts-v1 baselines."""

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "data" / "final_losses.csv"
DEFAULT_OUTPUT = HERE / "plots" / "final_losses.png"
METRIC_ORDER = ["ordinary contacts-v1 val", "think-augmented masked val"]
RUN_COLORS = {
    "exp177 CE baseline": "#4f83cc",
    "#117 E16 final": "#4f83cc",
    "#75 E8": "#8aa6c8",
    "exp124 think-masked": "#d95f02",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    df = pd.read_csv(args.input)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharey=False)

    for ax, metric in zip(axes, METRIC_ORDER, strict=True):
        metric_df = df[df["metric"] == metric].copy()
        bars = ax.bar(
            metric_df["run"],
            metric_df["loss"],
            color=[RUN_COLORS[run] for run in metric_df["run"]],
            edgecolor="#333333",
        )
        for bar, value in zip(bars, metric_df["loss"], strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                float(value) + 0.008,
                f"{float(value):.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(metric)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=15)

    contacts = df[df["metric"] == "ordinary contacts-v1 val"]
    exp124_contacts = float(contacts[contacts["run"] == "exp124 think-masked"].iloc[0]["loss"])
    exp177_contacts = float(contacts[contacts["run"] == "exp177 CE baseline"].iloc[0]["loss"])
    axes[0].text(
        0.98,
        0.95,
        f"exp124 − exp177 CE ≈ {exp124_contacts - exp177_contacts:+.3f} nats",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    axes[0].set_ylim(3.09, 3.15)
    axes[0].set_ylabel("validation loss (nats)")

    think = df[df["metric"] == "think-augmented masked val"]
    exp124_think = float(think[think["run"] == "exp124 think-masked"].iloc[0]["loss"])
    exp117_think = float(think[think["run"] == "#117 E16 final"].iloc[0]["loss"])
    axes[1].text(
        0.98,
        0.95,
        f"exp124 − #117 = {exp124_think - exp117_think:+.3f} nats",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    axes[1].set_ylim(3.06, 3.13)

    fig.suptitle("exp124 pause-token run: same-scale ordinary val and think-masked val")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    pdf_output = args.output.with_suffix(".pdf")
    fig.savefig(pdf_output)

    meta = {
        "script": Path(__file__).name,
        "args": ["--input", str(args.input), "--output", str(args.output)],
        "caption": "Final exp124 validation losses. Ordinary contacts-v1 loss is compared to the same-era exp177 CE baseline; historical #75/#117 W&B ordinary losses use an older loss scale and are not plotted as direct baselines.",
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    pdf_output.with_suffix(pdf_output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.output} and {pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
