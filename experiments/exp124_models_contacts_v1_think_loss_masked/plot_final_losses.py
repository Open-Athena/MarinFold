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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    df = pd.read_csv(args.input)
    contacts = df[df["comparable_to_contacts_v1"].astype(bool)].copy()
    think = df[~df["comparable_to_contacts_v1"].astype(bool)].copy()

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = ["#4f83cc", "#8aa6c8", "#d95f02"]
    bars = ax.bar(contacts["run"], contacts["loss"], color=colors, edgecolor="#333333")

    if not think.empty:
        think_loss = float(think.iloc[0]["loss"])
        ax.axhline(think_loss, color="#d95f02", linestyle="--", linewidth=1.2)
        ax.text(
            0.02,
            think_loss + 0.008,
            f"exp124 think-masked val = {think_loss:.3f} (not contacts-v1 comparable)",
            transform=ax.get_yaxis_transform(),
            color="#8c3b00",
            fontsize=9,
            va="bottom",
        )

    for bar, value in zip(bars, contacts["loss"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value) + 0.012,
            f"{float(value):.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    baseline = float(contacts.iloc[0]["loss"])
    exp124_loss = float(contacts.iloc[-1]["loss"])
    delta = exp124_loss - baseline
    ax.text(
        0.98,
        0.95,
        f"exp124 − #117 = +{delta:.3f} nats",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"},
    )

    ax.set_ylabel("contacts-v1 validation loss (nats)")
    ax.set_title("Think-masked training regressed contacts-v1 validation loss")
    ax.set_ylim(2.62, 3.22)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    pdf_output = args.output.with_suffix(".pdf")
    fig.savefig(pdf_output)

    meta = {
        "script": Path(__file__).name,
        "args": ["--input", str(args.input), "--output", str(args.output)],
        "caption": "Final exp124 contacts-v1 validation loss versus #75/#117 baselines.",
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    pdf_output.with_suffix(pdf_output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.output} and {pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
