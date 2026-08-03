# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot a representative contacts-v1 validation-loss gap against bootstrap SEM."""

import argparse
import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_LOSSES = HERE / "data" / "loss_delta_vs_sem.csv"
DEFAULT_BOOTSTRAP = HERE / "data" / "exp117_e16_final_step35679_bootstrap_summary.json"
DEFAULT_OUTPUT = HERE / "plots" / "loss_delta_vs_sem.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--losses", type=Path, default=DEFAULT_LOSSES)
    parser.add_argument("--bootstrap-summary", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    losses = pd.read_csv(args.losses)
    if len(losses) != 2:
        raise ValueError(f"expected exactly two runs, got {len(losses)}")

    with args.bootstrap_summary.open() as handle:
        bootstrap = json.load(handle)
    stderr = float(bootstrap["stderr"])

    val_losses = losses["val_loss"].astype(float).to_numpy()
    delta = abs(float(val_losses[0] - val_losses[1]))
    delta_over_sem = delta / stderr

    fig, ax = plt.subplots(figsize=(5.8, 4.4))
    colors = ["#8aa6c8", "#2f6fb0"]
    bars = ax.bar(losses["run"], val_losses, yerr=[stderr, stderr], capsize=6, color=colors, edgecolor="#333333")

    ax.set_ylabel("contacts-v1 validation loss (nats)")
    ax.set_title("A meaningful validation-loss gap vs bootstrap noise")
    ax.grid(axis="y", alpha=0.25)

    lower = min(val_losses) - max(0.01, stderr * 4)
    upper = max(val_losses) + max(0.012, stderr * 5)
    ax.set_ylim(lower, upper)

    for bar, value in zip(bars, val_losses, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + stderr * 1.4,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    x0 = bars[0].get_x() + bars[0].get_width() / 2
    x1 = bars[1].get_x() + bars[1].get_width() / 2
    y = max(val_losses) + stderr * 3.2
    ax.plot([x0, x0, x1, x1], [y - stderr * 0.4, y, y, y - stderr * 0.4], color="#333333", linewidth=1.2)
    ax.text(
        (x0 + x1) / 2,
        y + stderr * 0.25,
        f"Δ = {delta:.4f} nats = {delta_over_sem:.1f}× bootstrap SEM",
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.text(
        0.02,
        0.02,
        f"Error bars: document-level bootstrap SEM = {stderr:.4f} nats",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    pdf_output = args.output.with_suffix(".pdf")
    fig.savefig(pdf_output)

    meta = {
        "script": Path(__file__).name,
        "args": [
            "--losses",
            str(args.losses),
            "--bootstrap-summary",
            str(args.bootstrap_summary),
            "--output",
            str(args.output),
        ],
        "caption": (
            "Validation loss for #75 E8 and #117 E16 final, with document-level "
            "Poisson-bootstrap SEM error bars from exp188."
        ),
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    pdf_output.with_suffix(pdf_output.suffix + ".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {args.output} and {pdf_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
