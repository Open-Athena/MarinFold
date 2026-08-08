# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""exp201 Phase 0 plots: where the contacts-v1 loss budget actually goes.

Consumes ``data/entropy_by_document.csv`` + ``data/entropy_summary.csv`` written
by ``analyze_entropy.py`` and emits two figures:

* ``plots/loss_budget.png`` — the reported val loss split into its nuisance
  floor (by slot kind) and the informative remainder.
* ``plots/nuisance_vs_length.png`` — nuisance share as a function of chain
  length, which is why the effect grows for long proteins.

Usage::

    uv run python plot_entropy.py
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

# The #117 / #150 reference point this experiment measures against.
REFERENCE_VAL_LOSS = 2.7112

# Slot kinds in the order they are stacked, with plot labels and colours.
COMPONENTS = [
    ("statement_head", "sequence statement order", "#c44e52"),
    ("first_endpoint", "contact 1st endpoint", "#dd8452"),
    ("second_endpoint", "contact 2nd endpoint", "#ccb974"),
]


def read_summary(path: Path) -> dict[str, float]:
    with path.open() as handle:
        return {row["metric"]: float(row["value"]) for row in csv.DictReader(handle)}


def read_documents(path: Path) -> dict[str, np.ndarray]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.array([float(row[key]) for row in rows])
        for key in ("seq_len", "num_contacts", "num_tokens", "num_predicted",
                    "sequence_nats", "structure_nats", "nats_per_token")
    }


def plot_loss_budget(summary: dict[str, float], out: Path) -> None:
    """Stacked bar: the reported loss split into nuisance components + signal."""
    floors = [summary[f"floor_nats_per_token::{key}"] for key, _, _ in COMPONENTS]
    nuisance = sum(floors)
    signal = REFERENCE_VAL_LOSS - nuisance

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    left = 0.0
    for value, (_, label, colour) in zip(floors, COMPONENTS):
        ax.barh(0, value, left=left, color=colour, edgecolor="white",
                label=f"{label} — {value:.2f}")
        left += value
    ax.barh(0, signal, left=left, color="#4c72b0", edgecolor="white",
            label=f"informative remainder — {signal:.2f}")

    ax.set_xlim(0, REFERENCE_VAL_LOSS)
    ax.set_yticks([])
    ax.set_xlabel("nats / token")
    ax.set_title(
        f"contacts-v1 val loss {REFERENCE_VAL_LOSS:.4f}: "
        f"{100 * nuisance / REFERENCE_VAL_LOSS:.0f}% is nuisance permutation entropy"
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=2, frameon=False)
    save_plot_with_meta(
        fig, out,
        caption=(
            "The #117 val loss decomposed. Everything left of the blue bar is the "
            "entropy of the generator's two shuffles: an oracle that knew the "
            "structure exactly would still pay it."
        ),
        dpi=150,
    )
    plt.close(fig)


def plot_nuisance_vs_length(docs: dict[str, np.ndarray], out: Path) -> None:
    """Nuisance share vs chain length, binned."""
    seq_len = docs["seq_len"]
    nuisance = (docs["sequence_nats"] + docs["structure_nats"]) / docs["num_predicted"]
    edges = np.array([0, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 2001])
    idx = np.digitize(seq_len, edges) - 1

    centres, shares, counts = [], [], []
    for b in range(len(edges) - 1):
        mask = idx == b
        if mask.sum() < 20:
            continue
        centres.append(seq_len[mask].mean())
        shares.append(100 * nuisance[mask].mean() / REFERENCE_VAL_LOSS)
        counts.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.scatter(seq_len, 100 * nuisance / REFERENCE_VAL_LOSS, s=3, alpha=0.12,
               color="#4c72b0", linewidths=0, label="documents")
    ax.plot(centres, shares, "o-", color="#c44e52", label="binned mean")
    ax.set_xscale("log")
    ax.set_xlabel("chain length (residues)")
    ax.set_ylabel(f"nuisance floor as % of {REFERENCE_VAL_LOSS:.4f}")
    ax.set_title("The longer the protein, the more of the loss is permutation entropy")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save_plot_with_meta(
        fig, out,
        caption=(
            "Nuisance floor per document vs chain length, as a % of the 2.7112 "
            "corpus-average val loss. Rises because log(N!) outpaces N: worst "
            "exactly where the model is weakest (#142)."
        ),
        dpi=150,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    summary = read_summary(args.data_dir / "entropy_summary.csv")
    docs = read_documents(args.data_dir / "entropy_by_document.csv")
    plot_loss_budget(summary, args.plots_dir / "loss_budget.png")
    plot_nuisance_vs_length(docs, args.plots_dir / "nuisance_vs_length.png")
    print(f"wrote {args.plots_dir}/loss_budget.png and "
          f"{args.plots_dir}/nuisance_vs_length.png")


if __name__ == "__main__":
    main()
