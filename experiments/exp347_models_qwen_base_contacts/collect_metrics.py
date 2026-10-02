"""Save production learning curves from W&B and plot each document format."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import wandb
from build_summary import save_plot_with_meta
from common import FORMATS, MODELS, run_name

FIELDS = [
    "run",
    "size",
    "format",
    "train/step",
    "train/tokens",
    "run_progress",
    "train/loss",
    "train/tokens_per_second",
    "validation/contact_token_nll",
    "validation/nll_per_contact",
    "_timestamp",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh the CSV from W&B first"
    )
    args = parser.parse_args()
    folder = Path(__file__).parent
    lookup = {
        run_name(size, document_format): (size, document_format)
        for size in MODELS
        for document_format in FORMATS
    }
    rows = []
    for run in (
        wandb.Api().runs(
            "open-athena/MarinFold", filters={"group": "exp347-qwen-base-contacts"}
        )
        if args.refresh
        else []
    ):
        if run.id not in lookup:
            continue
        size, document_format = lookup[run.id]
        for item in run.scan_history():
            rows.append(
                {
                    **{key: item.get(key) for key in FIELDS},
                    "run": run.id,
                    "size": size,
                    "format": document_format,
                }
            )
    path = folder / "data/learning_curves.csv"
    if args.refresh:
        with path.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    else:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            for key in FIELDS[3:]:
                row[key] = float(row[key]) if row[key] else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    max_tokens = max((r["train/tokens"] or 0 for r in rows), default=1e6)
    for ax, document_format in zip(axes, FORMATS, strict=True):
        for size in MODELS:
            selected = sorted(
                (
                    r
                    for r in rows
                    if r["size"] == size
                    and r["format"] == document_format
                    and r["validation/contact_token_nll"] is not None
                ),
                key=lambda r: r["train/tokens"],
            )
            if selected:
                ax.plot(
                    [r["train/tokens"] / 1e6 for r in selected],
                    [r["validation/contact_token_nll"] for r in selected],
                    marker="o",
                    label=size,
                )
        ax.set(
            title=document_format,
            xlabel="Training tokens (millions)",
            ylabel="Contact-continuation token NLL",
            xlim=(0, max(1, max_tokens / 1e6)),
        )
        ax.legend()
        ax.grid(alpha=0.2)
    save_plot_with_meta(
        fig,
        folder / "plots/validation_nll.png",
        caption=(
            "Production validation on 64 held-out AFDB documents. Lower is better "
            "within a format; token NLL across formats is not contact accuracy. "
            "In-progress observations, without error bars."
        ),
    )
    plt.close(fig)
    print(f"Saved {len(rows)} W&B observations")


if __name__ == "__main__":
    main()
