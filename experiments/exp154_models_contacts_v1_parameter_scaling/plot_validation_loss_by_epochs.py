# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot validation loss against epochs for every selected W&B run.

Run from this directory::

    uv run python plot_validation_loss_by_epochs.py
"""

import csv
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D

from build_summary import save_plot_with_meta


HERE = Path(__file__).resolve().parent
SOURCE_CSV = HERE / "data" / "wandb_runs.csv"
PLOT_CSV = HERE / "data" / "validation_loss_vs_epochs.csv"
PNG_PATH = HERE / "plots" / "validation_loss_vs_epochs.png"
SVG_PATH = HERE / "plots" / "validation_loss_vs_epochs.svg"

MODEL_ORDER = ("1_5b", "3b")
MODEL_LABELS = {"1_5b": "1.5B", "3b": "3B"}
MODEL_COLORS = {"1_5b": "#2563EB", "3b": "#EA580C"}
ISSUE_MARKERS = {75: "o", 117: "s", 146: "^"}
Y_AXIS_MAX = 3.2
CLIPPED_Y = 3.185

PLOT_FIELDS = (
    "issue",
    "run_id",
    "run_name",
    "model_size",
    "epochs",
    "val_loss",
    "plot_y",
    "is_y_clipped",
    "is_group_best",
    "plot_x",
)


def load_source_rows(path: Path = SOURCE_CSV) -> list[dict[str, str]]:
    """Load the normalized W&B source table."""
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def stable_jitter(run_id: str, width: float = 0.34) -> float:
    """Return a deterministic horizontal jitter centered on zero."""
    digest = hashlib.sha256(run_id.encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    return (fraction - 0.5) * width


def build_plot_rows(source_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """Derive plot coordinates and best-run flags from normalized W&B rows."""
    epochs = sorted({int(row["epochs"]) for row in source_rows})
    epoch_positions = {epoch: index for index, epoch in enumerate(epochs)}

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if row["state"] != "finished":
            raise ValueError(f"plot source contains unfinished run {row['run_name']!r}")
        model_size = row["model_size"]
        if model_size not in MODEL_LABELS:
            raise ValueError(f"unknown model size {model_size!r} for {row['run_name']}")
        grouped[(model_size, int(row["epochs"]))].append(row)

    best_run_ids = {
        min(group, key=lambda row: (float(row["val_loss"]), row["run_name"]))["run_id"]
        for group in grouped.values()
    }

    plot_rows: list[dict[str, Any]] = []
    for row in source_rows:
        epoch = int(row["epochs"])
        val_loss = float(row["val_loss"])
        plot_rows.append(
            {
                "issue": int(row["issue"]),
                "run_id": row["run_id"],
                "run_name": row["run_name"],
                "model_size": row["model_size"],
                "epochs": epoch,
                "val_loss": val_loss,
                "plot_y": CLIPPED_Y if val_loss > Y_AXIS_MAX else val_loss,
                "is_y_clipped": val_loss > Y_AXIS_MAX,
                "is_group_best": row["run_id"] in best_run_ids,
                "plot_x": epoch_positions[epoch] + stable_jitter(row["run_id"]),
            }
        )
    plot_rows.sort(key=lambda row: (row["epochs"], row["model_size"], row["val_loss"], row["run_name"]))
    return plot_rows


def write_plot_csv(rows: Sequence[dict[str, Any]], path: Path = PLOT_CSV) -> None:
    """Write the exact data consumed by the figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=PLOT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_plot_csv(path: Path = PLOT_CSV) -> list[dict[str, Any]]:
    """Load typed plotting rows from the derived CSV."""
    with path.open(newline="") as file:
        raw_rows = list(csv.DictReader(file))
    return [
        {
            **row,
            "issue": int(row["issue"]),
            "epochs": int(row["epochs"]),
            "val_loss": float(row["val_loss"]),
            "plot_y": float(row["plot_y"]),
            "is_y_clipped": row["is_y_clipped"] == "True",
            "is_group_best": row["is_group_best"] == "True",
            "plot_x": float(row["plot_x"]),
        }
        for row in raw_rows
    ]


def plot(rows: Sequence[dict[str, Any]]) -> None:
    """Render the epoch-versus-loss scatter plot in PNG and SVG formats."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.18)

    epochs = sorted({int(row["epochs"]) for row in rows})
    epoch_positions = {epoch: index for index, epoch in enumerate(epochs)}

    for model_size in MODEL_ORDER:
        color = MODEL_COLORS[model_size]
        for issue, marker in ISSUE_MARKERS.items():
            selected = [
                row for row in rows
                if row["model_size"] == model_size and row["issue"] == issue
            ]
            if not selected:
                continue
            ax.scatter(
                [row["plot_x"] for row in selected],
                [row["plot_y"] for row in selected],
                s=38,
                marker=marker,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.58,
                zorder=2,
            )

    clipped_rows = [row for row in rows if row["is_y_clipped"]]
    for model_size in MODEL_ORDER:
        selected = [row for row in clipped_rows if row["model_size"] == model_size]
        if not selected:
            continue
        ax.scatter(
            [row["plot_x"] for row in selected],
            [Y_AXIS_MAX - 0.006] * len(selected),
            s=24,
            marker=6,
            color=MODEL_COLORS[model_size],
            linewidth=1.0,
            alpha=0.7,
            clip_on=False,
            zorder=3,
        )

    best_rows = [row for row in rows if row["is_group_best"]]
    for row in best_rows:
        color = MODEL_COLORS[row["model_size"]]
        ax.scatter(
            row["plot_x"],
            row["plot_y"],
            s=175,
            marker="o",
            facecolor="none",
            edgecolor="#111827",
            linewidth=1.8,
            zorder=4,
        )
        label_offset = (-10, 8) if row["model_size"] == "1_5b" else (10, 8)
        ax.annotate(
            f"{row['val_loss']:.3f}",
            (row["plot_x"], row["plot_y"]),
            xytext=label_offset,
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=color,
            fontweight="bold",
            path_effects=[path_effects.withStroke(linewidth=2.5, foreground="white")],
            zorder=5,
        )

    ax.set_title("Validation loss by training epochs", loc="left", fontsize=18, fontweight="bold", pad=27)
    ax.text(
        0,
        1.015,
        "Every finished latest-subversion run; rings mark the lowest loss per model size and epoch",
        transform=ax.transAxes,
        fontsize=10.5,
        color="#4B5563",
        ha="left",
        va="bottom",
    )
    ax.set_xlabel("Training epochs", labelpad=10)
    ax.set_ylabel("Validation loss (lower is better)", labelpad=10)
    ax.set_xticks([epoch_positions[epoch] for epoch in epochs], [str(epoch) for epoch in epochs])
    ax.set_xlim(-0.45, len(epochs) - 0.55)
    ax.set_ylim(min(row["plot_y"] for row in rows) - 0.05, Y_AXIS_MAX)
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if clipped_rows:
        ax.text(
            0.995,
            0.985,
            f"{len(clipped_rows)} runs above 3.2",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            color="#6B7280",
        )

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=MODEL_COLORS[model],
               markeredgecolor="none", markersize=8, label=f"{MODEL_LABELS[model]} model")
        for model in MODEL_ORDER
    ]
    legend_handles.extend(
        Line2D([0], [0], marker=marker, linestyle="none", markerfacecolor="#6B7280",
               markeredgecolor="none", markersize=7, label=f"Issue #{issue}")
        for issue, marker in ISSUE_MARKERS.items()
    )
    legend_handles.append(
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="#111827", markeredgewidth=1.6, markersize=10,
               label="Best per size × epoch")
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=6,
        frameon=False,
        fontsize=9,
        handletextpad=0.4,
        columnspacing=1.2,
    )

    caption = (
        "All 129 finished runs; color = model size, marker = issue, rings = best per size/epoch, "
        "carets = loss above 3.2."
    )
    save_plot_with_meta(fig, PNG_PATH, caption=caption, dpi=150)
    save_plot_with_meta(fig, SVG_PATH, caption=caption)
    SVG_PATH.write_text(
        "\n".join(line.rstrip() for line in SVG_PATH.read_text().splitlines()) + "\n"
    )
    plt.close(fig)


def main() -> int:
    plot_rows = build_plot_rows(load_source_rows())
    write_plot_csv(plot_rows)
    plot(load_plot_csv())
    print(f"Wrote {PLOT_CSV}")
    print(f"Wrote {PNG_PATH} (150 dpi)")
    print(f"Wrote {SVG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
