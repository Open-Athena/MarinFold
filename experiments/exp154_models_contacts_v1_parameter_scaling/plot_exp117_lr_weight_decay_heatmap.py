# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot best exp117 validation loss for each learning-rate/weight-decay cell.

Run from this directory::

    uv run --with matplotlib --with numpy --with pandas \
        python plot_exp117_lr_weight_decay_heatmap.py
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Colormap, Normalize
from matplotlib.patches import Rectangle


HERE = Path(__file__).resolve().parent
SOURCE_CSV = HERE / "data" / "wandb_runs.csv"
PLOT_DATA_CSV = HERE / "data" / "exp117_lr_weight_decay_heatmap.csv"
PNG_PATH = HERE / "plots" / "exp117_lr_weight_decay_heatmap.png"
SVG_PATH = HERE / "plots" / "exp117_lr_weight_decay_heatmap.svg"

ISSUE = 117
MODEL_SIZE = "1_5b"
EPOCHS = 8
COLOR_MIN = 2.70
COLOR_MAX = 3.15
MISSING_COLOR = "#F1F5F9"


def format_lr(value: float) -> str:
    """Format a configured learning rate compactly with a power of ten."""
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / 10**exponent
    if math.isclose(mantissa, 1.0, rel_tol=1e-4):
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:.2f}\times10^{{{exponent}}}$"


def load_runs(path: Path = SOURCE_CSV) -> pd.DataFrame:
    """Load finished latest-subversion exp117 1.5B eight-epoch runs."""
    runs = pd.read_csv(path)
    selected = runs.loc[
        (runs["issue"] == ISSUE)
        & (runs["model_size"] == MODEL_SIZE)
        & (runs["epochs"] == EPOCHS)
        & (runs["state"] == "finished")
    ].copy()
    if selected.empty:
        raise ValueError(f"no finished exp{ISSUE} {MODEL_SIZE} {EPOCHS}-epoch runs in {path}")
    if selected["run_id"].duplicated().any():
        raise ValueError("duplicate W&B run IDs in heatmap source")
    if selected["sweep_subversion"].nunique() != 1:
        versions = sorted(selected["sweep_subversion"].unique())
        raise ValueError(f"expected one latest sweep subversion, found {versions}")
    return selected


def build_cells(runs: pd.DataFrame) -> pd.DataFrame:
    """Create the complete LR/WD grid and select minimum loss across batch sizes."""
    learning_rates = sorted(float(value) for value in runs["learning_rate"].unique())
    weight_decays = sorted(float(value) for value in runs["weight_decay"].unique())
    rows: list[dict[str, object]] = []

    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            group = runs.loc[
                np.isclose(runs["learning_rate"], learning_rate)
                & np.isclose(runs["weight_decay"], weight_decay)
            ].sort_values(["val_loss", "run_name"])
            if group.empty:
                rows.append(
                    {
                        "issue": ISSUE,
                        "model_size": MODEL_SIZE,
                        "epochs": EPOCHS,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "n_runs": 0,
                        "n_batch_sizes": 0,
                        "best_val_loss": np.nan,
                        "best_batch_size": np.nan,
                        "best_run_id": "",
                        "best_run_name": "",
                        "best_run_url": "",
                        "is_observed": False,
                    }
                )
                continue

            best = group.iloc[0]
            rows.append(
                {
                    "issue": ISSUE,
                    "model_size": MODEL_SIZE,
                    "epochs": EPOCHS,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "n_runs": len(group),
                    "n_batch_sizes": group["batch_size"].nunique(),
                    "best_val_loss": float(best["val_loss"]),
                    "best_batch_size": int(best["batch_size"]),
                    "best_run_id": best["run_id"],
                    "best_run_name": best["run_name"],
                    "best_run_url": best["run_url"],
                    "is_observed": True,
                }
            )

    cells = pd.DataFrame(rows)
    cells["best_batch_size"] = cells["best_batch_size"].astype("Int64")
    observed = cells["is_observed"]
    global_best = float(cells.loc[observed, "best_val_loss"].min())
    cells["is_global_best"] = observed & np.isclose(cells["best_val_loss"], global_best)
    cells["is_color_clipped"] = observed & (cells["best_val_loss"] > COLOR_MAX)
    return cells


def annotation_color(value: float, cmap: Colormap, norm: Normalize) -> str:
    """Choose dark or light text from the heatmap-cell luminance."""
    red, green, blue, _ = cmap(norm(min(value, COLOR_MAX)))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "white" if luminance < 0.52 else "#111827"


def plot(cells: pd.DataFrame) -> None:
    """Render the annotated LR-by-WD heatmap in PNG and SVG formats."""
    learning_rates = sorted(float(value) for value in cells["learning_rate"].unique())
    weight_decays = sorted(float(value) for value in cells["weight_decay"].unique())
    loss_matrix = np.full((len(learning_rates), len(weight_decays)), np.nan)
    cell_lookup: dict[tuple[float, float], pd.Series] = {}
    for _, cell in cells.iterrows():
        key = (float(cell["learning_rate"]), float(cell["weight_decay"]))
        cell_lookup[key] = cell
        if bool(cell["is_observed"]):
            row = learning_rates.index(key[0])
            column = weight_decays.index(key[1])
            loss_matrix[row, column] = float(cell["best_val_loss"])

    plt.style.use("seaborn-v0_8-white")
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(MISSING_COLOR)
    norm = Normalize(vmin=COLOR_MIN, vmax=COLOR_MAX, clip=True)

    fig, axis = plt.subplots(figsize=(10.8, 6.8))
    fig.subplots_adjust(left=0.18, right=0.87, top=0.80, bottom=0.22)
    image = axis.imshow(np.ma.masked_invalid(loss_matrix), cmap=cmap, norm=norm, aspect="auto")

    axis.set_xticks(np.arange(len(weight_decays)), [f"{value:g}" for value in weight_decays])
    axis.set_yticks(np.arange(len(learning_rates)), [format_lr(value) for value in learning_rates])
    axis.set_xlabel("Weight decay", labelpad=12)
    axis.set_ylabel("Learning rate", labelpad=12)
    axis.set_xticks(np.arange(len(weight_decays) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(learning_rates) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color="white", linewidth=2.0)
    axis.tick_params(which="minor", bottom=False, left=False)
    axis.tick_params(axis="both", which="major", length=0)
    axis.spines[:].set_visible(False)

    for row, learning_rate in enumerate(learning_rates):
        for column, weight_decay in enumerate(weight_decays):
            cell = cell_lookup[(learning_rate, weight_decay)]
            if not bool(cell["is_observed"]):
                axis.text(column, row, "—", ha="center", va="center", color="#94A3B8", fontsize=13)
                continue

            value = float(cell["best_val_loss"])
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=annotation_color(value, cmap, norm),
                fontsize=11,
                fontweight="bold" if bool(cell["is_global_best"]) else "normal",
            )
            if bool(cell["is_global_best"]):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111827",
                        linewidth=2.2,
                    )
                )

    colorbar = fig.colorbar(image, ax=axis, fraction=0.045, pad=0.035, extend="max")
    colorbar.set_label("Best validation loss", labelpad=10)

    fig.suptitle(
        "Best exp117 validation loss by learning rate and weight decay",
        x=0.06,
        y=0.935,
        ha="left",
        fontsize=20,
    )
    fig.text(
        0.06,
        0.865,
        "Finished 1.5B, eight-epoch runs; each cell is the minimum across tested batch sizes. "
        "An em dash marks an untested cell.",
        color="#475569",
        fontsize=10.5,
    )
    clipped = cells.loc[cells["is_color_clipped"]]
    fig.text(
        0.18,
        0.055,
        f"Outline marks the overall best cell. Color is capped at {COLOR_MAX:g}; "
        f"{len(clipped)} higher value remains numerically annotated.",
        color="#64748B",
        fontsize=9,
    )

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    SVG_PATH.write_text("\n".join(line.rstrip() for line in SVG_PATH.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> int:
    """Build exact heatmap data and render the figure."""
    cells = build_cells(load_runs())
    PLOT_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    cells.to_csv(PLOT_DATA_CSV, index=False)
    plot(cells)
    print(f"Wrote {PLOT_DATA_CSV}")
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
