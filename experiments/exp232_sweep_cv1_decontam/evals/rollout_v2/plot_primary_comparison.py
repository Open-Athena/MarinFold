# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot exp232 per-protein boxes and validation-loss/R-precision scatter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TABLE = HERE / "data" / "comparison.csv"
ROWS = HERE / "data" / "all_r_rows.csv.gz"
DEFAULT_OUTPUT = HERE / "plots" / "final_checkpoint_rprecision.png"
BOX_ORDER = (
    "exp146",
    "exp166",
    "cw-p06-aug",
    "cw-p06-cool",
    "protenix",
    "exp232-m2-p06-decontam",
    "exp232-m1-p02-decontam",
)
BOX_LABELS = {
    "exp146": "#146 3B",
    "exp166": "#166\nAA aug",
    "cw-p06-aug": "CW p06\naug",
    "cw-p06-cool": "CW p06\ncooldown",
    "protenix": "Protenix-v2\nsingle-seq",
    "exp232-m2-p06-decontam": "#232 decontam\nm2-p06",
    "exp232-m1-p02-decontam": "#232 decontam\nm1-p02",
}
COLORS = {
    "previous": "#8f8b86",
    "computed_here": "#d55e00",
    "protenix": "#2a78d6",
    "fit": "#52514e",
}
ANNOTATIONS = {
    "exp146": ((10, 24), "left", "#146 · 3B"),
    "exp166": ((8, 18), "left", "#166 AA aug"),
    "cw-p06-aug": ((-10, 20), "right", "CW p06 aug"),
    "cw-p06-cool": ((-6, 22), "right", "CW p06 cooldown"),
    "exp232-m2-p06-decontam": ((10, 10), "left", "#232 decontam m2-p06"),
    "exp232-m1-p02-decontam": ((10, -22), "left", "#232 decontam m1-p02"),
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sigmoid(x: np.ndarray, upper: float, midpoint: float, width: float) -> np.ndarray:
    """Monotone loss-to-R-precision curve."""

    exponent = np.clip((x - midpoint) / width, -700, 700)
    return upper / (1.0 + np.exp(exponent))


def load_values() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load inputs and validate every per-protein mean."""

    table = pd.read_csv(TABLE)
    rows = pd.read_csv(ROWS)
    if set(rows.key) != set(BOX_ORDER):
        raise ValueError(f"unexpected per-protein keys: {sorted(rows.key.unique())}")
    values = {
        key: rows.loc[rows.key == key, "precision"].to_numpy(dtype=float)
        for key in BOX_ORDER
    }
    expected = table.set_index("key").r_all.to_dict()
    expected["protenix"] = 0.6031578401726864
    for key, key_values in values.items():
        if key_values.size != 554:
            raise ValueError(f"{key} has {key_values.size} values; expected 554")
        if not np.isclose(key_values.mean(), expected[key], atol=1e-14):
            raise ValueError(f"{key} mean does not match the comparison table")
    return table, values


def fit_sigmoid(table: pd.DataFrame) -> dict[str, object]:
    """Fit the descriptive curve to the five unique 1.5B checkpoints."""

    selected = table[table.fit_group.notna()]
    x = selected.loss.to_numpy(dtype=float)
    y = selected.r_all.to_numpy(dtype=float)
    parameters, _ = curve_fit(
        sigmoid,
        x,
        y,
        p0=(min(0.95, float(y.max()) + 0.03), 3.1, 0.05),
        bounds=([0.35, 2.5, 0.001], [1.0, 3.5, 2.0]),
        maxfev=100_000,
    )
    predicted = sigmoid(x, *parameters)
    residual_sum = float(np.square(y - predicted).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    return {
        "upper": float(parameters[0]),
        "midpoint": float(parameters[1]),
        "width": float(parameters[2]),
        "r_squared": 1.0 - residual_sum / total_sum,
        "input_groups": selected.fit_group.tolist(),
        "minimum_observed_loss": float(x.min()),
        "maximum_observed_loss": float(x.max()),
    }


def draw_boxplot(axis: plt.Axes, values: dict[str, np.ndarray]) -> None:
    """Draw legacy-554 per-protein distributions."""

    boxes = axis.boxplot(
        [values[key] for key in BOX_ORDER],
        widths=0.43,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#111111",
            "markersize": 4.5,
        },
        medianprops={"color": "#111111", "linewidth": 1.3},
        flierprops={"marker": ".", "markersize": 2.1, "alpha": 0.23},
    )
    new_keys = {"exp232-m2-p06-decontam", "exp232-m1-p02-decontam"}
    for patch, key in zip(boxes["boxes"], BOX_ORDER, strict=True):
        color = (
            COLORS["protenix"]
            if key == "protenix"
            else COLORS["computed_here"]
            if key in new_keys
            else COLORS["previous"]
        )
        patch.set_facecolor(color)
        patch.set_alpha(0.83)
    for position, key in enumerate(BOX_ORDER, start=1):
        axis.text(
            position,
            1.018,
            f"{values[key].mean():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(
        np.arange(1, len(BOX_ORDER) + 1),
        [BOX_LABELS[key] for key in BOX_ORDER],
        fontsize=8,
    )
    axis.set_ylabel("Per-protein all-range R-precision")
    axis.set_ylim(-0.02, 1.095)
    axis.set_title("A · Distributions across 554 proteins", pad=13)
    axis.grid(axis="y", color="#d8d7d2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def draw_scatter(
    axis: plt.Axes, table: pd.DataFrame, values: dict[str, np.ndarray]
) -> dict[str, object]:
    """Draw mean all-range R-precision against current validation loss."""

    fit = fit_sigmoid(table)
    parameters = (float(fit["upper"]), float(fit["midpoint"]), float(fit["width"]))
    observed = np.linspace(
        float(fit["maximum_observed_loss"]),
        float(fit["minimum_observed_loss"]),
        240,
    )
    extrapolated = np.linspace(float(fit["minimum_observed_loss"]), 2.92, 80)
    axis.plot(
        observed, sigmoid(observed, *parameters), color=COLORS["fit"], linewidth=1.8
    )
    axis.plot(
        extrapolated,
        sigmoid(extrapolated, *parameters),
        color=COLORS["fit"],
        linewidth=1.5,
        linestyle=":",
    )
    protenix = float(values["protenix"].mean())
    axis.axhline(protenix, color=COLORS["protenix"], linestyle="--", linewidth=1.6)
    axis.text(
        0.015,
        protenix + 0.003,
        f"Protenix-v2 single-seq · R = {protenix:.3f}",
        transform=axis.get_yaxis_transform(),
        color=COLORS["protenix"],
        fontsize=8.5,
        va="bottom",
    )
    for row in table.itertuples(index=False):
        color = COLORS[row.evaluation]
        axis.scatter(
            float(row.loss),
            float(row.r_all),
            s=90,
            marker="o" if row.evaluation == "computed_here" else "s",
            facecolor=color if row.evaluation == "computed_here" else "white",
            edgecolor=color,
            linewidth=1.8,
            zorder=4,
        )
        offset, alignment, label = ANNOTATIONS[row.key]
        axis.annotate(
            label,
            (float(row.loss), float(row.r_all)),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=8,
            color="#4f4c48",
            arrowprops={
                "arrowstyle": "-",
                "color": "#aaa7a1",
                "linewidth": 0.8,
                "shrinkA": 3,
                "shrinkB": 6,
            },
        )
    axis.text(
        0.28,
        0.05,
        (
            "1.5B sigmoid (descriptive)\n"
            f"R = {parameters[0]:.3f} / [1 + exp((loss − {parameters[1]:.3f}) / "
            f"{parameters[2]:.3f})]\n"
            f"R² = {float(fit['r_squared']):.3f}"
        ),
        transform=axis.transAxes,
        fontsize=8,
        linespacing=1.4,
        color=COLORS["fit"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    axis.set_xlim(3.10, 2.92)
    axis.set_ylim(0.48, 0.67)
    axis.set_xlabel("contacts-v1 validation loss (lower is better →)")
    axis.set_ylabel("Mean all-range R-precision")
    axis.set_title("B · Validation loss and mean R-precision", pad=13)
    axis.grid(color="#d8d7d2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=COLORS["computed_here"],
                markeredgecolor=COLORS["computed_here"],
                markersize=7,
                label="exp232 decontaminated checkpoints",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor="white",
                markeredgecolor=COLORS["previous"],
                markeredgewidth=1.7,
                markersize=7,
                label="previous evaluation",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["protenix"],
                linestyle="--",
                linewidth=1.6,
                label="Protenix-v2 baseline",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["fit"],
                linewidth=1.8,
                label="1.5B sigmoid fit",
            ),
        ],
        loc="lower right",
        fontsize=7.7,
        frameon=True,
        framealpha=0.92,
    )
    return fit


def run(output: Path) -> None:
    """Render the primary PR figure and provenance metadata."""

    table, values = load_values()
    figure, (box_axis, scatter_axis) = plt.subplots(1, 2, figsize=(17.2, 6.8))
    draw_boxplot(box_axis, values)
    fit = draw_scatter(scatter_axis, table, values)
    figure.suptitle(
        "Exp232 decontaminated contacts-v1 checkpoints", fontsize=14, y=0.973
    )
    figure.text(
        0.5,
        0.055,
        (
            "Orange marks the two checkpoints evaluated here; gray shows prior "
            "MarinFold evaluations and blue shows the Protenix-v2 reference."
        ),
        ha="center",
        fontsize=8.4,
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.955), w_pad=2.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
    metadata = {
        "schema_version": 1,
        "figure": "final_checkpoint_rprecision",
        "metric": "legacy-554 all-range R-precision",
        "box_order": list(BOX_ORDER),
        "computed_here": table.loc[table.evaluation == "computed_here", "key"].tolist(),
        "previous_evaluations": table.loc[
            table.evaluation == "previous", "key"
        ].tolist(),
        "sigmoid_fit": fit,
        "comparison_table": str(TABLE.relative_to(HERE.parents[3])),
        "comparison_table_sha256": sha256(TABLE),
        "per_protein_rows": str(ROWS.relative_to(HERE.parents[3])),
        "per_protein_rows_sha256": sha256(ROWS),
        "plot_sha256": sha256(output),
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output}")
    print(f"wrote {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse output arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().output)
