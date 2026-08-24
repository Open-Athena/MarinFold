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
DECONTAM_KEYS = {
    "exp232-m1-p02-decontam",
    "exp232-m2-p06-decontam",
    "exp232-m2-p06-training",
}
BOX_LABELS = {
    "exp75-reproduced": "#75 E8\nvalidation",
    "exp146": "#146 3B",
    "exp166": "#166\nAA aug",
    "cw-p06-aug": "#199 m1-p06\naug",
    "cw-p06-cool": "#199 m1-p06\ncooldown",
    "protenix": "Protenix-v2\nsingle-seq",
    "exp232-m2-p06-decontam": "#232 m2-p06\nsweep",
    "exp232-m1-p02-decontam": "#232 m1-p02\nsweep",
    "exp232-m2-p06-training": "#232 m2-p06\ntraining",
}
COLORS = {
    "previous": "#8f8b86",
    "computed_here": "#d55e00",
    "decontam_previous": "#e69f61",
    "validation": "#009e73",
    "protenix": "#2a78d6",
    "fit": "#52514e",
}
ANNOTATIONS = {
    "exp75-reproduced": ((-3, 56), "left", "#75 E8\nvalidation"),
    "exp146": ((-8, 38), "right", "#146 · 3B"),
    "exp166": ((8, 29), "left", "#166 AA aug"),
    "cw-p06-aug": ((-10, 20), "right", "#199 m1-p06\naug"),
    "cw-p06-cool": ((-6, 22), "right", "#199 m1-p06\ncooldown"),
    "exp232-m2-p06-decontam": ((7, -36), "left", "#232 m2-p06\nsweep"),
    "exp232-m1-p02-decontam": ((-8, -48), "left", "#232 m1-p02\nsweep"),
    "exp232-m2-p06-training": ((10, -28), "left", "#232 m2-p06\ntraining"),
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


def load_values() -> tuple[pd.DataFrame, dict[str, np.ndarray], tuple[str, ...]]:
    """Load inputs and validate every per-protein mean."""

    table = pd.read_csv(TABLE)
    rows = pd.read_csv(ROWS)
    expected_keys = {*table.key, "protenix"}
    if set(rows.key) != expected_keys:
        raise ValueError(f"unexpected per-protein keys: {sorted(rows.key.unique())}")
    values = {
        key: rows.loc[rows.key == key, "precision"].to_numpy(dtype=float)
        for key in expected_keys
    }
    expected = table.set_index("key").r_all.to_dict()
    expected["protenix"] = 0.6031578401726864
    for key, key_values in values.items():
        if key_values.size != 554:
            raise ValueError(f"{key} has {key_values.size} values; expected 554")
        if not np.isclose(key_values.mean(), expected[key], atol=1e-14):
            raise ValueError(f"{key} mean does not match the comparison table")
    box_order = tuple(sorted(values, key=lambda key: values[key].mean()))
    return table, values, box_order


def shifted_sigmoid(
    data: tuple[np.ndarray, np.ndarray],
    upper: float,
    historical_midpoint: float,
    width: float,
    decontam_loss_shift: float,
) -> np.ndarray:
    """Evaluate two shared-shape sigmoids separated by a loss-axis shift."""

    loss, is_decontam = data
    midpoint = historical_midpoint + decontam_loss_shift * is_decontam
    return sigmoid(loss, upper, midpoint, width)


def fit_shared_shape_sigmoids(table: pd.DataFrame) -> dict[str, object]:
    """Jointly fit shared shape with one decontaminated loss-axis shift."""

    selected = table[np.isfinite(table.loss)]
    if set(selected.loc[selected.key.isin(DECONTAM_KEYS), "key"]) != DECONTAM_KEYS:
        raise ValueError(
            "the shared-shape fit requires all three decontaminated points"
        )
    x = selected.loss.to_numpy(dtype=float)
    y = selected.r_all.to_numpy(dtype=float)
    is_decontam = selected.key.isin(DECONTAM_KEYS).to_numpy(dtype=float)
    parameters, _ = curve_fit(
        shifted_sigmoid,
        (x, is_decontam),
        y,
        p0=(min(0.95, float(y.max()) + 0.03), 3.1, 0.05, -0.01),
        bounds=([0.35, 2.5, 0.001, -0.10], [1.0, 3.5, 2.0, 0.10]),
        maxfev=100_000,
    )
    predicted = shifted_sigmoid((x, is_decontam), *parameters)
    residual_sum = float(np.square(y - predicted).sum())
    total_sum = float(np.square(y - y.mean()).sum())
    historical_keys = selected.loc[~selected.key.isin(DECONTAM_KEYS), "key"].tolist()
    decontam_keys = selected.loc[selected.key.isin(DECONTAM_KEYS), "key"].tolist()
    return {
        "upper": float(parameters[0]),
        "historical_midpoint": float(parameters[1]),
        "decontam_midpoint": float(parameters[1] + parameters[3]),
        "width": float(parameters[2]),
        "decontam_loss_shift": float(parameters[3]),
        "r_squared": 1.0 - residual_sum / total_sum,
        "historical_keys": historical_keys,
        "decontam_keys": decontam_keys,
        "minimum_observed_loss": float(x.min()),
        "maximum_observed_loss": float(x.max()),
    }


def draw_boxplot(
    axis: plt.Axes,
    table: pd.DataFrame,
    values: dict[str, np.ndarray],
    box_order: tuple[str, ...],
) -> None:
    """Draw legacy-554 per-protein distributions."""

    boxes = axis.boxplot(
        [values[key] for key in box_order],
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
    evaluations = table.set_index("key").evaluation.to_dict()
    for patch, key in zip(boxes["boxes"], box_order, strict=True):
        if key == "protenix":
            color = COLORS["protenix"]
        elif key in DECONTAM_KEYS and evaluations[key] == "previous":
            color = COLORS["decontam_previous"]
        else:
            color = COLORS[str(evaluations[key])]
        patch.set_facecolor(color)
        patch.set_alpha(0.83)
    for position, key in enumerate(box_order, start=1):
        axis.text(
            position,
            1.018,
            f"{values[key].mean():.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(
        np.arange(1, len(box_order) + 1),
        [BOX_LABELS[key] for key in box_order],
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

    fit = fit_shared_shape_sigmoids(table)
    historical_parameters = (
        float(fit["upper"]),
        float(fit["historical_midpoint"]),
        float(fit["width"]),
    )
    decontam_parameters = (
        float(fit["upper"]),
        float(fit["decontam_midpoint"]),
        float(fit["width"]),
    )
    observed = np.linspace(
        float(fit["maximum_observed_loss"]),
        float(fit["minimum_observed_loss"]),
        240,
    )
    extrapolation_limit = float(fit["minimum_observed_loss"]) - 0.02
    extrapolated = np.linspace(
        float(fit["minimum_observed_loss"]), extrapolation_limit, 80
    )
    axis.plot(
        observed,
        sigmoid(observed, *historical_parameters),
        color=COLORS["fit"],
        linewidth=1.8,
    )
    axis.plot(
        observed,
        sigmoid(observed, *decontam_parameters),
        color=COLORS["computed_here"],
        linewidth=1.8,
    )
    axis.plot(
        extrapolated,
        sigmoid(extrapolated, *historical_parameters),
        color=COLORS["fit"],
        linewidth=1.5,
        linestyle=":",
    )
    axis.plot(
        extrapolated,
        sigmoid(extrapolated, *decontam_parameters),
        color=COLORS["computed_here"],
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
        if row.key in DECONTAM_KEYS:
            color = COLORS["computed_here"]
            marker = "o"
            facecolor = color if row.evaluation == "computed_here" else "white"
        else:
            color = COLORS[row.evaluation]
            marker = {"validation": "D", "previous": "s"}[row.evaluation]
            facecolor = color if row.evaluation == "validation" else "white"
        axis.scatter(
            float(row.loss),
            float(row.r_all),
            s=90,
            marker=marker,
            facecolor=facecolor,
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
            "Joint shared-shape sigmoid\n"
            f"upper = {float(fit['upper']):.3f}; width = {float(fit['width']):.3f}\n"
            f"decontam loss shift = {float(fit['decontam_loss_shift']):+.3f}; "
            f"joint R² = {float(fit['r_squared']):.3f}"
        ),
        transform=axis.transAxes,
        fontsize=8,
        linespacing=1.4,
        color=COLORS["fit"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    axis.set_xlim(
        float(fit["maximum_observed_loss"]) + 0.01,
        extrapolation_limit,
    )
    axis.set_ylim(0.40, max(0.67, float(table.r_all.max()) + 0.025))
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
                label="new exp232 training checkpoint",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=COLORS["computed_here"],
                markeredgewidth=1.7,
                markersize=7,
                label="prior #232 decontaminated sweeps",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=COLORS["validation"],
                markeredgecolor=COLORS["validation"],
                markersize=6.5,
                label="#75 validation recomputed here",
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
                label="historical checkpoint",
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
                label="historical shared-shape fit",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["computed_here"],
                linewidth=1.8,
                label="decontaminated shifted fit",
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

    table, values, box_order = load_values()
    figure, (box_axis, scatter_axis) = plt.subplots(1, 2, figsize=(17.2, 6.8))
    draw_boxplot(box_axis, table, values, box_order)
    fit = draw_scatter(scatter_axis, table, values)
    figure.suptitle(
        "Exp232 decontaminated contacts-v1 training checkpoint",
        fontsize=14,
        y=0.973,
    )
    figure.text(
        0.5,
        0.055,
        (
            "Orange marks the #232 decontaminated family (filled circle = new "
            "training checkpoint); green marks the recomputed validation, gray "
            "shows historical MarinFold evaluations, and blue shows Protenix-v2."
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
        "box_order": list(box_order),
        "computed_here": table.loc[table.evaluation == "computed_here", "key"].tolist(),
        "previous_evaluations": table.loc[
            table.evaluation == "previous", "key"
        ].tolist(),
        "shared_shape_sigmoid_fit": fit,
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
