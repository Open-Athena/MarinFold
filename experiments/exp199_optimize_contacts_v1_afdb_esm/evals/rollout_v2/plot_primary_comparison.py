# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot per-protein boxes and the corrected loss/R-precision scatter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_pr_comparison as shared

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROWS = HERE / "data" / "pr_all_r_rows.csv.gz"
DEFAULT_OUTPUT = HERE / "plots" / "final_checkpoint_rprecision.png"
BOX_ORDER = (
    "exp75-historical",
    "exp75-reproduced",
    "exp146",
    "exp166",
    "trc-p03-aug",
    "trc-p03-base",
    "trc-cont",
    "protenix",
    "cw-p06-aug",
)
BOX_LABELS = {
    "exp75-historical": "#75 E8\nexp82",
    "exp75-reproduced": "#75 E8\nreproduced",
    "exp146": "#146 3B\nexp169",
    "exp166": "#166\nAA aug",
    "trc-p03-aug": "TRC p03\naug",
    "trc-p03-base": "TRC p03\nbase",
    "trc-cont": "TRC\ncontinuation",
    "protenix": "Protenix-v2\nsingle-seq",
    "cw-p06-aug": "CoreWeave p06\naug",
}
COLORS = {
    "previous": "#8f8b86",
    "computed_here": "#d55e00",
    "protenix": "#2a78d6",
    "fit": "#52514e",
}
ANNOTATIONS = {
    "exp75": ((8, 16), "left", "#75 E8\nexp82 + reproduced"),
    "exp146": ((8, -36), "left", "#146 · 3B"),
    "exp166": ((10, 20), "left", "#166 AA aug"),
    "trc-p03-aug": ((8, -33), "left", "TRC p03 aug"),
    "trc-p03-base": ((-18, 21), "right", "TRC p03 base"),
    "cw-p06-aug": ((-12, 24), "right", "CoreWeave p06 aug"),
    "trc-cont": ((-10, -31), "right", "TRC continuation"),
}


def load_values(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Load per-protein values and validate every plotted mean."""

    rows = pd.read_csv(ROWS)
    if set(rows.key) != set(BOX_ORDER):
        raise ValueError(f"unexpected per-protein keys: {sorted(rows.key.unique())}")
    values = {
        key: rows.loc[rows.key == key, "precision"].to_numpy(dtype=float)
        for key in BOX_ORDER
    }
    expected = table.set_index("key").r_all.to_dict()
    expected["protenix"] = shared.EXPECTED_PROTENIX["all"][0]
    for key, key_values in values.items():
        if key_values.size != 554:
            raise ValueError(f"{key} has {key_values.size} values; expected 554")
        if not np.isclose(key_values.mean(), expected[key], atol=1e-14):
            raise ValueError(f"{key} mean does not match the comparison table")
    return rows, values


def draw_boxplot(axis: plt.Axes, values: dict[str, np.ndarray]) -> None:
    """Draw per-protein distributions for every comparison result."""

    boxes = axis.boxplot(
        [values[key] for key in BOX_ORDER],
        widths=0.42,
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
    for patch, key in zip(boxes["boxes"], BOX_ORDER, strict=True):
        if key == "protenix":
            color = COLORS["protenix"]
        elif key in {"exp75-historical", "exp146", "exp166"}:
            color = COLORS["previous"]
        else:
            color = COLORS["computed_here"]
        patch.set_facecolor(color)
        patch.set_alpha(0.83)

    for position, key in enumerate(BOX_ORDER, start=1):
        axis.text(
            position,
            1.018,
            f"{values[key].mean():.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )
    axis.set_xticks(
        np.arange(1, len(BOX_ORDER) + 1),
        [BOX_LABELS[key] for key in BOX_ORDER],
        fontsize=7.5,
    )
    axis.set_ylabel("Per-protein all-range R-precision")
    axis.set_ylim(-0.02, 1.095)
    axis.set_title("A · Distributions across 554 proteins", pad=13)
    axis.grid(axis="y", color="#d8d7d2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def annotate_scatter(axis: plt.Axes, key: str, x: float, y: float) -> None:
    """Annotate one unique checkpoint on the scatter panel."""

    offset, alignment, label = ANNOTATIONS[key]
    axis.annotate(
        label,
        (x, y),
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


def draw_scatter(
    axis: plt.Axes,
    table: pd.DataFrame,
    values: dict[str, np.ndarray],
) -> dict[str, object]:
    """Draw mean all-range R-precision against validation loss."""

    fit = shared.fit_ranges(table)["all"]
    parameters = (
        float(fit["upper"]),
        float(fit["midpoint"]),
        float(fit["width"]),
    )
    observed = np.linspace(
        float(fit["maximum_observed_loss"]),
        float(fit["minimum_observed_loss"]),
        240,
    )
    extrapolated = np.linspace(float(fit["minimum_observed_loss"]), 2.94, 80)
    axis.plot(
        observed,
        shared.sigmoid(observed, *parameters),
        color=COLORS["fit"],
        linewidth=1.8,
        zorder=2,
    )
    axis.plot(
        extrapolated,
        shared.sigmoid(extrapolated, *parameters),
        color=COLORS["fit"],
        linewidth=1.5,
        linestyle=":",
        zorder=2,
    )

    protenix = float(values["protenix"].mean())
    axis.axhline(
        protenix,
        color=COLORS["protenix"],
        linestyle="--",
        linewidth=1.6,
        zorder=1,
    )
    axis.text(
        0.015,
        protenix + 0.003,
        f"Protenix-v2 single-seq · R = {protenix:.3f}",
        transform=axis.get_yaxis_transform(),
        color=COLORS["protenix"],
        fontsize=8.5,
        va="bottom",
    )

    exp75_offsets = {"previous": 0.0012, "computed_here": -0.0012}
    for row in table.itertuples(index=False):
        x = float(row.loss_current_scale)
        if row.fit_group == "exp75":
            x += exp75_offsets[row.evaluation]
        color = COLORS[row.evaluation]
        axis.scatter(
            x,
            float(values[row.key].mean()),
            s=90,
            marker="o" if row.evaluation == "computed_here" else "s",
            facecolor="white" if row.loss_scale == "historical" else color,
            edgecolor=color,
            linewidth=1.8,
            zorder=4,
        )

    annotation_rows = {
        "exp75": table[table.fit_group == "exp75"].mean(numeric_only=True),
        "exp146": table.loc[table.key == "exp146"].iloc[0],
        "exp166": table.loc[table.key == "exp166"].iloc[0],
        "trc-p03-aug": table.loc[table.key == "trc-p03-aug"].iloc[0],
        "trc-p03-base": table.loc[table.key == "trc-p03-base"].iloc[0],
        "cw-p06-aug": table.loc[table.key == "cw-p06-aug"].iloc[0],
        "trc-cont": table.loc[table.key == "trc-cont"].iloc[0],
    }
    for key, row in annotation_rows.items():
        draw_key = "exp75" if key == "exp75" else key
        draw_y = (
            float(table.loc[table.fit_group == "exp75", "r_all"].mean())
            if key == "exp75"
            else float(values[key].mean())
        )
        annotate_scatter(
            axis,
            draw_key,
            float(row.loss_current_scale),
            draw_y,
        )

    axis.text(
        0.28,
        0.04,
        (
            "1.5B sigmoid (descriptive)\n"
            f"R = {parameters[0]:.3f} / [1 + exp((loss − {parameters[1]:.3f}) / "
            f"{parameters[2]:.3f})]\n"
            f"R² = {float(fit['r_squared']):.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        fontsize=8,
        linespacing=1.4,
        color=COLORS["fit"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    axis.set_xlim(3.153, 2.94)
    axis.set_ylim(0.402, 0.625)
    loss_ticks = np.asarray([3.15, 3.10, 3.05, 3.00, 2.95])
    axis.set_xticks(
        loss_ticks,
        [
            f"{current:.2f}\n({current - shared.LOSS_OFFSET:.2f})"
            for current in loss_ticks
        ],
    )
    axis.set_xlabel(
        "contacts-v1 validation loss · current scale above, approximate historical "
        "equivalent in parentheses (lower is better →)"
    )
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
                label="computed here via /eval-checkpoint",
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
                label="unique-1.5B sigmoid fit",
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

    table = shared.load_comparison()
    _, values = load_values(table)
    figure, (box_axis, scatter_axis) = plt.subplots(
        1,
        2,
        figsize=(18.4, 6.9),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    draw_boxplot(box_axis, values)
    fit = draw_scatter(scatter_axis, table, values)
    figure.suptitle(
        "Corrected exp199 final-checkpoint contact prediction", fontsize=14, y=0.972
    )
    figure.text(
        0.5,
        0.088,
        (
            "Each box is one 554-protein evaluation: orange was computed here, gray "
            "is previous, and blue is Protenix. Scatter circles are computed here "
            "via /eval-checkpoint; squares are previous results."
        ),
        ha="center",
        fontsize=8.4,
        weight="bold",
    )
    figure.text(
        0.5,
        0.052,
        (
            "Historical losses use current ≈ historical + 0.38171. The fit uses "
            "each unique 1.5B checkpoint once (the mean of the two #75 results); "
            "#146 3B and Protenix are references."
        ),
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.13, 1, 0.955), w_pad=2.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    metadata = {
        "schema_version": 1,
        "figure": "final_checkpoint_rprecision",
        "metric": "all-range R-precision",
        "box_order": list(BOX_ORDER),
        "computed_here": table.loc[table.evaluation == "computed_here", "key"].tolist(),
        "previous_evaluations": table.loc[
            table.evaluation == "previous", "key"
        ].tolist(),
        "exp75_fit_treatment": (
            "Both evaluations are plotted; their mean enters the fit once."
        ),
        "sigmoid_fit": fit,
        "comparison_table": str(shared.TABLE.relative_to(shared.REPO_ROOT)),
        "comparison_table_sha256": shared.sha256(shared.TABLE),
        "per_protein_rows": str(ROWS.relative_to(shared.REPO_ROOT)),
        "per_protein_rows_sha256": shared.sha256(ROWS),
        "plot_sha256": shared.sha256(output),
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote {output}")
    print(f"wrote {metadata_path}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.output)
