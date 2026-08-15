# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot corrected exp199 rollout R-precision with historical context."""

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
REPO_ROOT = HERE.parents[3]
TABLE = HERE / "data" / "pr_comparison.csv"
PROTENIX_ROWS = (
    REPO_ROOT
    / "experiments"
    / "exp89_evals_contacts_v1_model_on_eval_set"
    / "data"
    / "contact_precision_all.csv"
)
DEFAULT_OUTPUT = HERE / "plots" / "rprecision_ranges_vs_loss.png"

LOSS_OFFSET = 0.38171
LOSS_CONVERSION = "current ~= historical + 0.38171"
RANGES = ("short", "medium", "all", "long")
RANGE_LABELS = {
    "short": "Short",
    "medium": "Medium",
    "all": "All",
    "long": "Long",
}
RANGE_COLORS = {
    "short": "#009e73",
    "medium": "#e69f00",
    "all": "#0072b2",
    "long": "#cc79a7",
}
EXPECTED_KEYS = {
    "exp75-historical",
    "exp75-reproduced",
    "exp146",
    "exp166",
    "trc-p03-aug",
    "trc-p03-base",
    "cw-p06-aug",
    "cw-p06-cool",
    "trc-cont",
}
EXPECTED_PROTENIX = {
    "short": (0.6454019764795024, 554),
    "medium": (0.6278108211907913, 553),
    "all": (0.6031578401726864, 554),
    "long": (0.571710332452209, 553),
}
ANNOTATIONS = {
    "exp75": ((8, -2), "left", "#75 E8 · exp82 + reproduced"),
    "exp146": ((8, -18), "left", "#146 · 3B (not fit)"),
    "exp166": ((-10, -24), "right", "#166 AA aug"),
    "trc-p03-aug": ((10, -20), "left", "TRC p03 aug"),
    "trc-p03-base": ((-10, 18), "right", "TRC p03 base"),
    "cw-p06-aug": ((-12, 22), "right", "CoreWeave p06 aug"),
    "trc-cont": ((-10, -26), "right", "TRC continuation"),
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


def load_comparison() -> pd.DataFrame:
    """Load and validate the curated PR comparison table."""

    table = pd.read_csv(TABLE)
    if set(table.key) != EXPECTED_KEYS:
        raise ValueError(f"unexpected comparison keys: {sorted(table.key)}")
    if (table.evaluation == "computed_here").sum() != 6:
        raise ValueError("expected six /eval-checkpoint results computed here")
    if (table.evaluation == "previous").sum() != 3:
        raise ValueError("expected three rows from previous evaluations")
    if (table.fit_group == "exp75").sum() != 2:
        raise ValueError("the exp75 fit group must contain both evaluations")

    metric_columns = [f"r_{range_name}" for range_name in RANGES]
    if not table[metric_columns].map(lambda value: 0 <= value <= 1).all().all():
        raise ValueError("R-precision values must lie in [0, 1]")
    historical = table[table.loss_scale == "historical"]
    converted = historical.loss_raw + LOSS_OFFSET
    if not np.allclose(converted, historical.loss_current_scale, atol=1e-12):
        raise ValueError(
            "historical loss conversion does not match the declared offset"
        )

    exp75 = table[table.fit_group == "exp75"].set_index("evaluation")
    differences = (
        exp75.loc["computed_here", metric_columns]
        - exp75.loc["previous", metric_columns]
    ).abs()
    if not (differences <= 0.005).all():
        raise ValueError("the reproduced exp75 ranges do not match exp82 within 0.005")
    return table


def load_protenix() -> dict[str, float]:
    """Load Protenix-v2 single-sequence structure references from exp89."""

    rows = pd.read_csv(PROTENIX_ROWS)
    selected = rows[
        (rows.model == "protenix-v2")
        & (rows["mode"] == "single_seq")
        & (rows.predictor == "structure")
        & (rows.cut == "R")
    ]
    references: dict[str, float] = {}
    for range_name, (expected_mean, expected_count) in EXPECTED_PROTENIX.items():
        values = selected.loc[selected.range == range_name, "precision"].dropna()
        if len(values) != expected_count:
            raise ValueError(
                f"Protenix {range_name} has {len(values)} rows; expected {expected_count}"
            )
        references[range_name] = float(values.mean())
        if not np.isclose(references[range_name], expected_mean, atol=1e-12):
            raise ValueError(f"Protenix {range_name} reference changed")
    return references


def fit_ranges(table: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Fit each range after averaging duplicate evaluations by checkpoint."""

    included = table[table.fit_group.notna()]
    fits: dict[str, dict[str, object]] = {}
    for range_name in RANGES:
        grouped = (
            included.groupby("fit_group", sort=False)
            .agg(x=("loss_current_scale", "first"), y=(f"r_{range_name}", "mean"))
            .reset_index()
        )
        x = grouped.x.to_numpy(dtype=float)
        y = grouped.y.to_numpy(dtype=float)
        parameters, _ = curve_fit(
            sigmoid,
            x,
            y,
            p0=(min(0.95, float(y.max()) + 0.03), 3.18, 0.05),
            bounds=([0.35, 2.5, 0.001], [1.0, 3.5, 2.0]),
            maxfev=100_000,
        )
        predicted = sigmoid(x, *parameters)
        residual_sum = float(np.square(y - predicted).sum())
        total_sum = float(np.square(y - y.mean()).sum())
        fits[range_name] = {
            "upper": float(parameters[0]),
            "midpoint": float(parameters[1]),
            "width": float(parameters[2]),
            "r_squared": 1.0 - residual_sum / total_sum,
            "input_groups": grouped.fit_group.tolist(),
            "minimum_observed_loss": float(x.min()),
            "maximum_observed_loss": float(x.max()),
        }
    return fits


def plot_reference_lines(axis: plt.Axes, references: dict[str, float]) -> None:
    """Add Protenix-v2 range references."""

    for range_name in RANGES:
        value = references[range_name]
        axis.axhline(
            value,
            color=RANGE_COLORS[range_name],
            linewidth=0.9,
            linestyle="--",
            alpha=0.45,
            zorder=0,
        )
        axis.text(
            0.992,
            value + 0.0025,
            f"{RANGE_LABELS[range_name]} {value:.3f}",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=RANGE_COLORS[range_name],
            fontsize=7.7,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 0.5},
        )
    axis.text(
        0.992,
        0.692,
        "Protenix-v2 single-seq references",
        transform=axis.get_yaxis_transform(),
        ha="right",
        va="top",
        color="#56534f",
        fontsize=8,
        weight="bold",
    )


def plot_points(axis: plt.Axes, table: pd.DataFrame) -> None:
    """Plot every evaluation row, including both exp75 measurements."""

    exp75_offsets = {"previous": 0.0012, "computed_here": -0.0012}
    for row in table.itertuples(index=False):
        x = float(row.loss_current_scale)
        if row.fit_group == "exp75":
            x += exp75_offsets[row.evaluation]
        marker = "o" if row.evaluation == "computed_here" else "s"
        for range_name in RANGES:
            color = RANGE_COLORS[range_name]
            axis.scatter(
                x,
                float(getattr(row, f"r_{range_name}")),
                s=57,
                marker=marker,
                facecolor="white" if row.loss_scale == "historical" else color,
                edgecolor=color,
                linewidth=1.45,
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
        offset, alignment, label = ANNOTATIONS[key]
        axis.annotate(
            label,
            (float(row.loss_current_scale), float(row.r_all)),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=8,
            color="#4f4c48",
            arrowprops={
                "arrowstyle": "-",
                "color": "#aaa7a1",
                "linewidth": 0.75,
                "shrinkA": 3,
                "shrinkB": 5,
            },
        )


def plot_figure(
    table: pd.DataFrame,
    references: dict[str, float],
    fits: dict[str, dict[str, object]],
    output: Path,
) -> None:
    """Render the comparison figure."""

    figure, axis = plt.subplots(figsize=(12.4, 7.2))
    for range_name in RANGES:
        fit = fits[range_name]
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
            sigmoid(observed, *parameters),
            color=RANGE_COLORS[range_name],
            linewidth=2,
            zorder=2,
        )
        axis.plot(
            extrapolated,
            sigmoid(extrapolated, *parameters),
            color=RANGE_COLORS[range_name],
            linewidth=1.5,
            linestyle=":",
            zorder=2,
        )

    plot_reference_lines(axis, references)
    plot_points(axis, table)

    loss_ticks = np.asarray([3.15, 3.10, 3.05, 3.00, 2.95])
    axis.set_xticks(
        loss_ticks,
        [f"{current:.2f}\n({current - LOSS_OFFSET:.2f})" for current in loss_ticks],
    )
    axis.set_xlim(3.153, 2.94)
    axis.set_ylim(0.34, 0.70)
    axis.set_xlabel(
        "contacts-v1 validation loss · current scale above, approximate historical "
        "equivalent in parentheses (lower is better →)"
    )
    axis.set_ylabel("Mean R-precision")
    axis.set_title("Corrected exp199 contact-range R-precision", pad=14)
    axis.grid(color="#d8d7d2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    range_legend = axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=RANGE_COLORS[range_name],
                marker="o",
                markersize=6,
                linewidth=2,
                label=RANGE_LABELS[range_name],
            )
            for range_name in RANGES
        ],
        loc="lower right",
        ncol=4,
        fontsize=8.3,
        frameon=True,
        framealpha=0.94,
    )
    axis.add_artist(range_legend)
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#66625e",
                marker="o",
                markerfacecolor="white",
                linestyle="none",
                markersize=6,
                label="computed here via /eval-checkpoint",
            ),
            Line2D(
                [0],
                [0],
                color="#66625e",
                marker="s",
                markerfacecolor="white",
                linestyle="none",
                markersize=6,
                label="previous evaluation",
            ),
        ],
        loc="upper left",
        fontsize=8.2,
        frameon=True,
        framealpha=0.94,
    )
    figure.text(
        0.5,
        0.047,
        (
            "Solid curves fit unique 1.5B checkpoints; the two #75 evaluations are "
            "shown separately but enter each fit once via their mean. Dotted curves "
            "extrapolate; dashed lines are Protenix-v2 references."
        ),
        ha="center",
        fontsize=8,
        weight="bold",
    )
    figure.text(
        0.5,
        0.019,
        (
            "Filled points use current-scale losses. Hollow points use historical "
            "losses converted with current ≈ historical + 0.38171. #146 is 3B and "
            "is shown but excluded from the fits."
        ),
        ha="center",
        fontsize=7.9,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def run(output: Path) -> None:
    """Validate data, render the plot, and record provenance metadata."""

    table = load_comparison()
    references = load_protenix()
    fits = fit_ranges(table)
    plot_figure(table, references, fits, output)

    metadata = {
        "schema_version": 1,
        "figure": "rprecision_ranges_vs_loss",
        "ranges": list(RANGES),
        "loss_conversion": LOSS_CONVERSION,
        "computed_here": table.loc[table.evaluation == "computed_here", "key"].tolist(),
        "previous_evaluations": table.loc[
            table.evaluation == "previous", "key"
        ].tolist(),
        "exp75_fit_treatment": (
            "Both evaluations are plotted; their per-range mean enters each fit once."
        ),
        "protenix_v2_single_seq": references,
        "fits": fits,
        "wandb_run_ids": dict(zip(table.display_name, table.wandb_run_id, strict=True)),
        "comparison_rows": json.loads(table.to_json(orient="records")),
        "table": str(TABLE.relative_to(REPO_ROOT)),
        "table_sha256": sha256(TABLE),
        "protenix_source": str(PROTENIX_ROWS.relative_to(REPO_ROOT)),
        "protenix_source_sha256": sha256(PROTENIX_ROWS),
        "plot_sha256": sha256(output),
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
