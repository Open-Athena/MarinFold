"""Plot contact-range R-precision against validation loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

import plot_contact_eval as shared

plt.switch_backend("Agg")

DEFAULT_OUTPUT = (
    shared.EXPERIMENT / "plots" / "rprecision_ranges_vs_loss.png"
)
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
ANNOTATIONS = {
    "exp75": ((8, -5), "left"),
    "exp166": ((-8, 18), "right"),
    "trc-p06-aug": ((9, -24), "left"),
    "trc-p03-aug": ((9, -25), "left"),
    "trc-p03-base": ((-10, 22), "right"),
    "cw-p06-aug": ((0, -28), "center"),
}


def select_rows(row: pd.Series, scratch: Path) -> pd.DataFrame:
    frame = pd.read_csv(shared.resolve_source(row, scratch))
    selected = frame[
        (frame["model"] == shared.MODELS[row.key]) & (frame["cut"] == "R")
    ]
    if "mode" in selected:
        selected = selected[selected["mode"] == "single_seq"]
    if "predictor" in selected:
        predictor = "structure" if row.category == "baseline" else "lm"
        selected = selected[selected["predictor"] == predictor]
    return selected


def load_means(
    table: pd.DataFrame, scratch: Path
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    means: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for _, row in table.iterrows():
        selected = select_rows(row, scratch)
        means[row.key] = {}
        counts[row.key] = {}
        for range_name in RANGES:
            range_rows = selected[selected["range"] == range_name]
            if range_rows.empty:
                continue
            if len(range_rows) != row.n_proteins:
                raise ValueError(
                    f"{row.key}/{range_name} has {len(range_rows)} rows, "
                    f"expected {row.n_proteins}"
                )
            values = range_rows.precision.to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            means[row.key][range_name] = float(values.mean())
            counts[row.key][range_name] = int(values.size)
        if not np.isclose(
            means[row.key]["all"], row.r_all, rtol=0, atol=1e-14
        ):
            raise ValueError(f"{row.key} all-range mean does not match the table")
    return means, counts


def fit_range(
    table: pd.DataFrame,
    means: dict[str, dict[str, float]],
    range_name: str,
) -> dict[str, object]:
    controls = table[table.category == "control"]
    xs = [float(controls.iloc[0].loss_current_scale)]
    ys = [float(np.mean([means[key][range_name] for key in controls.key]))]
    keys = ["control-mean"]
    fit_rows = table[
        (~table.category.isin(["control", "baseline"]))
        & (table.key != "exp146")
    ]
    for _, row in fit_rows.iterrows():
        if range_name not in means[row.key]:
            continue
        xs.append(float(row.loss_current_scale))
        ys.append(means[row.key][range_name])
        keys.append(row.key)

    fit_x = np.asarray(xs)
    fit_y = np.asarray(ys)
    parameters, _ = curve_fit(
        shared.sigmoid,
        fit_x,
        fit_y,
        p0=(min(0.95, fit_y.max() + 0.03), 3.18, 0.05),
        bounds=([0.3, 2.5, 0.001], [1.0, 3.5, 2.0]),
        maxfev=100_000,
    )
    predicted = shared.sigmoid(fit_x, *parameters)
    residual_sum = float(np.square(fit_y - predicted).sum())
    total_sum = float(np.square(fit_y - fit_y.mean()).sum())
    return {
        "upper": float(parameters[0]),
        "midpoint": float(parameters[1]),
        "width": float(parameters[2]),
        "r_squared": 1.0 - residual_sum / total_sum,
        "input_keys": keys,
        "minimum_observed_loss": float(fit_x.min()),
        "maximum_observed_loss": float(fit_x.max()),
    }


def plot_reference_lines(
    axis: plt.Axes, means: dict[str, dict[str, float]]
) -> None:
    for range_name in RANGES:
        value = means["protenix"][range_name]
        axis.axhline(
            value,
            color=RANGE_COLORS[range_name],
            linewidth=0.9,
            linestyle="--",
            alpha=0.42,
            zorder=0,
        )
        axis.text(
            0.992,
            value + 0.003,
            f"{RANGE_LABELS[range_name]} {value:.3f}",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=RANGE_COLORS[range_name],
            fontsize=7.8,
        )
    axis.text(
        0.992,
        0.684,
        "Protenix-v2 single-seq references",
        transform=axis.get_yaxis_transform(),
        ha="right",
        va="top",
        color="#56534f",
        fontsize=8.1,
        weight="bold",
    )


def plot_control_replicates(
    axis: plt.Axes,
    table: pd.DataFrame,
    means: dict[str, dict[str, float]],
) -> None:
    controls = table[table.category == "control"]
    control_x = float(controls.iloc[0].loss_current_scale)
    dodge = (-0.003, -0.001, 0.001, 0.003)
    for (_, row), x_offset in zip(controls.iterrows(), dodge, strict=True):
        for range_name in RANGES:
            axis.scatter(
                control_x + x_offset,
                means[row.key][range_name],
                s=35,
                marker=shared.MARKERS[row.key],
                facecolor="white",
                edgecolor=RANGE_COLORS[range_name],
                linewidth=1.15,
                alpha=0.9,
                zorder=5,
            )
    control_all = float(np.mean([means[key]["all"] for key in controls.key]))
    axis.annotate(
        "#117 control · four evals",
        (control_x, control_all),
        xytext=(-10, 20),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=8.3,
        color="#335950",
        arrowprops={
            "arrowstyle": "-",
            "color": "#7c7974",
            "linewidth": 0.8,
        },
    )


def plot_checkpoint_points(
    axis: plt.Axes,
    table: pd.DataFrame,
    means: dict[str, dict[str, float]],
) -> None:
    rows = table[~table.category.isin(["control", "baseline"])]
    for _, row in rows.iterrows():
        x = float(row.loss_current_scale)
        if row.key == "exp146":
            axis.scatter(
                x,
                means[row.key]["all"],
                s=62,
                marker="s",
                facecolor="white",
                edgecolor="#8f8b86",
                linewidth=1.4,
                zorder=4,
            )
            axis.annotate(
                "#146 · 3B\nR-all only",
                (x, means[row.key]["all"]),
                xytext=(-9, -20),
                textcoords="offset points",
                ha="right",
                va="top",
                fontsize=7.7,
                color="#6f6c67",
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#aaa7a1",
                    "linewidth": 0.7,
                },
            )
            continue

        historical = row.loss_raw_scale == "historical"
        for range_name in RANGES:
            axis.scatter(
                x,
                means[row.key][range_name],
                s=54,
                marker="o",
                facecolor="white" if historical else RANGE_COLORS[range_name],
                edgecolor=RANGE_COLORS[range_name],
                linewidth=1.4,
                zorder=4,
            )
        offset, alignment = ANNOTATIONS[row.key]
        label = row.display_name
        if row.category == "exp199":
            label = f"{label}\nloss {x:.4f}"
        axis.annotate(
            label,
            (x, means[row.key]["all"]),
            xytext=offset,
            textcoords="offset points",
            ha=alignment,
            va="center",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-",
                "color": "#aaa7a1",
                "linewidth": 0.8,
                "shrinkA": 3,
                "shrinkB": 5,
            },
        )


def run(*, output: Path, scratch: Path) -> None:
    table = pd.read_csv(shared.TABLE)
    means, counts = load_means(table, scratch)

    figure, axis = plt.subplots(figsize=(12.4, 7.1))
    fits = {}
    for range_name in RANGES:
        fit = fit_range(table, means, range_name)
        fits[range_name] = fit
        observed = np.linspace(
            float(fit["minimum_observed_loss"]),
            float(fit["maximum_observed_loss"]),
            240,
        )
        extrapolated = np.linspace(
            2.92, float(fit["minimum_observed_loss"]), 100
        )
        parameters = (
            float(fit["upper"]),
            float(fit["midpoint"]),
            float(fit["width"]),
        )
        axis.plot(
            observed,
            shared.sigmoid(observed, *parameters),
            color=RANGE_COLORS[range_name],
            linewidth=2.0,
            zorder=2,
        )
        axis.plot(
            extrapolated,
            shared.sigmoid(extrapolated, *parameters),
            color=RANGE_COLORS[range_name],
            linewidth=1.6,
            linestyle=":",
            zorder=2,
        )

    plot_reference_lines(axis, means)
    plot_control_replicates(axis, table, means)
    plot_checkpoint_points(axis, table, means)

    loss_ticks = np.asarray([3.15, 3.10, 3.05, 3.00, 2.95])
    axis.set_xticks(
        loss_ticks,
        [
            f"{current:.2f}\n({current - shared.LOSS_OFFSET:.2f})"
            for current in loss_ticks
        ],
    )
    axis.set_xlim(3.153, 2.92)
    axis.set_ylim(0.34, 0.69)
    axis.set_xlabel(
        "contacts-v1 validation loss · current scale above, approximate historical "
        "equivalent in parentheses (lower is better →)"
    )
    axis.set_ylabel("Mean R-precision")
    axis.set_title("Contact-range R-precision across validation loss", pad=14)
    axis.grid(color="#d8d7d2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
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
        fontsize=8.5,
        frameon=True,
        framealpha=0.92,
    )
    figure.text(
        0.5,
        0.048,
        (
            "Solid curves are descriptive 1.5B fits; dotted portions extrapolate. "
            "Dashed references are Protenix-v2 single-sequence scores."
        ),
        ha="center",
        fontsize=8.2,
        weight="bold",
    )
    figure.text(
        0.5,
        0.020,
        (
            "Filled points use current losses; hollow points use converted historical "
            "losses. Each #117 replicate is shown separately."
        ),
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    metadata = {
        "schema_version": 1,
        "figure": "rprecision_ranges_vs_loss",
        "ranges": list(RANGES),
        "loss_conversion": shared.LOSS_CONVERSION,
        "fits": fits,
        "counts": counts,
        "table": str(shared.TABLE.relative_to(shared.REPO_ROOT)),
        "table_sha256": shared.sha256(shared.TABLE),
        "plot_sha256": shared.sha256(output),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix(output.suffix + '.meta.json')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=shared.DEFAULT_SCRATCH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(output=args.output, scratch=args.scratch)
