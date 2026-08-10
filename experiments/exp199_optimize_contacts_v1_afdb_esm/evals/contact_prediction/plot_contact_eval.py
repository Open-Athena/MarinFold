"""Build the exp199 final-checkpoint comparison from published result rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.special import expit

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent.parent
REPO_ROOT = EXPERIMENT.parent.parent
TABLE = EXPERIMENT / "data" / "contact_eval_pr_comparison_summary.csv"
DEFAULT_SCRATCH = REPO_ROOT / "scratch" / "exp199-pr-figure"
DEFAULT_OUTPUT = EXPERIMENT / "plots" / "final_checkpoint_rprecision.png"

COLORS = {
    "historical": "#8f8b86",
    "control": "#4f8178",
    "exp199": "#d55e00",
    "baseline": "#2a78d6",
}
MODELS = {
    "exp75": "marinfold-cv1-exp75-rollout",
    "exp146": "exp146_3b_e8_step17839",
    "control-r0": "exp117_control_step35679",
    "control-r1": "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4-step-35679",
    "control-r2": "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4-step-35679",
    "control-r3": "prot-exp117-cv1-s02-1_5b-e16-lr3p162e-3-wd0p2-bs256-europe-west4-step-35679",
    "exp166": "exp166_aaaug_step35679",
    "trc-p06-aug": "prot-exp199-cv1-s01-m1-p06-aug-us-east1-step-72599",
    "trc-p03-aug": "prot-exp199-cv1-s01-m1-p03-aug-us-east1-step-72599",
    "trc-p03-base": "prot-exp199-cv1-s01-m1-p03-base-us-east5-step-72599",
    "cw-p06-aug": "prot-exp199-cw-cv1-s02-m1-p06-aug-step-145199",
    "protenix": "protenix-v2",
}
BOX_LABELS = {
    "exp75": "#75\nhistorical",
    "exp146": "#146 3B\nhistorical",
    "control-r0": "r0\n#190",
    "control-r1": "r1",
    "control-r2": "r2",
    "control-r3": "r3",
    "exp166": "#166\nhistorical",
    "trc-p06-aug": "TRC p06\naug",
    "trc-p03-aug": "TRC p03\naug",
    "trc-p03-base": "TRC p03\nbase",
    "cw-p06-aug": "CW p06\naug",
    "protenix": "Protenix-v2\nsingle-seq",
}
MARKERS = {
    "exp75": "o",
    "exp146": "s",
    "control-r0": "o",
    "control-r1": "s",
    "control-r2": "D",
    "control-r3": "^",
    "exp166": "^",
    "trc-p06-aug": "v",
    "trc-p03-aug": "o",
    "trc-p03-base": "s",
    "cw-p06-aug": "D",
    "protenix": "X",
}
ANNOTATIONS = {
    "exp75": ((9, 15), "left"),
    "exp146": ((8, -38), "left"),
    "exp166": ((12, 18), "left"),
    "trc-p06-aug": ((12, -30), "left"),
    "trc-p03-aug": ((8, -34), "left"),
    "trc-p03-base": ((-20, 21), "right"),
    "cw-p06-aug": ((0, -31), "center"),
}

LOSS_CONVERSION = {
    "method": "offset",
    "equation": "current ~= old + 0.38171",
    "reason": (
        "The four-point empirical study found the offset more stable than its "
        "fitted slope over the narrow observed loss range."
    ),
    "gist": "https://gist.github.com/eric-czech/9c40252457790a513eeb62a6a965c049",
    "issue": (
        "https://github.com/Open-Athena/MarinFold/issues/173#issuecomment-5227639661"
    ),
    "discord": (
        "https://discord.com/channels/1354881461060243556/"
        "1533900986446385202/1535720900165369906"
    ),
}
LOSS_OFFSET = 0.38171


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_source(row: pd.Series, scratch: Path) -> Path:
    source = str(row.source)
    if source.startswith("https://"):
        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / f"{row.key}.csv.gz"
        if not path.exists():
            partial = path.with_suffix(path.suffix + ".part")
            with httpx.stream(
                "GET", source, follow_redirects=True, timeout=180
            ) as response:
                response.raise_for_status()
                with partial.open("wb") as destination:
                    for chunk in response.iter_bytes(1024 * 1024):
                        destination.write(chunk)
            partial.replace(path)
    else:
        path = REPO_ROOT / source
    observed = sha256(path)
    if observed != row.source_sha256:
        raise ValueError(
            f"{row.key} checksum mismatch: expected {row.source_sha256}, got {observed}"
        )
    return path


def load_r_values(row: pd.Series, scratch: Path) -> np.ndarray:
    frame = pd.read_csv(resolve_source(row, scratch))
    selected = frame[
        (frame["model"] == MODELS[row.key])
        & (frame["range"] == "all")
        & (frame["cut"] == "R")
    ]
    if "mode" in selected:
        selected = selected[selected["mode"] == "single_seq"]
    if "predictor" in selected:
        predictor = "structure" if row.category == "baseline" else "lm"
        selected = selected[selected["predictor"] == predictor]
    values = selected.precision.to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size != row.n_proteins:
        raise ValueError(
            f"{row.key} has {values.size} all-range R values, expected {row.n_proteins}"
        )
    if not np.isclose(values.mean(), row.r_all, rtol=0, atol=1e-14):
        raise ValueError(f"{row.key} mean does not match the comparison table")
    return values


def ordered_boxes(
    table: pd.DataFrame, values: dict[str, np.ndarray]
) -> list[pd.Series]:
    controls = table[table.category == "control"].key.tolist()
    control_mean = float(np.mean([values[key].mean() for key in controls]))

    def order(row: pd.Series) -> tuple[float, int]:
        if row.category == "control":
            return control_mean, controls.index(row.key)
        return float(values[row.key].mean()), 0

    return sorted((row for _, row in table.iterrows()), key=order)


def box_positions(rows: list[pd.Series]) -> np.ndarray:
    positions = []
    cursor = 1.0
    in_controls = False
    for row in rows:
        if row.category == "control":
            if in_controls:
                cursor += 0.28
            positions.append(cursor)
            in_controls = True
            continue
        if in_controls:
            cursor += 0.75
            in_controls = False
        positions.append(cursor)
        cursor += 1.0
    return np.asarray(positions)


def draw_boxplot(
    axis: plt.Axes, rows: list[pd.Series], values: dict[str, np.ndarray]
) -> None:
    positions = box_positions(rows)
    widths = [0.14 if row.category == "control" else 0.34 for row in rows]
    boxes = axis.boxplot(
        [values[row.key] for row in rows],
        positions=positions,
        widths=widths,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#111111",
            "markersize": 4.5,
        },
        medianprops={"color": "#111111", "linewidth": 1.3},
        flierprops={"marker": ".", "markersize": 2.2, "alpha": 0.25},
    )
    for patch, row in zip(boxes["boxes"], rows, strict=True):
        patch.set_facecolor(COLORS[row.category])
        patch.set_alpha(0.84)

    control_positions = [
        position
        for position, row in zip(positions, rows, strict=True)
        if row.category == "control"
    ]
    axis.axvspan(
        min(control_positions) - 0.18,
        max(control_positions) + 0.18,
        color=COLORS["control"],
        alpha=0.08,
        zorder=0,
    )
    axis.text(
        float(np.mean(control_positions)),
        1.055,
        (
            "same #117 checkpoint · four evals\n"
            + " · ".join(
                f"{values[row.key].mean():.4f}"
                for row in rows
                if row.category == "control"
            )
        ),
        ha="center",
        va="bottom",
        fontsize=7.7,
        color="#335950",
        weight="bold",
    )
    for position, row in zip(positions, rows, strict=True):
        if row.category == "control":
            continue
        axis.text(
            position,
            1.018,
            f"{values[row.key].mean():.3f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )
    axis.set_xticks(positions, [BOX_LABELS[row.key] for row in rows], fontsize=7.7)
    axis.set_ylabel("Per-protein all-range R-precision")
    axis.set_ylim(-0.02, 1.105)
    axis.set_title("A · Distributions across 554 proteins", pad=13)
    axis.grid(axis="y", color="#d8d7d2", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def sigmoid(
    loss: np.ndarray, upper: float, midpoint: float, width: float
) -> np.ndarray:
    return upper * expit((midpoint - loss) / width)


def fit_sigmoid(
    table: pd.DataFrame, values: dict[str, np.ndarray]
) -> dict[str, object]:
    fit_rows = table[(~table.category.isin(["baseline"])) & (table.key != "exp146")]
    controls = fit_rows[fit_rows.category == "control"]
    xs = [float(controls.iloc[0].loss_current_scale)]
    ys = [float(np.mean([values[key].mean() for key in controls.key]))]
    input_keys = ["control-mean"]
    for _, row in fit_rows[fit_rows.category != "control"].iterrows():
        xs.append(float(row.loss_current_scale))
        ys.append(float(values[row.key].mean()))
        input_keys.append(row.key)
    fit_x = np.asarray(xs)
    fit_y = np.asarray(ys)
    parameters, _ = curve_fit(
        sigmoid,
        fit_x,
        fit_y,
        p0=(0.60, 3.18, 0.05),
        bounds=([0.5, 2.5, 0.001], [1.0, 3.5, 2.0]),
        maxfev=100_000,
    )
    upper, midpoint, width = (float(value) for value in parameters)
    predicted = sigmoid(fit_x, upper, midpoint, width)
    residual_sum = float(np.square(fit_y - predicted).sum())
    total_sum = float(np.square(fit_y - fit_y.mean()).sum())
    return {
        "equation": "R = upper / (1 + exp((loss - midpoint) / width))",
        "upper": upper,
        "midpoint": midpoint,
        "width": width,
        "r_squared": 1.0 - residual_sum / total_sum,
        "rmse": float(np.sqrt(residual_sum / fit_y.size)),
        "input_keys": input_keys,
        "minimum_observed_loss": float(fit_x.min()),
        "maximum_observed_loss": float(fit_x.max()),
        "control_replicates_enter_fit_as_mean": True,
        "excluded": ["exp146", "protenix"],
    }


def annotate(
    axis: plt.Axes,
    row: pd.Series,
    x: float,
    y: float,
) -> None:
    offset, alignment = ANNOTATIONS[row.key]
    label = row.display_name
    if row.category == "exp199":
        label = f"{label}\nloss {x:.4f} · R {y:.3f}"
    axis.annotate(
        label,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        ha=alignment,
        va="center",
        fontsize=8.2,
        arrowprops={
            "arrowstyle": "-",
            "color": "#aaa7a1",
            "linewidth": 0.9,
            "shrinkA": 3,
            "shrinkB": 7,
        },
    )


def draw_scatter(
    axis: plt.Axes, table: pd.DataFrame, values: dict[str, np.ndarray]
) -> dict[str, object]:
    fit = fit_sigmoid(table, values)
    upper = float(fit["upper"])
    midpoint = float(fit["midpoint"])
    width = float(fit["width"])
    observed_grid = np.linspace(
        float(fit["minimum_observed_loss"]),
        float(fit["maximum_observed_loss"]),
        240,
    )
    extrapolated_grid = np.linspace(2.92, float(fit["minimum_observed_loss"]), 100)
    axis.plot(
        observed_grid,
        sigmoid(observed_grid, upper, midpoint, width),
        color="#52514e",
        linewidth=1.8,
        zorder=2,
    )
    axis.plot(
        extrapolated_grid,
        sigmoid(extrapolated_grid, upper, midpoint, width),
        color="#52514e",
        linewidth=1.5,
        linestyle=":",
        zorder=2,
    )
    baseline_r = float(values["protenix"].mean())
    fit["protenix_r"] = baseline_r
    fit["has_protenix_crossing"] = upper >= baseline_r
    axis.axhline(
        baseline_r,
        color=COLORS["baseline"],
        linestyle="--",
        linewidth=1.6,
        zorder=1,
    )
    axis.text(
        0.015,
        baseline_r + 0.004,
        f"Protenix-v2 single-seq · R = {baseline_r:.3f}",
        transform=axis.get_yaxis_transform(),
        color=COLORS["baseline"],
        fontsize=8.8,
        va="bottom",
    )

    controls = table[table.category == "control"]
    control_x = float(controls.iloc[0].loss_current_scale)
    control_ys = [float(values[row.key].mean()) for _, row in controls.iterrows()]
    axis.vlines(
        control_x,
        min(control_ys) - 0.005,
        max(control_ys) + 0.005,
        color=COLORS["control"],
        linewidth=1,
        linestyle=":",
        alpha=0.8,
        zorder=1,
    )
    dodge = (-0.003, -0.001, 0.001, 0.003)
    for (_, row), x_offset, y in zip(
        controls.iterrows(), dodge, control_ys, strict=True
    ):
        shown_x = control_x + x_offset
        axis.plot(
            [control_x, shown_x],
            [y, y],
            color=COLORS["control"],
            linewidth=0.7,
            alpha=0.7,
            zorder=2,
        )
        axis.scatter(
            shown_x,
            y,
            s=70,
            marker=MARKERS[row.key],
            facecolor="white",
            edgecolor=COLORS["control"],
            linewidth=1.8,
            zorder=4,
        )
    axis.annotate(
        "#117 control\nfour evals",
        (control_x, max(control_ys) + 0.004),
        xytext=(-18, 28),
        textcoords="offset points",
        ha="right",
        fontsize=8.2,
        color="#335950",
        arrowprops={
            "arrowstyle": "-",
            "color": COLORS["control"],
            "linewidth": 0.9,
        },
    )

    points = table[~table.category.isin(["control", "baseline"])]
    for _, row in points.iterrows():
        x = float(row.loss_current_scale)
        y = float(values[row.key].mean())
        historical = row.loss_raw_scale == "historical"
        axis.scatter(
            x,
            y,
            s=92,
            marker=MARKERS[row.key],
            facecolor="white" if historical else COLORS[row.category],
            edgecolor=COLORS[row.category] if historical else "white",
            linewidth=1.8,
            zorder=4,
        )
        annotate(axis, row, x, y)

    axis.text(
        0.31,
        0.04,
        (
            "1.5B sigmoid (descriptive)\n"
            f"R = {upper:.3f} / [1 + exp((loss − {midpoint:.3f}) / {width:.3f})]\n"
            f"R² = {float(fit['r_squared']):.3f} · fitted upper asymptote = {upper:.3f}"
        ),
        transform=axis.transAxes,
        ha="left",
        fontsize=8.1,
        linespacing=1.45,
        color="#52514e",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 3},
    )
    axis.set_xlim(3.153, 2.92)
    axis.set_ylim(0.402, 0.625)
    axis.set_xlabel("contacts-v1 validation loss on current scale (lower is better →)")
    axis.set_ylabel("Mean all-range R-precision")
    old_loss_axis = axis.secondary_xaxis(
        -0.20,
        functions=(
            lambda current: current - LOSS_OFFSET,
            lambda old: old + LOSS_OFFSET,
        ),
    )
    old_loss_axis.set_xlabel(
        "Approximate historical loss scale",
        color="#77746f",
        fontsize=8.2,
        labelpad=4,
    )
    old_loss_axis.tick_params(
        axis="x",
        colors="#77746f",
        labelsize=7.8,
        length=3,
        width=0.7,
        pad=2,
    )
    old_loss_axis.spines["bottom"].set_color("#aaa7a1")
    old_loss_axis.spines["bottom"].set_linewidth(0.7)
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
                markerfacecolor=COLORS["exp199"],
                markeredgecolor="white",
                markersize=8,
                label="exp199 · current loss",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=COLORS["historical"],
                markeredgewidth=1.7,
                markersize=8,
                label="historical loss · converted",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=COLORS["control"],
                markeredgewidth=1.7,
                markersize=8,
                label="#117 eval replicate",
            ),
            Line2D(
                [0],
                [0],
                color=COLORS["baseline"],
                linestyle="--",
                linewidth=1.6,
                label="Protenix-v2 baseline",
            ),
            Line2D(
                [0],
                [0],
                color="#52514e",
                linewidth=1.8,
                label="1.5B sigmoid fit",
            ),
        ],
        loc="lower right",
        fontsize=7.8,
        frameon=True,
        framealpha=0.92,
    )
    return fit


def run(*, output: Path, scratch: Path) -> None:
    table = pd.read_csv(TABLE)
    missing = set(table.key) - set(MODELS)
    if missing:
        raise ValueError(f"missing plot configuration for {sorted(missing)}")
    values = {row.key: load_r_values(row, scratch) for _, row in table.iterrows()}
    rows = ordered_boxes(table, values)

    figure, (box_axis, scatter_axis) = plt.subplots(
        1,
        2,
        figsize=(18.4, 6.9),
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    draw_boxplot(box_axis, rows, values)
    fit = draw_scatter(scatter_axis, table, values)
    figure.suptitle(
        "AFDB/ESM mixing checkpoint contact prediction", fontsize=14, y=0.972
    )
    figure.text(
        0.5,
        0.054,
        (
            "Each box is one 554-protein evaluation. #117 r0 is PR #190; "
            "r1–r3 are fresh repeats. Scatter points are dodged only for visibility; "
            "the dotted curve is extrapolated."
        ),
        ha="center",
        fontsize=8.5,
        weight="bold",
    )
    figure.text(
        0.5,
        0.020,
        (
            "Historical losses for #75, #117, #146, and #166 use the empirical "
            "conversion current ≈ old + 0.38171. The sigmoid uses each unique "
            "1.5B checkpoint once; #146 3B and Protenix are references."
        ),
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.15, 1, 0.955), w_pad=2.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    metadata = {
        "schema_version": 1,
        "figure": "final_checkpoint_rprecision",
        "metric": "all-range R-precision",
        "box_order": [row.key for row in rows],
        "loss_conversion": LOSS_CONVERSION,
        "sigmoid_fit": fit,
        "control_replicates": [
            {
                "key": row.key,
                "evaluation": row.evaluation,
                "r_all": float(values[row.key].mean()),
            }
            for _, row in table[table.category == "control"].iterrows()
        ],
        "table": str(TABLE.relative_to(REPO_ROOT)),
        "table_sha256": sha256(TABLE),
        "plot_sha256": sha256(output),
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix(output.suffix + '.meta.json')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--scratch", type=Path, default=DEFAULT_SCRATCH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(output=args.output, scratch=args.scratch)
