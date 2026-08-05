# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot exp117 training dynamics and validation outcomes by learning rate.

The run set comes from the normalized latest-subversion sweep table. Learning
rate, weight decay, and batch size are read from W&B config and checked against
their tags; run names are never parsed. The W&B history sampler limits transfer
volume before each training-loss series is Gaussian-smoothed on a common epoch
grid.

Run from this directory::

    uv run --with wandb --with matplotlib --with numpy --with pandas \
        python plot_exp117_lr_training.py --refresh
"""

import argparse
import csv
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
SOURCE_RUNS_CSV = HERE / "data" / "wandb_runs.csv"
PLOT_DATA_CSV = HERE / "data" / "exp117_lr_training_validation.csv"
PNG_PATH = HERE / "plots" / "exp117_lr_training_validation.png"
SVG_PATH = HERE / "plots" / "exp117_lr_training_validation.svg"
PLOT_DATA_COLUMNS = (
    "run_id",
    "sweep_subversion",
    "model_size",
    "epochs",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "val_loss",
    "val_loss_key",
    "history_samples",
    "smoothing_bandwidth_epochs",
    "training_epoch",
    "train_loss_smoothed",
)

ISSUE = 117
MODEL_SIZE = "1_5b"
EPOCHS = 8
HISTORY_SAMPLES = 1_000
SMOOTHING_BANDWIDTH_EPOCHS = 0.08
SMOOTHING_GRID_POINTS = 401
FETCH_WORKERS = 6
TRAINING_AXIS_MAX = 4.25
VALIDATION_AXIS_MIN = 2.62
VALIDATION_AXIS_MAX = 3.2
CLIPPED_VALIDATION_Y = 3.185


@dataclass(frozen=True)
class SelectedRun:
    """Normalized metadata for one finished exp117 sweep run."""

    project: str
    run_id: str
    run_name: str
    run_url: str
    sweep_subversion: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    val_loss: float
    val_loss_key: str
    total_steps: int
    tags: tuple[str, ...]


def parse_tag_values(tags: Sequence[str]) -> dict[str, str]:
    """Parse key-value W&B tags and reject conflicts."""
    values: dict[str, str] = {}
    for tag in tags:
        if "=" not in tag:
            continue
        key, value = tag.split("=", 1)
        previous = values.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting tag {key!r}: {previous!r} != {value!r}")
        values[key] = value
    return values


def nested_value(config: Mapping[str, Any], path: str) -> Any:
    """Read a required dotted path from a nested W&B config."""
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"missing W&B config field {path!r}")
        value = value[part]
    return value


def assert_close(actual: float, expected: float, label: str, run_name: str) -> None:
    """Require two normalized floating-point metadata values to agree."""
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{run_name}: {label} mismatch: {actual!r} != {expected!r}")


def load_selected_runs(path: Path = SOURCE_RUNS_CSV) -> list[SelectedRun]:
    """Select finished latest-subversion exp117 1.5B eight-epoch runs."""
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    selected_rows = [
        row
        for row in rows
        if int(row["issue"]) == ISSUE
        and row["model_size"] == MODEL_SIZE
        and int(row["epochs"]) == EPOCHS
        and row["state"] == "finished"
    ]
    if not selected_rows:
        raise ValueError(f"no finished exp{ISSUE} {MODEL_SIZE} {EPOCHS}-epoch runs in {path}")

    subversions = {int(row["sweep_subversion"]) for row in selected_rows}
    if len(subversions) != 1:
        raise ValueError(f"expected one latest sweep subversion, found {sorted(subversions)}")
    run_ids = [row["run_id"] for row in selected_rows]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("duplicate W&B run IDs in normalized source table")

    selected: list[SelectedRun] = []
    for row in selected_rows:
        if not row["learning_rate_source"].startswith("config:"):
            raise ValueError(f"{row['run_name']}: learning rate did not come from config")
        if not row["weight_decay_source"].startswith("config:"):
            raise ValueError(f"{row['run_name']}: weight decay did not come from config")
        if not row["batch_size_source"].startswith("config:"):
            raise ValueError(f"{row['run_name']}: batch size did not come from config")

        tags = tuple(str(tag) for tag in json.loads(row["tags_json"]))
        tag_values = parse_tag_values(tags)
        required_tags = {"exp117", "model_size", "epochs", "steps", "lr", "wd", "global_batch"}
        missing = required_tags - (set(tag_values) | set(tags))
        if missing:
            raise ValueError(f"{row['run_name']}: missing required tags {sorted(missing)}")
        if tag_values["model_size"] != MODEL_SIZE or int(tag_values["epochs"]) != EPOCHS:
            raise ValueError(f"{row['run_name']}: tag selection does not match 1.5B/eight epochs")

        learning_rate = float(row["learning_rate"])
        weight_decay = float(row["weight_decay"])
        batch_size = int(row["batch_size"])
        assert_close(float(tag_values["lr"]), learning_rate, "LR tag/source", row["run_name"])
        assert_close(float(tag_values["wd"]), weight_decay, "WD tag/source", row["run_name"])
        if int(tag_values["global_batch"]) != batch_size:
            raise ValueError(f"{row['run_name']}: batch-size tag/source mismatch")

        selected.append(
            SelectedRun(
                project=row["project"],
                run_id=row["run_id"],
                run_name=row["run_name"],
                run_url=row["run_url"],
                sweep_subversion=int(row["sweep_subversion"]),
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                batch_size=batch_size,
                val_loss=float(row["val_loss"]),
                val_loss_key=row["val_loss_key"],
                total_steps=int(tag_values["steps"]),
                tags=tags,
            )
        )
    return sorted(selected, key=lambda run: (run.learning_rate, run.weight_decay, run.batch_size, run.run_id))


def gaussian_smooth(
    epochs: np.ndarray,
    losses: np.ndarray,
    grid: np.ndarray,
    bandwidth: float = SMOOTHING_BANDWIDTH_EPOCHS,
) -> np.ndarray:
    """Smooth irregular history samples with Gaussian kernel regression."""
    scaled_distance = (grid[:, None] - epochs[None, :]) / bandwidth
    weights = np.exp(-0.5 * scaled_distance**2)
    denominators = weights.sum(axis=1)
    if np.any(denominators <= 0):
        raise ValueError("Gaussian smoother has an empty evaluation neighborhood")
    return (weights @ losses) / denominators


def fetch_one_run(api: Any, selected: SelectedRun) -> list[dict[str, Any]]:
    """Fetch and smooth one W&B training-loss history."""
    run = api.run(f"{selected.project}/{selected.run_id}")
    if run.state != "finished":
        raise ValueError(f"{selected.run_name}: W&B state changed to {run.state!r}")

    tag_values = parse_tag_values([str(tag) for tag in run.tags])
    config = run.config if isinstance(run.config, Mapping) else {}
    config_lr = float(nested_value(config, "optimizer.learning_rate"))
    config_wd = float(nested_value(config, "optimizer.weight_decay"))
    config_bs = int(nested_value(config, "trainer.train_batch_size"))
    assert_close(config_lr, selected.learning_rate, "config LR/source", selected.run_name)
    assert_close(config_wd, selected.weight_decay, "config WD/source", selected.run_name)
    if config_bs != selected.batch_size:
        raise ValueError(f"{selected.run_name}: config batch size/source mismatch")
    assert_close(float(tag_values["lr"]), config_lr, "LR tag/config", selected.run_name)
    assert_close(float(tag_values["wd"]), config_wd, "WD tag/config", selected.run_name)
    if int(tag_values["global_batch"]) != config_bs:
        raise ValueError(f"{selected.run_name}: batch-size tag/config mismatch")

    sampled = run.history(
        samples=HISTORY_SAMPLES,
        keys=["global_step", "train/loss"],
        pandas=False,
    )
    by_step: dict[int, float] = {}
    for row in sampled:
        raw_step = row.get("global_step")
        raw_loss = row.get("train/loss")
        if raw_step is None or raw_loss is None:
            continue
        step = int(raw_step)
        loss = float(raw_loss)
        if step >= 0 and math.isfinite(loss):
            by_step[step] = loss
    if len(by_step) < HISTORY_SAMPLES // 2:
        raise ValueError(f"{selected.run_name}: only {len(by_step)} usable training-loss samples")

    steps = np.asarray(sorted(by_step), dtype=float)
    losses = np.asarray([by_step[int(step)] for step in steps], dtype=float)
    sampled_epochs = EPOCHS * steps / selected.total_steps
    if sampled_epochs[0] > 0.1 or sampled_epochs[-1] < EPOCHS - 0.1:
        raise ValueError(
            f"{selected.run_name}: sampled history does not span training "
            f"({sampled_epochs[0]:.3f} to {sampled_epochs[-1]:.3f} epochs)"
        )

    grid = np.linspace(0.0, float(EPOCHS), SMOOTHING_GRID_POINTS)
    smoothed = gaussian_smooth(sampled_epochs, losses, grid)
    return [
        {
            "issue": ISSUE,
            "project": selected.project,
            "run_id": selected.run_id,
            "run_name": selected.run_name,
            "run_url": selected.run_url,
            "sweep_subversion": selected.sweep_subversion,
            "model_size": MODEL_SIZE,
            "epochs": EPOCHS,
            "total_steps": selected.total_steps,
            "learning_rate": selected.learning_rate,
            "weight_decay": selected.weight_decay,
            "batch_size": selected.batch_size,
            "val_loss": selected.val_loss,
            "val_loss_key": selected.val_loss_key,
            "history_samples": len(by_step),
            "smoothing_bandwidth_epochs": SMOOTHING_BANDWIDTH_EPOCHS,
            "training_epoch": epoch,
            "train_loss_smoothed": loss,
        }
        for epoch, loss in zip(grid, smoothed, strict=True)
    ]


def fetch_plot_data(selected_runs: Sequence[SelectedRun]) -> pd.DataFrame:
    """Fetch selected histories concurrently and return exact plot rows."""
    import wandb

    api = wandb.Api(timeout=180)
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as executor:
        future_to_run = {
            executor.submit(fetch_one_run, api, selected): selected
            for selected in selected_runs
        }
        for completed, future in enumerate(as_completed(future_to_run), start=1):
            selected = future_to_run[future]
            rows.extend(future.result())
            print(f"Fetched {completed:>2}/{len(selected_runs)}: {selected.run_name}")

    data = pd.DataFrame(rows)
    return data.sort_values(["learning_rate", "weight_decay", "batch_size", "run_id", "training_epoch"])


def stable_jitter(run_id: str, width: float = 0.34) -> float:
    """Return deterministic horizontal jitter centered on zero."""
    digest = hashlib.sha256(run_id.encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / (2**64 - 1)
    return (fraction - 0.5) * width


def tukey_inlier_mask(values: np.ndarray) -> np.ndarray:
    """Return the per-group 1.5-IQR inlier mask used by boxes and summaries."""
    first_quartile, third_quartile = np.quantile(values, [0.25, 0.75])
    interquartile_range = third_quartile - first_quartile
    lower = first_quartile - 1.5 * interquartile_range
    upper = third_quartile + 1.5 * interquartile_range
    return (values >= lower) & (values <= upper)


def format_lr(value: float) -> str:
    """Format a configured learning rate compactly with a power of ten."""
    exponent = int(math.floor(math.log10(value)))
    mantissa = value / 10**exponent
    if math.isclose(mantissa, 1.0, rel_tol=1e-4):
        return rf"$10^{{{exponent}}}$"
    return rf"${mantissa:.2f}\times10^{{{exponent}}}$"


def run_level_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Return one validated record per W&B run for the distribution panel."""
    constant_columns = [
        "learning_rate",
        "weight_decay",
        "batch_size",
        "val_loss",
        "val_loss_key",
        "history_samples",
    ]
    unique_counts = data.groupby("run_id")[constant_columns].nunique(dropna=False)
    if (unique_counts != 1).any().any():
        raise ValueError("run-level metadata varies within a smoothed curve")
    return data.drop_duplicates("run_id").sort_values(["learning_rate", "val_loss", "run_id"])


def plot(data: pd.DataFrame) -> pd.DataFrame:
    """Render LR-faceted training curves with an aligned validation distribution."""
    run_rows = run_level_rows(data)
    learning_rates = sorted(run_rows["learning_rate"].unique())
    if len(learning_rates) != 5:
        raise ValueError(f"expected five learning-rate groups, found {learning_rates}")

    val_min = float(run_rows["val_loss"].min())
    val_max = float(run_rows["val_loss"].max())
    norm = Normalize(vmin=val_min, vmax=val_max)
    cmap = plt.get_cmap("coolwarm")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(16.2, 8.25))
    grid = fig.add_gridspec(2, len(learning_rates), height_ratios=[1.45, 1.25], hspace=0.38, wspace=0.12)
    training_axes = []
    shared_axis = None
    for index in range(len(learning_rates)):
        axis = fig.add_subplot(grid[0, index], sharey=shared_axis)
        if shared_axis is None:
            shared_axis = axis
        training_axes.append(axis)
    distribution_axis = fig.add_subplot(grid[1, :])
    fig.subplots_adjust(left=0.065, right=0.925, top=0.82, bottom=0.12)

    for axis, learning_rate in zip(training_axes, learning_rates, strict=True):
        lr_runs = run_rows.loc[run_rows["learning_rate"] == learning_rate]
        for run in lr_runs.sort_values("val_loss", ascending=False).itertuples(index=False):
            curve = data.loc[data["run_id"] == run.run_id]
            axis.plot(
                curve["training_epoch"],
                curve["train_loss_smoothed"],
                color=cmap(norm(run.val_loss)),
                linewidth=1.0,
                alpha=0.7,
                zorder=2,
            )
        median_curve = (
            data.loc[data["learning_rate"] == learning_rate]
            .groupby("training_epoch", as_index=False)["train_loss_smoothed"]
            .median()
        )
        axis.plot(
            median_curve["training_epoch"],
            median_curve["train_loss_smoothed"],
            color="#111827",
            linewidth=2.0,
            zorder=3,
        )
        axis.set_title(f"LR {format_lr(learning_rate)}  ·  n={len(lr_runs)}", fontsize=12.5, pad=10)
        axis.set_xlim(0, EPOCHS)
        axis.set_xticks([0, 2, 4, 6, 8])
        axis.set_xlabel("Training epoch")
        axis.grid(color="#D1D5DB", linewidth=0.65, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    training_axes[0].set_ylabel("Smoothed training loss")
    training_axes[0].set_ylim(top=TRAINING_AXIS_MAX)
    for axis in training_axes[1:]:
        axis.tick_params(labelleft=False)

    positions = np.arange(len(learning_rates), dtype=float)
    tick_labels = []
    summary_rows: list[dict[str, Any]] = []
    for position, learning_rate in zip(positions, learning_rates, strict=True):
        lr_runs = run_rows.loc[run_rows["learning_rate"] == learning_rate]
        values = lr_runs["val_loss"].to_numpy(dtype=float)
        inliers = tukey_inlier_mask(values)
        inlier_values = values[inliers]
        outliers = ~inliers
        boxplot = distribution_axis.boxplot(
            [inlier_values],
            positions=[position],
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            boxprops={"facecolor": "#E2E8F0", "edgecolor": "#64748B", "linewidth": 1.0},
            medianprops={"color": "#111827", "linewidth": 1.8},
            whiskerprops={"color": "#64748B", "linewidth": 1.0},
            capprops={"color": "#64748B", "linewidth": 1.0},
        )
        for artist in (*boxplot["boxes"], *boxplot["medians"], *boxplot["whiskers"], *boxplot["caps"]):
            artist.set_zorder(2)

        x_values = np.asarray([position + stable_jitter(run_id) for run_id in lr_runs["run_id"]])
        clipped = values > VALIDATION_AXIS_MAX
        retained_and_visible = inliers & ~clipped
        distribution_axis.scatter(
            x_values[retained_and_visible],
            values[retained_and_visible],
            c=[cmap(norm(value)) for value in values[retained_and_visible]],
            s=48,
            edgecolor="white",
            linewidth=0.65,
            alpha=0.9,
            zorder=3,
        )
        visible_outliers = outliers & ~clipped
        distribution_axis.scatter(
            x_values[visible_outliers],
            values[visible_outliers],
            c=[cmap(norm(value)) for value in values[visible_outliers]],
            marker="^",
            s=62,
            edgecolor="white",
            linewidth=0.65,
            alpha=0.95,
            zorder=4,
        )
        distribution_axis.scatter(
            x_values[clipped],
            np.full(int(clipped.sum()), CLIPPED_VALIDATION_Y),
            c=[cmap(norm(value)) for value in values[clipped]],
            marker="^",
            s=62,
            edgecolor="white",
            linewidth=0.65,
            alpha=0.95,
            clip_on=False,
            zorder=4,
        )
        for x_value, value in zip(x_values[clipped], values[clipped], strict=True):
            distribution_axis.annotate(
                f"{value:.2f}",
                (x_value, CLIPPED_VALIDATION_Y),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7.5,
                color="#7F1D1D",
                zorder=5,
            )
        mean = float(np.mean(inlier_values))
        standard_deviation = (
            float(np.std(inlier_values, ddof=1)) if len(inlier_values) > 1 else float("nan")
        )
        median = float(np.median(inlier_values))
        best_index = int(np.argmin(values))
        distribution_axis.scatter(
            [position],
            [mean],
            marker="D",
            s=62,
            color="#111827",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
        distribution_axis.scatter(
            [x_values[best_index]],
            [values[best_index]],
            s=112,
            facecolor="none",
            edgecolor="#111827",
            linewidth=1.4,
            zorder=5,
        )
        distribution_axis.annotate(
            f"{values[best_index]:.3f}",
            (x_values[best_index], values[best_index]),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color="#111827",
            zorder=5,
        )
        tick_labels.append(
            f"{format_lr(learning_rate)}\n"
            f"n={len(values)}  ·  mean={mean:.3f}  ·  SD={standard_deviation:.3f}"
        )
        summary_rows.append(
            {
                "learning_rate": learning_rate,
                "n_runs": len(values),
                "n_inliers": int(inliers.sum()),
                "n_outliers": int(outliers.sum()),
                "mean_val_loss": mean,
                "median_val_loss": median,
                "sd_val_loss": standard_deviation,
                "min_val_loss": float(np.min(values)),
                "max_val_loss": float(np.max(values)),
            }
        )

    distribution_axis.set_title("Final contacts-v1 validation-loss distribution", loc="left", fontsize=13.5, pad=11)
    distribution_axis.set_ylabel("Final validation loss")
    distribution_axis.set_xlabel("Configured learning rate")
    distribution_axis.set_xticks(positions, tick_labels)
    distribution_axis.set_xlim(-0.55, len(learning_rates) - 0.45)
    distribution_axis.set_ylim(VALIDATION_AXIS_MIN, VALIDATION_AXIS_MAX)
    distribution_axis.grid(axis="x", visible=False)
    distribution_axis.grid(axis="y", color="#D1D5DB", linewidth=0.65, alpha=0.65)
    distribution_axis.spines[["top", "right"]].set_visible(False)
    outlier_count = int(sum(row["n_outliers"] for row in summary_rows))
    clipped_count = int((run_rows["val_loss"] > VALIDATION_AXIS_MAX).sum())
    distribution_axis.text(
        0.995,
        1.02,
        f"Boxes and summary statistics use Tukey inliers; triangles are {outlier_count} excluded "
        f"outliers ({clipped_count} capped above {VALIDATION_AXIS_MAX:g})",
        transform=distribution_axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        color="#64748B",
        clip_on=False,
    )
    distribution_axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markerfacecolor="#111827",
                markeredgecolor="white",
                markersize=7,
                label="Inlier mean",
            )
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 0.98),
        frameon=False,
        fontsize=8.5,
    )

    fig.suptitle(
        "Exp117 1.5B, eight epochs: training dynamics and validation outcomes",
        x=0.055,
        y=0.965,
        ha="left",
        fontsize=20,
    )
    fig.text(
        0.055,
        0.915,
        "Each line is one finished latest-subversion run. Color links its smoothed training curve "
        "to its final validation-loss point below.",
        color="#475569",
        fontsize=10.5,
    )
    fig.legend(
        handles=[Line2D([0], [0], color="#111827", linewidth=2.0, label="Median training curve")],
        loc="upper left",
        bbox_to_anchor=(0.055, 0.893),
        frameon=False,
        fontsize=9.5,
    )
    colorbar_axis = fig.add_axes([0.94, 0.54, 0.012, 0.23])
    colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_axis)
    colorbar.set_label("Final validation loss", labelpad=9)

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=150, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")
    SVG_PATH.write_text("\n".join(line.rstrip() for line in SVG_PATH.read_text().splitlines()) + "\n")
    plt.close(fig)
    return pd.DataFrame(summary_rows)


def main() -> int:
    """Fetch when requested, render the combined figure, and print LR summaries."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Re-fetch sampled training histories from W&B")
    args = parser.parse_args()

    selected_runs = load_selected_runs()
    if args.refresh or not PLOT_DATA_CSV.exists():
        data = fetch_plot_data(selected_runs)
    else:
        data = pd.read_csv(PLOT_DATA_CSV)
    data = data.loc[:, PLOT_DATA_COLUMNS]
    PLOT_DATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PLOT_DATA_CSV, index=False, float_format="%.9g")
    print(f"Wrote {len(data):,} exact plot rows to {PLOT_DATA_CSV}")

    summary = plot(data)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6g}"))
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
