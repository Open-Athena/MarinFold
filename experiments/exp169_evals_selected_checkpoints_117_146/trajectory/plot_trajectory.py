# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot validation loss and range-specific R-precision over training tokens."""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from huggingface_hub import HfFileSystem

from checkpoint_specs import CHECKPOINTS, HF_BUCKET_ROOT

HERE = Path(__file__).parent
EXPERIMENT = HERE.parent
INPUT = EXPERIMENT / "data" / "trajectory_checkpoint_metrics.csv"
OUTPUT = EXPERIMENT / "plots" / "checkpoint_trajectory.png"
COLORS = {
    "exp146-3b-e8": "#6b7280",
    "exp117-1_5b-e16": "#c65f00",
    "exp117-1_5b-e8-bs64": "#2673a6",
}
LABELS = {
    "exp146-3b-e8": "3B E8 (+ parameters)",
    "exp117-1_5b-e16": "1.5B E16 (+ epochs)",
    "exp117-1_5b-e8-bs64": "1.5B E8 (control)",
}
TABLE_LABELS = {
    "exp146-3b-e8": "+ parameters",
    "exp117-1_5b-e16": "+ epochs",
    "exp117-1_5b-e8-bs64": "control",
}
RUN_ORDER = ("exp117-1_5b-e8-bs64", "exp117-1_5b-e16", "exp146-3b-e8")
RANGE_TITLES = {
    "all": "R-precision, all",
    "short": "R-precision, short",
    "medium": "R-precision, medium",
    "long": "R-precision, long",
}


def plot_series(
    ax, data: pd.DataFrame, column: str, *, error_column: str | None
) -> None:
    """Plot each training run against billions of tokens on one axis."""

    for run_key in RUN_ORDER:
        group = data[data["run_key"] == run_key].sort_values("training_tokens")
        if group.empty:
            continue
        kwargs = {
            "color": COLORS[run_key],
            "label": LABELS[run_key],
            "marker": "o",
            "markersize": 5.5 if run_key == "exp117-1_5b-e8-bs64" else 5,
            "linewidth": 2.5 if run_key == "exp117-1_5b-e8-bs64" else 1.8,
            "zorder": 3 if run_key == "exp117-1_5b-e8-bs64" else 2,
        }
        if error_column is None:
            ax.plot(group["training_tokens_billions"], group[column], **kwargs)
        else:
            ax.errorbar(
                group["training_tokens_billions"],
                group[column],
                yerr=1.96 * group[error_column],
                capsize=2,
                **kwargs,
            )
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.set_xlabel("Training tokens (billions)")


def build_plot(data: pd.DataFrame, output: Path) -> None:
    """Build the five-panel trajectory figure."""

    required = {
        "run_key",
        "run_name",
        "training_tokens",
        "training_tokens_billions",
        "validation_loss",
        *{f"r_precision_{name}" for name in RANGE_TITLES},
        *{f"r_precision_{name}_sem" for name in RANGE_TITLES},
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"trajectory table is missing columns: {missing}")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    loss_ax = axes[0, 0]
    plot_series(loss_ax, data, "validation_loss", error_column=None)
    loss_ax.set_title("Validation loss")
    loss_ax.set_ylabel("eval/tokenized/contacts-v1-val/loss")

    range_axes = {
        "all": axes[0, 1],
        "short": axes[0, 2],
        "medium": axes[1, 0],
        "long": axes[1, 1],
    }
    for range_name, ax in range_axes.items():
        plot_series(
            ax,
            data,
            f"r_precision_{range_name}",
            error_column=f"r_precision_{range_name}_sem",
        )
        ax.set_title(RANGE_TITLES[range_name])
        ax.set_ylabel("Mean precision")
        ax.set_ylim(0, 0.75)

    legend_ax = axes[1, 2]
    legend_ax.axis("off")
    handles, labels = loss_ax.get_legend_handles_labels()
    legend_ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=11)

    max_header = f"{'comparison':<13} {'all':>5} {'short':>6} {'med':>5} {'long':>5}"
    max_rows = []
    for run_key in RUN_ORDER:
        group = data[data["run_key"] == run_key]
        values = [group[f"r_precision_{name}"].max() for name in RANGE_TITLES]
        max_rows.append(
            f"{TABLE_LABELS[run_key]:<13} "
            + " ".join(f"{value:>5.3f}" for value in values)
        )
    legend_ax.text(
        0,
        0.70,
        "MAX R-PRECISION",
        transform=legend_ax.transAxes,
        va="top",
        fontsize=9,
        fontweight="bold",
    )
    legend_ax.text(
        0,
        0.64,
        "\n".join((max_header, *max_rows)),
        transform=legend_ax.transAxes,
        va="top",
        fontsize=8.2,
        family="monospace",
        linespacing=1.35,
    )

    legend_ax.text(
        0,
        0.34,
        "W&B RUN IDS",
        transform=legend_ax.transAxes,
        va="top",
        fontsize=7.5,
        color="#4b5563",
        fontweight="bold",
    )
    for y, run_key in zip((0.29, 0.23, 0.17), RUN_ORDER, strict=True):
        run_names = data.loc[data["run_key"] == run_key, "run_name"].unique()
        if len(run_names) != 1:
            raise ValueError(f"expected one W&B run ID for {run_key}: {run_names}")
        legend_ax.text(
            0,
            y,
            str(run_names[0]),
            transform=legend_ax.transAxes,
            va="top",
            fontsize=5.7,
            family="monospace",
            color="#6b7280",
        )
    legend_ax.text(
        0,
        0.07,
        "Error bars show 95% CIs over 554 proteins.\n"
        "Each checkpoint uses 100 rollout-resampled generations per protein.",
        transform=legend_ax.transAxes,
        va="top",
        fontsize=7.2,
        color="#4b5563",
        linespacing=1.35,
    )
    fig.suptitle(
        "Training trajectories from a 1.5B E8 control "
        f"({len(data)}/{len(CHECKPOINTS)} checkpoints)",
        fontsize=15,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_metadata(output: Path) -> Path:
    """Write summary-slide provenance for the generated plot."""

    path = output.with_name(f"{output.name}.meta.json")
    path.write_text(
        json.dumps(
            {
                "script": str(Path(sys.argv[0]).as_posix()),
                "args": sys.argv[1:],
                "caption": (
                    "Validation loss and all/short/medium/long R-precision versus "
                    "training tokens for a 1.5B E8 control and two comparison "
                    "configurations that add epochs or parameters. Maxima are "
                    "tabulated and exact W&B run IDs appear in the side panel."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path


def upload(path: Path, token: str) -> None:
    """Publish one aggregate plot artifact to the public HF bucket."""

    fs = HfFileSystem(token=token)
    destination = f"{HF_BUCKET_ROOT}/summary/{path.name}"
    with path.open("rb") as source, fs.open(destination, "wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    print(f"[plot] {path} -> hf://{destination}")


def parse_args() -> argparse.Namespace:
    """Parse input/output overrides and publication mode."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--no-publish", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Render and optionally publish the current trajectory figure."""

    args = parse_args()
    data = pd.read_csv(args.input)
    if data.empty:
        raise ValueError("trajectory table has no completed checkpoints")
    build_plot(data, args.output)
    metadata = write_metadata(args.output)
    if not args.no_publish:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN must contain the open-athena write token")
        upload(args.output, token)
        upload(metadata, token)
    print(f"[plot] complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
