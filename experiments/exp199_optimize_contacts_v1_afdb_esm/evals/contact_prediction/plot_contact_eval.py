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
    "control-r0": "#117 r0\nPR #190",
    "control-r1": "#117 r1\nfresh",
    "control-r2": "#117 r2\nfresh",
    "control-r3": "#117 r3\nfresh",
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
    "exp146": ((10, -31), "left"),
    "exp166": ((12, 18), "left"),
    "trc-p06-aug": ((-8, 20), "right"),
    "trc-p03-aug": ((29, -30), "left"),
    "trc-p03-base": ((-20, 29), "right"),
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


def draw_boxplot(
    axis: plt.Axes, rows: list[pd.Series], values: dict[str, np.ndarray]
) -> None:
    positions = np.arange(1, len(rows) + 1)
    boxes = axis.boxplot(
        [values[row.key] for row in rows],
        positions=positions,
        widths=0.34,
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
        min(control_positions) - 0.47,
        max(control_positions) + 0.47,
        color=COLORS["control"],
        alpha=0.08,
        zorder=0,
    )
    axis.text(
        float(np.mean(control_positions)),
        1.066,
        "same #117 checkpoint · four separate evals",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#335950",
        weight="bold",
    )
    for position, row in zip(positions, rows, strict=True):
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
) -> None:
    baseline_r = float(values["protenix"].mean())
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
    text_offsets = ((-8, -13), (-5, 13), (8, 13), (10, -13))
    for (_, row), x_offset, text_offset, y in zip(
        controls.iterrows(), dodge, text_offsets, control_ys, strict=True
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
            row.display_name.split()[-1],
            (shown_x, y),
            xytext=text_offset,
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color="#335950",
        )
    axis.annotate(
        "#117 control\nfour evals",
        (control_x, max(control_ys) + 0.004),
        xytext=(0, 24),
        textcoords="offset points",
        ha="center",
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

    axis.set_xlim(2.952, 3.153)
    axis.set_ylim(0.402, 0.625)
    axis.set_xlabel("contacts-v1 validation loss on current scale (← lower is better)")
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
        ],
        loc="lower left",
        fontsize=7.8,
        frameon=True,
        framealpha=0.92,
    )


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
        gridspec_kw={"width_ratios": [1.42, 1.0]},
    )
    draw_boxplot(box_axis, rows, values)
    draw_scatter(scatter_axis, table, values)
    figure.suptitle(
        "AFDB/ESM mixing checkpoint contact prediction", fontsize=14, y=0.988
    )
    figure.text(
        0.5,
        0.044,
        (
            "Each box is one 554-protein evaluation. #117 r0 is PR #190; "
            "r1–r3 are fresh repeats of the same checkpoint."
        ),
        ha="center",
        fontsize=8.5,
        weight="bold",
    )
    figure.text(
        0.5,
        0.014,
        (
            "Historical losses for #75, #117, #146, and #166 use the empirical "
            "conversion current ≈ old + 0.38171. #117 scatter points are "
            "horizontally dodged for visibility."
        ),
        ha="center",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.078, 1, 0.955), w_pad=2.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)

    metadata = {
        "schema_version": 1,
        "figure": "final_checkpoint_rprecision",
        "metric": "all-range R-precision",
        "box_order": [row.key for row in rows],
        "loss_conversion": LOSS_CONVERSION,
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
