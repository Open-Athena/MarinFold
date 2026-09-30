# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the September 2026 project-update slide from saved contact evaluations.

The historical 554-protein series is retained for within-project tracking.
Structure-predictor comparisons use the 97 natural eval-val proteins instead:
the legacy set is mostly designs and overlaps baseline training data. Checkpoint
dates are training dates, even when their scores were measured retrospectively.
No network access or predictor execution is needed to reproduce this snapshot.

    uv run --project experiments/exp180_evals_contacts_v1_progress_over_time \
        --frozen python experiments/exp180_evals_contacts_v1_progress_over_time/plot_update_slide.py
"""

import json
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP232 = ROOT / "experiments/exp232_sweep_cv1_decontam"
EXP277 = ROOT / "experiments/exp277_models_single_mpnn_pilot"
AS_OF = pd.Timestamp("2026-09-30")
BLUE = "#2877cf"
TEAL = "#008578"
ORANGE = "#de713c"
INK = "#20282e"
MUTED = "#64717b"
GRID = "#e4e9ed"


def one_row(frame: pd.DataFrame, **filters: object) -> pd.Series:
    """Select exactly one result, rejecting missing or ambiguous source rows."""
    selected = frame
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def build_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble dated legacy results and a matched natural-protein comparison."""
    legacy = pd.read_csv(HERE / "data/rprecision_checkpoints.csv")
    legacy = legacy[legacy.inference != "oracle_best_of_100"].copy()
    legacy["training_data"] = "before decontamination"
    legacy["date_source"] = "exp180 historical checkpoint table"
    additions = []
    comparison_path = EXP232 / "evals/2026-08-24_rollout_v2/data/comparison.csv"
    comparison = pd.read_csv(comparison_path)
    # Dates queried from the named W&B runs on 2026-09-30. For the intermediate
    # 363k checkpoint use its history timestamp, not the later run termination.
    identities = [
        (
            "exp232-m1-p02-decontam",
            "#232 m1-p02",
            "2026-08-17",
            145199,
            "summary _timestamp=2026-08-17T18:28:27.972029Z",
        ),
        (
            "exp232-m2-p06-decontam",
            "#232 sweep",
            "2026-08-18",
            145199,
            "summary _timestamp=2026-08-18T10:52:10.290667Z",
        ),
        (
            "exp232-m2-p06-training",
            "#232 continued",
            "2026-08-22",
            363000,
            "global_step=363000 _timestamp=2026-08-22T23:54:09.232685Z",
        ),
    ]
    for key, label, date, step, evidence in identities:
        row = one_row(comparison, key=key)
        additions.append(
            {
                "label": label,
                "model": f"{row.wandb_run_id}-step-{step}",
                "date": date,
                "params": "1.5B",
                "issue": 232,
                "r_precision": row.r_all,
                "inference": "rollout",
                "training_data": "decontaminated native corpus",
                "source": f"{comparison_path.relative_to(ROOT)} (key={key})",
                "date_source": f"https://wandb.ai/open-athena/MarinFold/runs/{row.wandb_run_id}; {evidence}",
            }
        )

    epoch_path = EXP277 / "data/eval_rollout_v2_epochs/epoch_comparison.csv"
    epochs = pd.read_csv(epoch_path)
    legacy_epochs = one_row(epochs, subset="legacy_554", range="all")
    if legacy_epochs.n != 554:
        raise ValueError("The epoch comparison must cover all 554 legacy proteins")
    for epoch, date, run_name, step in [
        (1, "2026-09-13", "contacts-v1-exp277-m2-p06-full-epoch-1.5B", 266344),
        (
            2,
            "2026-09-19",
            "contacts-v1-exp277-m2-p06-full-epoch2-from213072-1.5B",
            479417,
        ),
    ]:
        additions.append(
            {
                "label": f"#277 epoch {epoch}",
                "model": f"{run_name}-step-{step}",
                "date": date,
                "params": "1.5B",
                "issue": 277,
                "r_precision": legacy_epochs[f"epoch{epoch}"],
                "inference": "rollout",
                "training_data": "decontaminated native + MPNN redesigns",
                "source": f"{epoch_path.relative_to(ROOT)} (legacy_554, all, epoch{epoch})",
                "date_source": f"exp277 README / history: training completed {date}; https://wandb.ai/open-athena/MarinFold/runs/{run_name}",
            }
        )
    timeline = pd.concat([legacy, pd.DataFrame(additions)], ignore_index=True)
    timeline["date"] = pd.to_datetime(timeline.date)
    timeline = timeline.sort_values(["date", "r_precision"]).reset_index(drop=True)
    if timeline.date.max() > AS_OF or not timeline.r_precision.between(0, 1).all():
        raise ValueError("Invalid date or precision in the timeline")

    natural_path = EXP277 / "data/eval_rollout_v2/figure_summary.csv"
    natural = pd.read_csv(natural_path)
    natural = natural[
        (natural.subset == "eval-val") & (natural.predictor != "MarinFold exp277")
    ].copy()
    natural["source"] = str(natural_path.relative_to(ROOT))
    natural_epochs = one_row(epochs, subset="eval-val", range="all")
    natural = pd.concat(
        [
            natural,
            pd.DataFrame(
                [
                    {
                        "subset": "eval-val",
                        "predictor": f"MarinFold exp277 epoch {epoch}",
                        "n": 97,
                        "mean": natural_epochs[f"epoch{epoch}"],
                        "source": str(epoch_path.relative_to(ROOT)),
                    }
                    for epoch in (1, 2)
                ]
            ),
        ],
        ignore_index=True,
    )
    if len(natural) != 8 or not (natural.n == 97).all():
        raise ValueError("Natural panel must contain eight predictors on 97 proteins")
    return timeline, natural.sort_values("mean", ascending=False)


def style(ax: Axes) -> None:
    """Apply the restrained slide style."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c9d1d7")
    ax.tick_params(colors=MUTED, labelsize=11, length=0, pad=8)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def staircase(ax: Axes, rows: pd.DataFrame, color: str, width: float) -> None:
    """Draw a running maximum, grouping same-day measurements before stepping."""
    daily = rows.groupby("date").r_precision.max().sort_index().cummax()
    ax.step(
        [*daily.index, AS_OF],
        [*daily.values, daily.iloc[-1]],
        where="post",
        color=color,
        linewidth=width,
        zorder=2,
    )


def draw_timeline(ax: Axes, timeline: pd.DataFrame) -> None:
    """Draw separate historical and decontaminated-lineage frontiers."""
    style(ax)
    rollout = timeline[timeline.inference == "rollout"]
    old = rollout[rollout.training_data == "before decontamination"]
    new = rollout[rollout.training_data != "before decontamination"]
    pairwise = timeline[timeline.inference == "pairwise"]
    staircase(ax, rollout, INK, 2.5)
    staircase(ax, new, TEAL, 2.6)
    ax.scatter(
        old.date,
        old.r_precision,
        s=67,
        color=BLUE,
        edgecolor="white",
        linewidth=1.5,
        zorder=4,
    )
    ax.scatter(
        new.date,
        new.r_precision,
        s=83,
        color=TEAL,
        marker="D",
        edgecolor="white",
        linewidth=1.5,
        zorder=5,
    )
    ax.scatter(
        pairwise.date,
        pairwise.r_precision,
        s=56,
        color=ORANGE,
        marker="^",
        edgecolor="white",
        linewidth=1.3,
        zorder=4,
    )
    ax.axhline(0.6, color="#bb7777", linewidth=1.1, linestyle=(0, (4, 4)), zorder=0)
    ax.text(
        pd.Timestamp("2026-06-25"),
        0.611,
        "August goal  0.600",
        color="#a15b5b",
        fontsize=11,
    )

    # Positions are in data coordinates so the same annotated timeline also
    # works as a full-width standalone slide. Only milestones carry labels.
    labels = [
        ("#61/#75 E8", "#75  8 epochs", "2026-06-24", 0.46, "left"),
        ("#120 re-epoch", "#120  re-epoch", "2026-07-05", 0.37, "center"),
        ("#117 E16 final", "#117  16 epochs", "2026-07-16", 0.67, "center"),
        ("#166 AA aug", "#166  AA augmentation", "2026-08-02", 0.75, "center"),
        ("#199 CW p06-aug", "#199  AFDB + ESM", "2026-07-30", 0.44, "center"),
        ("#199 CW cooldown", "#199  cooldown", "2026-08-21", 0.80, "center"),
        ("#232 sweep", "#232  decontaminated", "2026-08-11", 0.29, "center"),
        ("#232 continued", "#232  more training", "2026-08-30", 0.39, "center"),
        ("#277 epoch 1", "#277  + MPNN, epoch 1", "2026-09-06", 0.48, "center"),
        ("#277 epoch 2", "#277  epoch 2", "2026-09-23", 0.72, "center"),
    ]
    for label, name, date, y, align in labels:
        row = one_row(rollout, label=label)
        color = TEAL if row.training_data != "before decontamination" else INK
        ax.annotate(
            f"{name}\n{row.r_precision:.3f}",
            xy=(row.date, row.r_precision),
            xytext=(pd.Timestamp(date), y),
            ha=align,
            va="center",
            fontsize=10.5,
            color=color,
            linespacing=1.4,
            arrowprops={"arrowstyle": "-", "color": "#b5c0c7", "lw": 0.9, "shrinkB": 7},
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
            zorder=6,
        )

    ax.annotate(
        "Early pairwise scores\n0.028–0.031",
        xy=(pd.Timestamp("2026-06-17"), 0.03),
        xytext=(pd.Timestamp("2026-06-24"), 0.13),
        color=ORANGE,
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#c9b7ac", "lw": 0.9},
        ha="center",
    )
    ax.text(
        pd.Timestamp("2026-09-30"),
        0.654,
        "Historical best  0.631",
        ha="right",
        color=INK,
        fontsize=10.5,
    )
    ax.text(
        pd.Timestamp("2026-09-30"),
        0.577,
        "New lineage  0.622",
        ha="right",
        color=TEAL,
        fontsize=10.5,
    )
    ax.set_xlim(pd.Timestamp("2026-06-10"), AS_OF + pd.Timedelta(days=2))
    ax.set_ylim(0, 0.87)
    ax.set_yticks(np.arange(0, 0.81, 0.2))
    ticks = pd.to_datetime(
        [
            "2026-06-15",
            "2026-07-01",
            "2026-07-15",
            "2026-08-01",
            "2026-08-15",
            "2026-09-01",
            "2026-09-15",
            "2026-09-30",
        ]
    )
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_ylabel(
        "Contact R-precision · all ranges", color=MUTED, fontsize=12, labelpad=14
    )
    ax.set_title(
        "Historical benchmark · 554 proteins",
        loc="left",
        fontsize=15,
        color=INK,
        pad=18,
    )
    ax.legend(
        handles=[
            Line2D(
                [], [], color=INK, lw=2.5, label="Running best · rollout + resample"
            ),
            Line2D(
                [],
                [],
                color=BLUE,
                marker="o",
                ls="none",
                label="Before decontamination",
            ),
            Line2D(
                [],
                [],
                color=TEAL,
                marker="D",
                lw=2,
                label="Decontaminated native data (+ MPNN in #277)",
            ),
            Line2D(
                [],
                [],
                color=ORANGE,
                marker="^",
                ls="none",
                label="Original pairwise scoring · separate recipe",
            ),
        ],
        loc="lower right",
        bbox_to_anchor=(1, 0.015),
        frameon=False,
        fontsize=10,
        labelcolor=MUTED,
        handlelength=2.2,
        labelspacing=0.8,
    )


def draw_natural(ax: Axes, natural: pd.DataFrame) -> None:
    """Show current and baseline scores on the same 97 natural proteins."""
    labels = {
        "MarinFold exp277 epoch 1": "#277 epoch 1 · current default",
        "MarinFold exp277 epoch 2": "#277 epoch 2",
        "MarinFold exp232": "#232 continued",
        "seq-KNN (decontaminated corpus)": "Sequence-KNN · native corpus",
    }
    for i, row in enumerate(natural.itertuples()):
        y = 7 - i
        color = TEAL if "MarinFold" in row.predictor else "#8c98a2"
        ax.text(
            0,
            y + 0.26,
            labels.get(row.predictor, row.predictor),
            fontsize=11,
            color=INK,
            va="bottom",
        )
        ax.hlines(y, 0, 0.91, color=GRID, lw=2)
        ax.hlines(y, 0, row.mean, color=color, lw=3)
        ax.plot(row.mean, y, "o", color=color, markersize=5)
        ax.text(
            0.985,
            y + 0.02,
            f"{row.mean:.3f}",
            ha="right",
            fontsize=12,
            color=color,
            fontweight="bold" if "MarinFold" in row.predictor else "normal",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.4, 8)
    ax.axis("off")
    ax.set_title(
        "Natural proteins · 97 eval-val", loc="left", fontsize=15, color=INK, pad=18
    )


def save_figure(fig: Figure, stem: str, caption: str) -> None:
    """Save a slide in raster and vector formats with reproduction metadata."""
    for extension in ("png", "svg", "pdf"):
        path = HERE / "plots" / f"{stem}.{extension}"
        fig.savefig(path, dpi=200, facecolor="white")
        if extension == "png":
            path.with_suffix(".png.meta.json").write_text(
                json.dumps(
                    {
                        "script": "plot_update_slide.py",
                        "args": [],
                        "caption": caption,
                        "as_of": str(AS_OF.date()),
                        "source_data": "data/progress_slide_checkpoints.csv; data/progress_slide_natural.csv",
                    },
                    indent=2,
                )
                + "\n"
            )
    plt.close(fig)


def main() -> None:
    """Build the source tables, combined project slide, and full-width timeline."""
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42}
    )
    timeline, natural = build_data()
    timeline.to_csv(
        HERE / "data/progress_slide_checkpoints.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    natural.to_csv(HERE / "data/progress_slide_natural.csv", index=False)
    title = "Best contacts-v1 model trained to date"
    caption = "Historical 554-protein progress with a separate decontaminated-data lineage; natural eval-val comparisons use 97 matched proteins. Second epoch ties the current default."
    for combined in (True, False):
        fig = plt.figure(figsize=(20, 11.25))
        fig.text(0.06, 0.937, title, fontsize=27, color=INK)
        fig.text(
            0.06,
            0.900,
            "Contact R-precision  ·  June–September 2026",
            fontsize=14,
            color=MUTED,
        )
        fig.text(
            0.96, 0.939, "UPDATED 30 SEP 2026", fontsize=10, ha="right", color=MUTED
        )
        ax = fig.add_axes((0.075, 0.235, 0.595 if combined else 0.87, 0.585))
        draw_timeline(ax, timeline)
        if combined:
            draw_natural(fig.add_axes((0.725, 0.31, 0.245, 0.51)), natural)
            fig.text(
                0.725,
                0.253,
                "Second epoch: +0.002 on natural proteins",
                fontsize=12,
                color=TEAL,
            )
            fig.text(
                0.725,
                0.227,
                "95% CI −0.006 to +0.011 · effectively tied",
                fontsize=10.5,
                color=MUTED,
            )
        fig.text(
            0.075,
            0.16,
            "Since August: decontaminated training → more training → ProteinMPNN sequence expansion",
            fontsize=15,
            color=INK,
        )
        fig.text(
            0.075,
            0.122,
            "Legacy-554 tracks project history; ~75% are designs. Structure baselines are compared on natural eval-val only.",
            fontsize=10.8,
            color=MUTED,
        )
        fig.text(
            0.075,
            0.095,
            "Dates mark checkpoint training (scores may be retrospective). #277 uses decontaminated native backbones; redesign homology is unaudited.",
            fontsize=10.3,
            color=MUTED,
        )
        fig.text(
            0.075,
            0.068,
            "100 rollouts + resampling; capped samples excluded (epoch 1: 1; epoch 2: 44). Differences <0.005 are ties; eval-test was not re-read.",
            fontsize=10.3,
            color=MUTED,
        )
        fig.text(
            0.075,
            0.035,
            "Sources: #180 · #232 · #277 / PR #297     |     plot_update_slide.py (no arguments)",
            fontsize=9.5,
            color=MUTED,
        )
        stem = (
            "contacts_v1_progress_2026-09-30"
            if combined
            else "rprecision_timeline_2026-09-30"
        )
        save_figure(fig, stem, caption)
        print(f"Wrote plots/{stem}.{{png,svg,pdf}}")


if __name__ == "__main__":
    main()
