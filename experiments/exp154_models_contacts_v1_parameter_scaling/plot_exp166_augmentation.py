# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the exp166 amino-acid augmentation design and validation results.

Run after ``fetch_exp166_wandb.py`` from the exp154 analysis directory::

    uv run --with matplotlib --with numpy --with pandas --with scipy \
        python plot_exp166_augmentation.py
"""

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PLOTS_DIR = HERE / "plots"
SOURCE_RUNS_CSV = DATA_DIR / "wandb_runs.csv"
FINAL_CSV = DATA_DIR / "exp166_final_comparisons.csv"
HISTORY_CSV = DATA_DIR / "exp166_validation_trajectories.csv"
SCHEMATIC_CSV = DATA_DIR / "exp166_augmentation_schematic.csv"
DISTRIBUTION_CSV = DATA_DIR / "exp166_epoch_distributions.csv"
HIGHLIGHT_RUN_ID = "prot-exp166-cv1-aaaug-1_5b-e8-lr3p162e-3-wd0p1-bs128-exp117-init-us-east1"
TRAJECTORY_AXIS_MAX = 3.72
TRAJECTORY_CLIP_Y = 3.69

COLORS = {
    "exp117": "#64748B",
    "scratch": "#2563EB",
    "continued": "#7C3AED",
    "highlight": "#D97706",
    "e16": "#111827",
    "improved": "#0F766E",
    "worse": "#C2410C",
    "neutral": "#94A3B8",
    "residue": "#DBEAFE",
    "terminus": "#FEF3C7",
    "fixed": "#F1F5F9",
    "sequence": "#EDE9FE",
}

DOCUMENT_SECTIONS = (
    {"element_id": "doc_type", "label": "<contacts-v1>", "section": "document type", "affected_by_augmentation": False},
    {"element_id": "sequence", "label": "<begin_sequence>  sequence statements", "section": "sequence", "affected_by_augmentation": True},
    {"element_id": "structure", "label": "<begin_statements>  contact statements", "section": "structure", "affected_by_augmentation": False},
    {"element_id": "end", "label": "<end>", "section": "end", "affected_by_augmentation": False},
)

SCHEMATIC_STATEMENTS = (
    {"statement_id": "s1", "tokens": "<p22> <PHE>", "statement_type": "residue", "original_order": 1, "augmented_order": 2},
    {"statement_id": "s2", "tokens": "<n-term> <p20>", "statement_type": "terminus", "original_order": 2, "augmented_order": 4},
    {"statement_id": "s3", "tokens": "<p21> <ALA>", "statement_type": "residue", "original_order": 3, "augmented_order": 1},
    {"statement_id": "s4", "tokens": "<c-term> <p22>", "statement_type": "terminus", "original_order": 4, "augmented_order": 5},
    {"statement_id": "s5", "tokens": "<p20> <ALA>", "statement_type": "residue", "original_order": 5, "augmented_order": 3},
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    """Save a figure as SVG and 150 dpi PNG."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    svg_path = PLOTS_DIR / f"{stem}.svg"
    fig.savefig(svg_path, bbox_inches="tight")
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def write_schematic_csv() -> None:
    """Write the document sections and statement orders used in the diagram."""
    SCHEMATIC_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "record_type",
        "element_id",
        "label",
        "section",
        "original_order",
        "augmented_order",
        "is_moved",
        "affected_by_augmentation",
    )
    with SCHEMATIC_CSV.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for section in DOCUMENT_SECTIONS:
            writer.writerow(
                {
                    "record_type": "document_section",
                    **section,
                    "original_order": "",
                    "augmented_order": "",
                    "is_moved": False,
                }
            )
        for statement in SCHEMATIC_STATEMENTS:
            writer.writerow(
                {
                    "record_type": "sequence_statement",
                    "element_id": statement["statement_id"],
                    "label": statement["tokens"],
                    "section": statement["statement_type"],
                    "original_order": statement["original_order"],
                    "augmented_order": statement["augmented_order"],
                    "is_moved": statement["original_order"] != statement["augmented_order"],
                    "affected_by_augmentation": True,
                }
            )


def draw_token_row(
    ax: plt.Axes,
    statements: Sequence[Mapping[str, Any]],
    order_key: str,
    y: float,
    label: str,
) -> None:
    """Draw one ordered row of two-token sequence statements."""
    ordered = sorted(statements, key=lambda statement: int(statement[order_key]))
    start_x = 2.15
    box_width = 1.62
    gap = 0.14
    ax.text(0.18, y, label, ha="left", va="center", fontsize=10.2, fontweight="bold", color="#0F172A")
    for index, statement in enumerate(ordered):
        x = start_x + index * (box_width + gap)
        fill = COLORS[str(statement["statement_type"])]
        box = FancyBboxPatch(
            (x, y - 0.33),
            box_width,
            0.66,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=fill,
            edgecolor="#94A3B8",
            linewidth=0.8,
        )
        ax.add_patch(box)
        ax.text(x + box_width / 2, y, str(statement["tokens"]), ha="center", va="center", fontsize=8.6, color="#0F172A")


def plot_augmentation_schematic() -> None:
    """Show the contacts-v1 document structure and the only shuffled section."""
    write_schematic_csv()
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(13.0, 4.8))
    fig.subplots_adjust(left=0.035, right=0.985, top=0.82, bottom=0.06)
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    sections = (
        (0.2, 1.45, "<contacts-v1>", "document type", COLORS["fixed"], "#94A3B8"),
        (1.78, 5.15, "<begin_sequence>\nsequence statements", "position ↔ amino acid · termini", COLORS["sequence"], COLORS["continued"]),
        (7.05, 4.65, "<begin_statements>\ncontact statements", "<contact> <pX> <pY>", COLORS["fixed"], "#94A3B8"),
        (11.82, 0.95, "<end>", "", COLORS["fixed"], "#94A3B8"),
    )
    for x, width, label, subtitle, fill, edge in sections:
        box = FancyBboxPatch(
            (x, 3.25),
            width,
            0.82,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.4 if edge == COLORS["continued"] else 0.8,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, 3.73, label, ha="center", va="center", fontsize=9.4, color="#0F172A", fontweight="bold")
        if subtitle:
            ax.text(x + width / 2, 3.42, subtitle, ha="center", va="center", fontsize=8.1, color="#475569")
    ax.text(0.2, 4.22, "contacts-v1 document", ha="left", va="center", fontsize=10.5, fontweight="bold", color="#0F172A")

    zoom_arrow = FancyArrowPatch(
        (4.35, 3.22),
        (4.35, 2.72),
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.2,
        color=COLORS["continued"],
    )
    ax.add_patch(zoom_arrow)
    ax.text(
        4.55,
        2.96,
        "only this section is augmented",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#5B21B6",
    )

    draw_token_row(ax, SCHEMATIC_STATEMENTS, "original_order", 2.25, "Cached sequence order")
    draw_token_row(ax, SCHEMATIC_STATEMENTS, "augmented_order", 1.10, "Training view")
    permutation_arrow = FancyArrowPatch(
        (11.35, 2.05),
        (11.35, 1.30),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.3,
        color=COLORS["continued"],
    )
    ax.add_patch(permutation_arrow)
    ax.text(
        11.55,
        1.68,
        "fresh permutation\nper training occurrence",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#5B21B6",
    )
    ax.text(
        2.15,
        0.28,
        "Each two-token statement stays intact. Position assignments, contacts, document boundaries, and validation data are unchanged.",
        ha="left",
        va="center",
        fontsize=8.8,
        color="#475569",
    )

    fig.suptitle("Augmentation reshuffles only the contacts-v1 sequence section", x=0.035, y=0.955, ha="left", fontsize=18)
    fig.text(
        0.035,
        0.875,
        "The document still describes the same amino acids at the same positions and the same target contacts.",
        fontsize=10.2,
        color="#475569",
    )
    save_figure(fig, "exp166_augmentation_schematic")


def load_final_data() -> pd.DataFrame:
    """Load and type the long-form final-loss comparison table."""
    data = pd.read_csv(FINAL_CSV)
    for column in ("is_canonical", "is_highlighted_best"):
        data[column] = data[column].astype(str).str.lower().eq("true")
    if data.loc[data["condition"].eq("exp117_e8")].shape[0] != 6:
        raise ValueError("expected six matched exp117 eight-epoch baselines")
    canonical_exp166 = data.loc[data["condition"].str.startswith("exp166_") & data["is_canonical"]]
    if canonical_exp166.shape[0] != 12:
        raise ValueError("expected twelve canonical exp166 trials")
    return data


def delta_color(delta: float) -> str:
    """Color a paired change by whether lower validation loss was achieved."""
    if delta < -1e-9:
        return COLORS["improved"]
    if delta > 1e-9:
        return COLORS["worse"]
    return COLORS["neutral"]


def best_attempts_by_config(data: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Select the lowest-loss finished attempt for each matched configuration."""
    attempts = data.loc[data["condition"].eq(condition)]
    best_indices = attempts.groupby("config_rank")["val_loss"].idxmin()
    best = attempts.loc[best_indices].set_index("config_rank").sort_index()
    if len(best) != 6:
        raise ValueError(f"expected six best attempts for {condition}, found {len(best)}")
    return best


def plot_pair_panel(
    ax: plt.Axes,
    data: pd.DataFrame,
    condition: str,
    title: str,
    show_labels: bool,
) -> tuple[np.ndarray, float]:
    """Draw matched baselines against each configuration's best exp166 attempt."""
    baselines = data.loc[data["condition"].eq("exp117_e8")].set_index("config_rank")
    targets = best_attempts_by_config(data, condition)
    ranks = sorted(baselines.index)
    deltas: list[float] = []

    for y, rank in enumerate(ranks):
        baseline = baselines.loc[rank]
        target = targets.loc[rank]
        x0 = float(baseline["val_loss"])
        x1 = float(target["val_loss"])
        delta = x1 - x0
        deltas.append(delta)
        color = delta_color(delta)
        ax.add_patch(
            FancyArrowPatch(
                (x0, y),
                (x1, y),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.7,
                color=color,
                alpha=0.85,
                shrinkA=5,
                shrinkB=6,
            )
        )
        ax.scatter(x0, y, s=42, facecolor="white", edgecolor=COLORS["exp117"], linewidth=1.4, zorder=3)
        target_color = COLORS["scratch"] if condition == "exp166_scratch" else COLORS["continued"]
        is_overall_best = bool(target["is_highlighted_best"])
        ax.scatter(
            x1,
            y,
            s=125 if is_overall_best else 54,
            marker="*" if is_overall_best else "D",
            facecolor=COLORS["highlight"] if is_overall_best else target_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=5 if is_overall_best else 4,
        )
        ax.text(2.815, y, f"{delta:+.3f}", ha="right", va="center", fontsize=8.9, color=color, fontweight="bold")

        other_attempts = data.loc[
            data["config_rank"].eq(rank)
            & data["condition"].eq(condition)
            & data["run_id"].ne(target["run_id"])
        ]
        if not other_attempts.empty:
            ax.scatter(
                other_attempts["val_loss"],
                np.full(len(other_attempts), y),
                s=40,
                marker="D",
                facecolor="white",
                edgecolor=target_color,
                linewidth=1.1,
                zorder=3,
            )

    delta_values = np.asarray(deltas, dtype=float)
    p_value = float(wilcoxon(delta_values, method="exact").pvalue)
    median_delta = float(np.median(delta_values))
    ax.set_title(f"{title}\nmedian Δ vs exp117 e8 = {median_delta:+.3f}", loc="left", fontsize=12.5, pad=9)
    ax.set_xlim(2.64, 2.82)
    ax.set_xticks(np.arange(2.65, 2.81, 0.05))
    ax.set_xlabel("Final validation loss   (← lower is better)")
    ax.set_yticks(range(len(ranks)))
    if show_labels:
        labels = [str(baselines.loc[rank, "config_label"]) for rank in ranks]
        ax.set_yticklabels(labels, fontsize=9.2)
    else:
        ax.tick_params(axis="y", labelleft=False)
    ax.grid(axis="x", color="#CBD5E1", linewidth=0.7, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return delta_values, p_value


def plot_matched_final_losses(data: pd.DataFrame) -> None:
    """Plot exp117 e8 baselines against the best finished exp166 attempts."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 7.8), sharey=True)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.68, bottom=0.16, wspace=0.08)

    scratch_delta, scratch_p = plot_pair_panel(
        axes[0],
        data,
        "exp166_scratch",
        "Eight augmented epochs from scratch",
        show_labels=True,
    )
    continued_delta, continued_p = plot_pair_panel(
        axes[1],
        data,
        "exp166_continued",
        "Eight augmented epochs after exp117",
        show_labels=False,
    )
    axes[0].invert_yaxis()

    legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["exp117"], label="Matched exp117 e8"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=COLORS["scratch"], markeredgecolor="white", label="Best exp166 scratch attempt"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=COLORS["continued"], markeredgecolor="white", label="Best exp166 after-exp117 attempt"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor=COLORS["continued"], label="Other finished regional race"),
        Line2D([0], [0], marker="*", linestyle="none", markersize=10, markerfacecolor=COLORS["highlight"], markeredgecolor="white", label="Overall best finished exp166 run"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.61, 0.83), ncol=3, frameon=False, fontsize=9.2)
    fig.suptitle("Matched configurations distinguish scratch from warm-started augmentation", x=0.03, y=0.96, ha="left", fontsize=19)
    fig.text(
        0.03,
        0.885,
        "Arrows end at the lowest-loss finished exp166 attempt for the same LR, weight decay, and batch size. Δ = exp166 − exp117 e8; negative is better.",
        color="#475569",
        fontsize=10.3,
    )
    fig.text(
        0.23,
        0.055,
        f"Paired Wilcoxon using the best finished exp166 attempt per configuration (descriptive only, n=6): "
        f"scratch p={scratch_p:.3f}; after exp117 p={continued_p:.3f}.",
        color="#475569",
        fontsize=9.2,
    )
    if int(np.sum(scratch_delta < 0)) != 2 or int(np.sum(continued_delta < 0)) != 4:
        raise ValueError("unexpected paired-direction counts")
    save_figure(fig, "exp166_matched_final_loss")


def build_epoch_distribution_data(final_data: pd.DataFrame) -> pd.DataFrame:
    """Build the three requested run distributions without config matching."""
    source = pd.read_csv(SOURCE_RUNS_CSV)
    source = source.loc[
        source["issue"].eq(117)
        & source["model_size"].eq("1_5b")
        & source["epochs"].isin([8, 16])
        & source["state"].eq("finished")
    ].copy()
    if source.groupby("epochs").size().to_dict() != {8: 42, 16: 8}:
        raise ValueError(f"unexpected exp117 epoch counts: {source.groupby('epochs').size().to_dict()}")
    if set(source["val_loss_key"]) != {"eval/tokenized/contacts-v1-val/loss"}:
        raise ValueError("exp117 distribution contains an unexpected validation-loss key")

    rows: list[dict[str, Any]] = []
    condition_meta = {
        8: ("exp117_e8_all", "exp117 · 8 epochs", 0),
        16: ("exp117_e16_all", "exp117 · 16 epochs", 1),
    }
    for row in source.itertuples(index=False):
        condition, label, order = condition_meta[int(row.epochs)]
        rows.append(
            {
                "condition": condition,
                "condition_label": label,
                "condition_order": order,
                "issue": 117,
                "effective_total_epochs": int(row.epochs),
                "run_id": row.run_id,
                "run_name": row.run_name,
                "run_url": row.run_url,
                "trial_id": "",
                "learning_rate": float(row.learning_rate),
                "weight_decay": float(row.weight_decay),
                "batch_size": int(row.batch_size),
                "val_loss": float(row.val_loss),
                "val_loss_key": row.val_loss_key,
                "is_canonical": True,
                "is_highlighted_best": False,
            }
        )

    continued = final_data.loc[final_data["condition"].eq("exp166_continued")]
    if len(continued) != 8:
        raise ValueError(f"expected 8 finished exp166 warm-start attempts, found {len(continued)}")
    for row in continued.itertuples(index=False):
        rows.append(
            {
                "condition": "exp166_e8_plus_e8_aug",
                "condition_label": "exp166 · 8 + 8 augmented",
                "condition_order": 2,
                "issue": 166,
                "effective_total_epochs": 16,
                "run_id": row.run_id,
                "run_name": row.run_name,
                "run_url": row.run_url,
                "trial_id": row.trial_id,
                "learning_rate": float(row.learning_rate),
                "weight_decay": float(row.weight_decay),
                "batch_size": int(row.batch_size),
                "val_loss": float(row.val_loss),
                "val_loss_key": "eval/tokenized/contacts-v1-val/loss",
                "is_canonical": bool(row.is_canonical),
                "is_highlighted_best": bool(row.is_highlighted_best),
            }
        )

    data = pd.DataFrame(rows)
    data["is_winner"] = False
    winner_indices = data.groupby("condition")["val_loss"].idxmin()
    data.loc[winner_indices, "is_winner"] = True
    highlighted = data.loc[data["is_highlighted_best"]]
    if len(highlighted) != 1 or highlighted.iloc[0]["run_id"] != HIGHLIGHT_RUN_ID or not highlighted.iloc[0]["is_winner"]:
        raise ValueError("specified best augmentation run is not the highlighted exp166 distribution winner")
    data = data.sort_values(["condition_order", "val_loss", "run_id"]).reset_index(drop=True)
    data.to_csv(DISTRIBUTION_CSV, index=False, float_format="%.9g")
    return data


def plot_epoch_distributions(data: pd.DataFrame) -> None:
    """Compare all exp117 e8/e16 and finished exp166 warm-start losses."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(13.2, 5.8))
    grid = fig.add_gridspec(2, 1, height_ratios=[0.8, 2.8], hspace=0.035)
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.815, bottom=0.19)

    conditions = [
        "exp117_e8_all",
        "exp117_e16_all",
        "exp166_e8_plus_e8_aug",
    ]
    colors = [COLORS["exp117"], "#2563EB", COLORS["continued"]]
    rng = np.random.default_rng(166)
    jitter_by_run: dict[str, float] = {}
    for position, condition in enumerate(conditions):
        group = data.loc[data["condition"].eq(condition)].sort_values("run_id")
        jitter = rng.uniform(-0.13, 0.13, size=len(group))
        for run_id, x in zip(group["run_id"], position + jitter, strict=True):
            jitter_by_run[str(run_id)] = float(x)

    for ax in (top, bottom):
        for position, (condition, color) in enumerate(zip(conditions, colors, strict=True)):
            group = data.loc[data["condition"].eq(condition)]
            x = np.asarray([jitter_by_run[str(run_id)] for run_id in group["run_id"]])
            ax.scatter(
                x,
                group["val_loss"],
                s=34,
                color=color,
                edgecolor="white",
                linewidth=0.55,
                alpha=0.78,
                zorder=3,
            )

    box_data = [data.loc[data["condition"].eq(condition), "val_loss"].to_numpy() for condition in conditions]
    boxes = bottom.boxplot(
        box_data,
        positions=np.arange(3),
        widths=0.38,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0F172A", "linewidth": 1.5},
        whiskerprops={"color": "#64748B", "linewidth": 1.0},
        capprops={"color": "#64748B", "linewidth": 1.0},
        boxprops={"color": "#64748B", "linewidth": 1.0},
        zorder=2,
    )
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.16)

    for position, condition in enumerate(conditions):
        winner = data.loc[data["condition"].eq(condition) & data["is_winner"]].iloc[0]
        bottom.scatter(
            position,
            float(winner["val_loss"]),
            s=170,
            marker="*",
            facecolor=COLORS["highlight"],
            edgecolor="white",
            linewidth=0.9,
            zorder=6,
        )
        bottom.annotate(
            f"{float(winner['val_loss']):.3f}\nLR {float(winner['learning_rate']):.3g} · WD {float(winner['weight_decay']):g} · BS {int(winner['batch_size'])}",
            (position, float(winner["val_loss"])),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8.4,
            color="#92400E",
            fontweight="bold",
        )

    bottom.set_ylim(2.60, 2.92)
    top.set_ylim(2.92, 3.72)
    top.spines["bottom"].set_visible(False)
    bottom.spines["top"].set_visible(False)
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    bottom.set_yticks([2.65, 2.70, 2.75, 2.80, 2.85, 2.90])
    top.set_yticks([3.0, 3.2, 3.4, 3.6])
    for ax in (top, bottom):
        ax.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.7)
        ax.grid(axis="x", visible=False)
        ax.spines[["right"]].set_visible(False)
    break_size = 0.009
    top.plot((-break_size, +break_size), (-break_size, +break_size), transform=top.transAxes, color="#64748B", clip_on=False)
    bottom.plot((-break_size, +break_size), (1 - break_size, 1 + break_size), transform=bottom.transAxes, color="#64748B", clip_on=False)

    counts = [int(data["condition"].eq(condition).sum()) for condition in conditions]
    bottom.set_xticks(
        np.arange(3),
        [
            f"exp117 · 8 epochs\nn={counts[0]}",
            f"exp117 · 16 epochs\nn={counts[1]}",
            f"exp166 · 8 + 8 augmented\nn={counts[2]} attempts / 6 trials",
        ],
    )
    fig.text(0.025, 0.48, "Final validation loss", rotation=90, va="center", fontsize=10.5)
    fig.suptitle("Validation-loss distributions across 8- and 16-epoch training", x=0.03, y=0.96, ha="left", fontsize=19)
    fig.text(
        0.03,
        0.885,
        "Every finished exp117 run is shown. Exp166 includes every finished warm-started regional attempt; stars mark the best observed run in each group.",
        color="#475569",
        fontsize=10.1,
    )
    save_figure(fig, "exp166_epoch_distributions")


def load_history_data() -> pd.DataFrame:
    """Load validation trajectories and normalize boolean flags."""
    data = pd.read_csv(HISTORY_CSV)
    for column in ("is_derived_anchor", "is_highlighted_best"):
        data[column] = data[column].astype(str).str.lower().eq("true")
    highlighted_ids = set(data.loc[data["is_highlighted_best"], "run_id"])
    if highlighted_ids != {HIGHLIGHT_RUN_ID}:
        raise ValueError(f"expected highlighted trajectory {HIGHLIGHT_RUN_ID!r}, found {sorted(highlighted_ids)}")
    return data


def plot_validation_trajectories(data: pd.DataFrame) -> None:
    """Plot validation trajectories for all six matched configurations."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.785, bottom=0.12, wspace=0.10, hspace=0.28)
    styles = {
        "exp117_e8": {"color": COLORS["exp117"], "linestyle": "--", "linewidth": 1.4, "marker": "o", "markersize": 2.8},
        "exp166_scratch": {"color": COLORS["scratch"], "linestyle": "-", "linewidth": 1.8, "marker": "o", "markersize": 2.8},
        "exp166_continued": {"color": COLORS["continued"], "linestyle": "-", "linewidth": 1.8, "marker": "o", "markersize": 2.8},
        "exp117_e16": {"color": COLORS["e16"], "linestyle": ":", "linewidth": 1.5, "marker": "s", "markersize": 2.5},
    }

    for ax, rank in zip(axes.flat, sorted(data["config_rank"].unique()), strict=True):
        facet = data.loc[data["config_rank"].eq(rank)]
        ax.axvspan(8, 16, color="#F5F3FF", alpha=0.65, zorder=0)
        ax.axvline(8, color="#A78BFA", linewidth=0.8, zorder=1)
        run_groups = facet[["condition", "run_id", "is_highlighted_best"]].drop_duplicates()
        run_groups = run_groups.sort_values("is_highlighted_best")
        for group in run_groups.itertuples(index=False):
            line = facet.loc[
                facet["condition"].eq(group.condition) & facet["run_id"].eq(group.run_id)
            ].sort_values("effective_epoch")
            style = styles[str(group.condition)].copy()
            if bool(group.is_highlighted_best):
                style = {
                    "color": COLORS["highlight"],
                    "linestyle": "-",
                    "linewidth": 2.4,
                    "marker": "o",
                    "markersize": 3.1,
                }
            plotted_loss = np.minimum(line["val_loss"].to_numpy(dtype=float), TRAJECTORY_CLIP_Y)
            ax.plot(
                line["effective_epoch"],
                plotted_loss,
                **style,
                alpha=0.96 if bool(group.is_highlighted_best) else 0.88,
                zorder=5 if bool(group.is_highlighted_best) else 2,
            )
            clipped = line.loc[line["val_loss"].gt(TRAJECTORY_AXIS_MAX)]
            if not clipped.empty:
                ax.scatter(
                    clipped["effective_epoch"],
                    np.full(len(clipped), TRAJECTORY_CLIP_Y),
                    marker="^",
                    s=35,
                    facecolor=style["color"],
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=4,
                )
                peak = clipped.loc[clipped["val_loss"].idxmax()]
                ax.annotate(
                    f"{float(peak['val_loss']):.2f}",
                    (float(peak["effective_epoch"]), TRAJECTORY_CLIP_Y),
                    xytext=(0, -13),
                    textcoords="offset points",
                    ha="center",
                    va="top",
                    fontsize=7.8,
                    color=style["color"],
                )
            if bool(group.is_highlighted_best):
                final = line.loc[line["effective_epoch"].idxmax()]
                final_y = min(float(final["val_loss"]), TRAJECTORY_CLIP_Y)
                ax.scatter(
                    float(final["effective_epoch"]),
                    final_y,
                    s=115,
                    marker="*",
                    facecolor=COLORS["highlight"],
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=7,
                )
                ax.annotate(
                    f"best {float(final['val_loss']):.3f}",
                    (float(final["effective_epoch"]), final_y),
                    xytext=(-4, 11),
                    textcoords="offset points",
                    ha="right",
                    va="bottom",
                    fontsize=8.2,
                    color="#92400E",
                    fontweight="bold",
                )
        label = str(facet.iloc[0]["config_label"])
        ax.set_title(label, fontsize=10.3, loc="left", pad=6)
        ax.set_xlim(0, 16.2)
        ax.set_ylim(2.62, TRAJECTORY_AXIS_MAX)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_yticks([2.7, 2.9, 3.1, 3.3, 3.5, 3.7])
        ax.grid(color="#CBD5E1", linewidth=0.6, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Effective training epochs")
    for ax in axes[:, 0]:
        ax.set_ylabel("Validation loss")

    legend = [
        Line2D([0], [0], color=COLORS["exp117"], linestyle="--", marker="o", markersize=3, label="exp117 e8 · no augmentation"),
        Line2D([0], [0], color=COLORS["scratch"], linestyle="-", marker="o", markersize=3, label="exp166 scratch · augmentation"),
        Line2D([0], [0], color=COLORS["continued"], linestyle="-", marker="o", markersize=3, label="exp166 after exp117 · augmentation"),
        Line2D([0], [0], color=COLORS["e16"], linestyle=":", marker="s", markersize=3, label="exp117 e16 exact match"),
        Line2D([0], [0], color=COLORS["highlight"], linestyle="-", marker="*", markersize=8, label="Best finished augmented run"),
    ]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.55, 0.845), ncol=5, frameon=False, fontsize=8.9)
    fig.suptitle("Validation trajectories reveal configuration-dependent warm-start behavior", x=0.03, y=0.96, ha="left", fontsize=19)
    fig.text(
        0.03,
        0.89,
        "The purple segment begins at the matched exp117 e8 loss, then shows eight epochs with a fresh optimizer and cosine schedule. Values above 3.72 are capped and labeled.",
        color="#475569",
        fontsize=10.2,
    )
    save_figure(fig, "exp166_validation_trajectories")


def print_summary(data: pd.DataFrame) -> None:
    """Print the small-sample descriptive results used in the narrative."""
    for condition in ("exp166_scratch", "exp166_continued"):
        subset = best_attempts_by_config(data, condition)
        deltas = subset["delta_vs_exp117_e8"].to_numpy(dtype=float)
        result = wilcoxon(deltas, method="exact")
        print(
            f"{condition}: {int(np.sum(deltas < 0))}/6 lower; "
            f"mean delta={np.mean(deltas):+.4f}; median delta={np.median(deltas):+.4f}; "
            f"paired Wilcoxon p={result.pvalue:.4f}"
        )


def main() -> int:
    final_data = load_final_data()
    history_data = load_history_data()
    distribution_data = build_epoch_distribution_data(final_data)
    plot_augmentation_schematic()
    plot_matched_final_losses(final_data)
    plot_epoch_distributions(distribution_data)
    plot_validation_trajectories(history_data)
    print_summary(final_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
