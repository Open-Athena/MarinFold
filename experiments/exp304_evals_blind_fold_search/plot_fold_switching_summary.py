#!/usr/bin/env python
"""Build one figure summarizing the exp304 fold-switching results."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE / "plots" / "fold_switching_summary.png"

NAVY = "#17324d"
BLUE = "#397eb8"
LIGHT_BLUE = "#dcebf5"
ORANGE = "#c66b22"
LIGHT_ORANGE = "#f7e5d4"
GREEN = "#21855b"
LIGHT_GREEN = "#dcefe5"
PURPLE = "#646e9f"
GRAY = "#66717c"
LIGHT_GRAY = "#edf0f2"
GRID = "#d8dde2"


def load_results() -> dict[str, pd.DataFrame]:
    """Load and validate the saved result tables used in the figure."""
    tables = {
        "funnel": pd.read_csv(DATA / "selection_funnel_summary.csv"),
        "search": pd.read_csv(DATA / "summary.csv"),
        "contact_curve": pd.read_csv(DATA / "iid1000_primary_test_curve.csv"),
        "structural_curve": pd.read_csv(DATA / "helico_iid_structural_curve.csv"),
        "structural_per_protein": pd.read_csv(DATA / "helico_iid_structural_per_protein.csv"),
        "selected": pd.read_csv(DATA / "helico_iid_selected_rollouts.csv"),
        "controls": pd.read_csv(DATA / "helico_iid_control_scores.csv"),
    }

    funnel_counts = tables["funnel"].set_index("stage").n_pairs.to_dict()
    expected_funnel = {
        "source_table_s1": 93,
        "premise_valid": 68,
        "near_identical": 65,
        "region_contact_signal": 45,
        "uncapped_primary": 44,
        "development": 15,
        "held_out_test": 29,
    }
    if funnel_counts != expected_funnel:
        raise ValueError(f"unexpected cohort funnel: {funnel_counts}")

    search = tables["search"].set_index("method")
    branch10 = search.loc["branch10"]
    if not (
        round(branch10.paired_delta_vs_iid, 3) == 0.013
        and round(branch10.paired_delta_lo, 3) == -0.014
        and round(branch10.paired_delta_hi, 3) == 0.043
    ):
        raise ValueError("primary search comparison changed")

    budgets = [100, 200, 500, 750, 1000]
    contact = tables["contact_curve"].set_index("budget").loc[budgets]
    structural = tables["structural_curve"].set_index("budget").loc[budgets]
    if contact.both.tolist() != [6, 9, 9, 11, 11]:
        raise ValueError("contact coverage curve changed")
    if structural.both.tolist() != [0, 1, 2, 2, 2]:
        raise ValueError("structural coverage curve changed")

    per_protein = tables["structural_per_protein"]
    both_controls = per_protein.true_fold1_control_pass & per_protein.true_fold2_control_pass
    if int(both_controls.sum()) != 17:
        raise ValueError("true-contact control gate count changed")
    dual_ids = set(per_protein.loc[per_protein.state_1000.eq(2), "pair_id"])
    if dual_ids != {"2lela_2k0qa", "3j7wb_3j7vg"}:
        raise ValueError(f"strict structural dual set changed: {dual_ids}")
    return tables


def style_panel(axis, title: str) -> None:
    """Apply common framing to one panel."""
    axis.set_facecolor("white")
    for spine in axis.spines.values():
        spine.set_color("#cfd5da")
        spine.set_linewidth(1.0)
    axis.set_title(title, loc="left", fontsize=15, fontweight="bold", color=NAVY, pad=12)


def rounded_box(axis, xy: tuple[float, float], width: float, height: float, color: str) -> None:
    """Draw a rounded box in axes coordinates."""
    axis.add_patch(
        FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=color,
            edgecolor="none",
            transform=axis.transAxes,
        )
    )


def plot_design(axis, tables: dict[str, pd.DataFrame]) -> None:
    """Summarize cohort selection and structural evaluation scale."""
    style_panel(axis, "A  Cohort and evaluation scale")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])

    funnel = tables["funnel"].set_index("stage").n_pairs
    stage_specs = [
        ("source_table_s1", "source", NAVY),
        ("premise_valid", "premise", BLUE),
        ("near_identical", "≥98% ID", "#4f91bf"),
        ("region_contact_signal", "region signal", "#62a3c8"),
        ("uncapped_primary", "primary", "#458b78"),
        ("held_out_test", "test", GREEN),
    ]
    left, width, gap, y, height = 0.035, 0.125, 0.032, 0.69, 0.19
    for index, (stage, label, color) in enumerate(stage_specs):
        x = left + index * (width + gap)
        rounded_box(axis, (x, y), width, height, color)
        axis.text(
            x + width / 2,
            y + 0.115,
            str(int(funnel.loc[stage])),
            ha="center",
            va="center",
            color="white",
            fontsize=19,
            fontweight="bold",
            transform=axis.transAxes,
        )
        axis.text(
            x + width / 2,
            y + 0.045,
            label,
            ha="center",
            va="center",
            color="white",
            fontsize=8.5,
            transform=axis.transAxes,
        )
        if index < len(stage_specs) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (x + width + 0.003, y + height / 2),
                    (x + width + gap - 0.003, y + height / 2),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    color="#9ca7b1",
                    linewidth=1.2,
                    transform=axis.transAxes,
                )
            )
    axis.text(
        0.5,
        0.63,
        "Eligibility gates → deterministic sequence-group holdout",
        ha="center",
        va="center",
        fontsize=9.5,
        color=GRAY,
        transform=axis.transAxes,
    )

    compute_specs = [
        (0.055, LIGHT_BLUE, "29,000", "MarinFold maps", "1,000 / test case"),
        (0.375, LIGHT_ORANGE, "29,087", "Helico structures", "+87 true/no-contact controls"),
        (0.695, LIGHT_GREEN, "17 / 29", "both controls pass", "structurally assessable"),
    ]
    for index, (x, color, count, label, detail) in enumerate(compute_specs):
        rounded_box(axis, (x, 0.13), 0.25, 0.34, color)
        axis.text(
            x + 0.125,
            0.36,
            count,
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color=NAVY if index < 2 else GREEN,
            transform=axis.transAxes,
        )
        axis.text(
            x + 0.125,
            0.275,
            label,
            ha="center",
            va="center",
            fontsize=10.2,
            fontweight="bold",
            color=NAVY,
            transform=axis.transAxes,
        )
        axis.text(
            x + 0.125,
            0.195,
            detail,
            ha="center",
            va="center",
            fontsize=8.5,
            color=GRAY,
            transform=axis.transAxes,
        )
        if index < len(compute_specs) - 1:
            axis.add_patch(
                FancyArrowPatch(
                    (x + 0.255, 0.30),
                    (x + 0.315, 0.30),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    color="#9ca7b1",
                    linewidth=1.2,
                    transform=axis.transAxes,
                )
            )


def plot_search_comparison(axis, tables: dict[str, pd.DataFrame]) -> None:
    """Plot paired enrichment differences for search variants versus iid."""
    style_panel(axis, "B  Engineered search did not beat iid")
    search = tables["search"].set_index("method")
    order = ["temp", "random", "branch5", "branch10", "branch20"]
    labels = ["Higher temp†", "Random self-seed", "Cluster branch 5", "Cluster branch 10*", "Cluster branch 20"]
    y_values = list(range(len(order)))[::-1]
    for method, label, y in zip(order, labels, y_values, strict=True):
        row = search.loc[method]
        primary = method == "branch10"
        color = GREEN if primary else BLUE
        axis.hlines(y, row.paired_delta_lo, row.paired_delta_hi, color=color, linewidth=3 if primary else 2)
        axis.scatter(
            row.paired_delta_vs_iid,
            y,
            s=90 if primary else 65,
            color=color,
            edgecolor="white",
            linewidth=1,
            zorder=3,
        )
        axis.text(
            0.062,
            y,
            f"{int(row.dual_contact_hits)}/29 dual",
            ha="right",
            va="center",
            fontsize=8.6,
            color=GREEN if primary else GRAY,
            fontweight="bold" if primary else "normal",
        )
    axis.axvline(0, color="#7f8992", linestyle="--", linewidth=1.2)
    axis.set_xlim(-0.065, 0.065)
    axis.set_ylim(-1.35, len(order) - 0.2)
    axis.set_yticks(y_values, labels)
    axis.tick_params(axis="y", labelsize=9.5)
    axis.set_xlabel(
        "Paired Δ best minority enrichment vs iid (95% bootstrap CI)\n"
        "* prespecified comparison   † unequal compute",
        fontsize=9.3,
    )
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    branch10 = search.loc["branch10"]
    axis.text(
        -0.062,
        -1.12,
        (
            f"Primary comparison: +{branch10.paired_delta_vs_iid:.3f} "
            f"[{branch10.paired_delta_lo:.3f}, +{branch10.paired_delta_hi:.3f}]\n"
            f"Minority recall: iid {search.loc['iid'].mean_minor_recall:.3f}, "
            f"branch 10 {branch10.mean_minor_recall:.3f}"
        ),
        fontsize=9,
        color=NAVY,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT_GREEN, "edgecolor": "none"},
    )


def plot_sampling_curve(axis, tables: dict[str, pd.DataFrame]) -> None:
    """Compare reference-aware contact coverage with reconstructed structures."""
    style_panel(axis, "C  More iid draws: contact gains exceed 3D gains")
    budgets = [100, 200, 500, 750, 1000]
    contact = tables["contact_curve"].set_index("budget").loc[budgets, "both"]
    structural = tables["structural_curve"].set_index("budget").loc[budgets, "both"]
    axis.plot(
        budgets,
        contact,
        color=BLUE,
        linewidth=2.8,
        marker="o",
        markersize=7,
        label="Contact-screen oracle",
    )
    axis.plot(
        budgets,
        structural,
        color=GREEN,
        linewidth=2.8,
        marker="o",
        markersize=7,
        label="Strict reconstructed structures",
    )
    for x, value in contact.items():
        axis.text(x, value + 0.45, str(int(value)), color=BLUE, ha="center", fontsize=9, fontweight="bold")
    for x, value in structural.items():
        axis.text(
            x,
            value - 0.62 if value else value + 0.45,
            str(int(value)),
            color=GREEN,
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axis.axvline(500, color="#8a939b", linewidth=1.1, linestyle="--")
    axis.text(510, 11.5, "500 draws", color=GRAY, fontsize=8.5, va="top")
    axis.set_xlim(60, 1040)
    axis.set_ylim(-0.9, 12.5)
    axis.set_xticks(budgets)
    axis.set_yticks(range(0, 13, 2))
    axis.set_xlabel("Independent rollouts per test case", fontsize=10)
    axis.set_ylabel("Cases with both modes (out of 29)", fontsize=10)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.legend(frameon=False, loc="upper left", fontsize=9.5)
    axis.text(
        0.98,
        0.06,
        "17/29 pass both true-contact controls\nNo new strict 3D dual after draw 249",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color=NAVY,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT_GRAY, "edgecolor": "none"},
    )


def plot_exemplar(axis, tables: dict[str, pd.DataFrame]) -> None:
    """Show whole-structure TM-scores for the cleanest dual-fold example."""
    style_panel(axis, "D  Cleanest example: 3j7w / 3j7v")
    pair_id = "3j7wb_3j7vg"
    selected = tables["selected"].loc[tables["selected"].pair_id.eq(pair_id)].set_index("fold")
    controls = tables["controls"].loc[tables["controls"].pair_id.eq(pair_id)].set_index("kind")
    if set(selected.index) != {1, 2} or not selected.passes_primary.all():
        raise ValueError("3j7 selected structural candidates changed")
    required_controls = {"true_fold1", "true_fold2", "no_contacts"}
    if not required_controls.issubset(controls.index):
        raise ValueError("3j7 structural controls are incomplete")

    axis.plot([0.2, 0.86], [0.2, 0.86], color="#9aa3ab", linestyle="--", linewidth=1)
    points = [
        (
            selected.loc[1, "tm_common_fold1"],
            selected.loc[1, "tm_common_fold2"],
            BLUE,
            "D",
            "best Fold1 candidate",
            (12, -22),
        ),
        (
            selected.loc[2, "tm_common_fold1"],
            selected.loc[2, "tm_common_fold2"],
            ORANGE,
            "D",
            "best Fold2 candidate",
            (-118, 12),
        ),
        (
            controls.loc["true_fold1", "tm_common_fold1"],
            controls.loc["true_fold1", "tm_common_fold2"],
            BLUE,
            "o",
            "true Fold1 contacts",
            (12, 8),
        ),
        (
            controls.loc["true_fold2", "tm_common_fold1"],
            controls.loc["true_fold2", "tm_common_fold2"],
            ORANGE,
            "o",
            "true Fold2 contacts",
            (-122, -22),
        ),
    ]
    for x, y, color, marker, label, offset in points:
        open_control = marker == "o"
        axis.scatter(
            x,
            y,
            s=105 if marker == "D" else 80,
            marker=marker,
            facecolor="white" if open_control else color,
            edgecolor=color,
            linewidth=2,
            zorder=3,
        )
        axis.annotate(
            label,
            (x, y),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.2,
            color=NAVY if marker == "D" else GRAY,
            arrowprops={"arrowstyle": "-", "color": "#a7afb6", "linewidth": 0.8},
        )
    no_contact = controls.loc["no_contacts"]
    axis.scatter(
        no_contact.tm_common_fold1,
        no_contact.tm_common_fold2,
        marker="X",
        s=85,
        color=GRAY,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    axis.annotate(
        "no contacts",
        (no_contact.tm_common_fold1, no_contact.tm_common_fold2),
        xytext=(10, -16),
        textcoords="offset points",
        fontsize=8.2,
        color=GRAY,
    )
    axis.set_xlim(0.2, 0.86)
    axis.set_ylim(0.2, 0.86)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Whole-structure TM-score to Fold1", fontsize=10)
    axis.set_ylabel("Whole-structure TM-score to Fold2", fontsize=10)
    axis.grid(color=GRID, linewidth=0.7)
    axis.text(0.83, 0.235, "Fold1 favored", color=BLUE, fontsize=8.5, ha="right")
    axis.text(0.235, 0.83, "Fold2 favored", color=ORANGE, fontsize=8.5, va="top", rotation=90)
    axis.text(
        0.98,
        0.05,
        "Strict dual by draw 144\nNo-contact matches neither",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color=NAVY,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": LIGHT_GREEN, "edgecolor": "none"},
    )


def main() -> None:
    """Render the four-panel summary figure from saved result tables."""
    tables = load_results()
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.975,
        top=0.82,
        bottom=0.23,
        wspace=0.20,
        hspace=0.40,
    )
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    plot_design(axes[0], tables)
    plot_search_comparison(axes[1], tables)
    plot_sampling_curve(axes[2], tables)
    plot_exemplar(axes[3], tables)

    fig.text(
        0.055,
        0.965,
        "Fold-switching inference: rare dual-fold recovery, no demonstrated search gain",
        fontsize=23,
        fontweight="bold",
        color=NAVY,
        va="top",
    )
    fig.text(
        0.055,
        0.923,
        "29 sequence-group held-out cases • 1,000 iid maps per case • every map folded individually with Helico",
        fontsize=12,
        color=GRAY,
        va="top",
    )
    fig.text(
        0.055,
        0.185,
        (
            "Contact dual: separate maps reach ≥25% switching-region fold-specific recall and ≥0.10 enrichment. "
            "Strict 3D dual: global GDT-TS ≥ max(0.35, 90% of true-contact control) and region advantage ≥0.10; "
            "this structural cutoff is post-hoc. Panel D uses whole-structure TM-score for an interpretable view."
        ),
        fontsize=8.6,
        color=GRAY,
        va="top",
    )

    fig.patches.append(
        Rectangle(
            (0.04, 0.025),
            0.92,
            0.105,
            transform=fig.transFigure,
            facecolor=NAVY,
            edgecolor="none",
            zorder=-1,
        )
    )
    fig.text(
        0.065,
        0.079,
        "Conclusion",
        color="white",
        fontsize=12,
        fontweight="bold",
        va="center",
    )
    fig.text(
        0.165,
        0.079,
        (
            "IID rollouts can recover both experimental folds, but rarely: 11/29 contact-screen duals → "
            "2/29 strict structural duals → 1 clean no-contact-controlled case.\n"
            "The tested branching search has no demonstrated gain; draws 500–1,000 added no strict 3D duals."
        ),
        color="white",
        fontsize=10.7,
        va="center",
    )

    save_plot_with_meta(
        fig,
        OUTPUT,
        caption=(
            "One-figure summary of the exp304 fold-switching analysis: cohort and evaluation scale, "
            "paired inference-search comparisons, iid contact and structure coverage through 1,000 draws, "
            "and the cleanest dual-fold example in whole-structure TM-score space."
        ),
        dpi=190,
        facecolor="white",
    )
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
