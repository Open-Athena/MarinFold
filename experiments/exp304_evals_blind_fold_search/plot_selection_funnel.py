#!/usr/bin/env python
"""Plot the auditable selection funnel for the 29-protein test cohort."""

import json
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from build_summary import save_plot_with_meta

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "exp301_evals_fold_switching_proteins" / "data"
DATA = HERE / "data"
OUTPUT = HERE / "plots" / "selection_funnel.png"

NAVY = "#17324d"
BLUE = "#397eb8"
LIGHT_BLUE = "#d9eaf5"
GREEN = "#25845b"
LIGHT_GREEN = "#dcefe5"
GRAY = "#66717c"
LIGHT_GRAY = "#edf0f2"
ORANGE = "#c66b22"


def load_membership() -> pd.DataFrame:
    """Reconstruct every funnel stage from the frozen source manifests."""
    gate = pd.read_csv(SOURCE / "premise_gate.csv")
    cohort = pd.read_csv(DATA / "cohort.csv").set_index("pair_id")
    universe = pd.DataFrame(
        [
            json.loads(line)
            for line in (SOURCE / "foldswitch_universe.jsonl").read_text().splitlines()
        ]
    ).set_index("pair_id")
    frame = gate[["pair_id", "passes_gate", "reason"]].copy().set_index("pair_id")
    frame["pair_identity"] = universe.pair_identity
    frame["n_seq_mismatch"] = universe.n_seq_mismatch
    identical = (
        frame.index.to_series()
        .map(cohort.seq_class.eq("identical"))
        .fillna(False)
        .astype(bool)
    )
    enough_fold1 = frame.index.to_series().map(cohort.n_a_fs.ge(10)).fillna(False).astype(bool)
    enough_fold2 = frame.index.to_series().map(cohort.n_b_fs.ge(10)).fillna(False).astype(bool)
    budget_capped = frame.index.to_series().map(cohort.budget_capped).fillna(False).astype(bool)
    frame["near_identical"] = frame.passes_gate & identical
    frame["region_contact_signal"] = frame.near_identical & (
        enough_fold1 & enough_fold2
    )
    frame["uncapped_primary"] = frame.region_contact_signal & ~budget_capped
    frame["split"] = frame.index.to_series().map(cohort.split)
    frame["final_test"] = frame.uncapped_primary & frame.split.eq("test")
    frame["development"] = frame.uncapped_primary & frame.split.eq("dev")
    frame["literal_exact"] = frame.n_seq_mismatch.eq(0)
    frame = frame.reset_index()

    counts = {
        "source": len(frame),
        "premise": int(frame.passes_gate.sum()),
        "identity": int(frame.near_identical.sum()),
        "region": int(frame.region_contact_signal.sum()),
        "primary": int(frame.uncapped_primary.sum()),
        "dev": int(frame.development.sum()),
        "test": int(frame.final_test.sum()),
        "test_exact": int((frame.final_test & frame.literal_exact).sum()),
    }
    expected = {
        "source": 93,
        "premise": 68,
        "identity": 65,
        "region": 45,
        "primary": 44,
        "dev": 15,
        "test": 29,
        "test_exact": 17,
    }
    if counts != expected:
        raise ValueError(f"selection funnel changed: {counts} != {expected}")
    return frame


def write_tables(frame: pd.DataFrame) -> None:
    """Save pair-level membership and stage counts behind the figure."""
    frame.sort_values("pair_id").to_csv(DATA / "selection_funnel_membership.csv", index=False)
    stages = [
        ("source_table_s1", 93, 0, "Literature-curated AF2_benchmark Table S1 pairs"),
        ("premise_valid", 68, 25, "Global contact/coverage/region premise gate"),
        ("near_identical", 65, 3, "At least 98% pairwise sequence identity"),
        (
            "region_contact_signal",
            45,
            20,
            "At least 10 Fold1-only and 10 Fold2-only contacts touching the switch region",
        ),
        ("uncapped_primary", 44, 1, "Exclude context-capped 4zt0c_4cmqb"),
        ("development", 15, 0, "Sequence-group development split"),
        ("held_out_test", 29, 0, "Sequence-group held-out test split used here"),
    ]
    pd.DataFrame(stages, columns=["stage", "n_pairs", "excluded_from_previous", "definition"]).to_csv(
        DATA / "selection_funnel_summary.csv", index=False
    )


def add_stage(
    axis,
    y: float,
    count: int,
    title: str,
    subtitle: str,
    color: str,
    text_color: str = "white",
) -> None:
    """Draw one count-scaled rounded stage box."""
    center, height = 4.1, 0.73
    width = 6.2 * (count / 93) ** 0.55
    patch = FancyBboxPatch(
        (center - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=color, edgecolor="none",
    )
    axis.add_patch(patch)
    axis.text(
        center - width / 2 + 0.24,
        y + 0.08,
        f"{count}",
        color=text_color,
        fontsize=20,
        fontweight="bold",
        va="center",
    )
    axis.text(
        center - width / 2 + 1.05,
        y + 0.11,
        title,
        color=text_color,
        fontsize=11.5,
        fontweight="bold",
        va="center",
    )
    axis.text(
        center - width / 2 + 1.05,
        y - 0.17,
        subtitle,
        color=text_color,
        fontsize=8.4,
        va="center",
    )


def add_arrow(axis, y_top: float, y_bottom: float) -> None:
    axis.add_patch(FancyArrowPatch(
        (4.1, y_top), (4.1, y_bottom), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.2, color="#98a2ad",
    ))


def main() -> None:
    """Build the selection tables and funnel figure."""
    frame = load_membership()
    write_tables(frame)

    fig, axis = plt.subplots(figsize=(13.33, 7.5))
    axis.set_xlim(0, 13.33)
    axis.set_ylim(0, 7.5)
    axis.axis("off")
    fig.patch.set_facecolor("white")

    axis.text(0.55, 7.12, "How the 29-protein fold-switching test set was selected",
              fontsize=22, fontweight="bold", color=NAVY, va="top")
    axis.text(
        0.55, 6.76,
        "Cohort eligibility and split, frozen before exp304 held-out accuracy was inspected",
        fontsize=11.5, color=GRAY, va="top",
    )

    stages = [
        (6.13, 93, "SOURCE PAIRS", "NCBI AF2 benchmark Table S1", NAVY),
        (5.05, 68, "PREMISE-VALID PAIRS", "contact distinction, coverage, region annotation", BLUE),
        (3.97, 65, "NEAR-IDENTICAL SEQUENCES", "pairwise identity ≥98%", "#4f91bf"),
        (2.89, 45, "REGION-SPECIFIC SIGNAL", "≥10 fold-specific contacts on each side", "#62a3c8"),
        (1.81, 44, "PRIMARY COHORT", "uncapped; development/test eligible", "#458b78"),
    ]
    for stage in stages:
        add_stage(axis, *stage)
    for upper, lower in pairwise(stages):
        add_arrow(axis, upper[0] - 0.39, lower[0] + 0.39)

    axis.add_patch(FancyArrowPatch(
        (4.1, 1.40), (2.75, 0.89), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.2, color="#98a2ad", connectionstyle="arc3,rad=0.08",
    ))
    axis.add_patch(FancyArrowPatch(
        (4.1, 1.40), (5.45, 0.89), arrowstyle="-|>", mutation_scale=11,
        linewidth=1.2, color="#98a2ad", connectionstyle="arc3,rad=-0.08",
    ))
    for x, width, count, title, subtitle, color in (
        (1.50, 2.50, 15, "DEVELOPMENT", "global settings only", LIGHT_GRAY),
        (4.18, 2.95, 29, "HELD-OUT TEST", "the structural testbed", LIGHT_GREEN),
    ):
        box = FancyBboxPatch(
            (x, 0.20), width, 0.70, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=color, edgecolor=GREEN if count == 29 else "#aab2ba", linewidth=1.7,
        )
        axis.add_patch(box)
        axis.text(x + 0.20, 0.61, str(count), fontsize=19, fontweight="bold",
                  color=GREEN if count == 29 else GRAY, va="center")
        axis.text(x + 0.82, 0.65, title, fontsize=10.5, fontweight="bold",
                  color=GREEN if count == 29 else GRAY, va="center")
        axis.text(x + 0.82, 0.39, subtitle, fontsize=8.5, color=GRAY, va="center")

    axis.plot([8.15, 8.15], [0.35, 6.38], color="#d6dbe0", linewidth=1)
    axis.text(8.50, 6.34, "EXCLUSIONS AND SPLIT", fontsize=11.5, fontweight="bold", color=NAVY)
    notes = [
        (
            5.69,
            "−25 premise failures",
            (
                "17: <10 global unique contacts on one side\n"
                "14: <50% common resolved-chain coverage\n3: switch region not located\n"
                "Reason counts overlap."
            ),
        ),
        (4.60, "−3 sequence variants", "Below 98% identity; retained in secondary analyses."),
        (
            3.52,
            "−20 weak regional signal",
            (
                "Fewer than 10 fold-specific contacts touching the\n"
                "annotated switching region for at least one fold."
            ),
        ),
        (
            2.44,
            "−1 context-capped",
            (
                "4zt0c_4cmqb, L=1,338; only 6% of rollouts finished\n"
                "under the 8,192-token context budget."
            ),
        ),
        (
            1.34,
            "Sequence-group holdout",
            (
                "43 groups at ≥30% identity and ≥50% either-direction coverage.\n"
                "SHA-256 group order: 15 development, 29 test; no group crosses."
            ),
        ),
    ]
    for y, heading, body in notes:
        axis.text(8.50, y, heading, fontsize=10.5, fontweight="bold",
                  color=ORANGE if heading.startswith("−") else NAVY, va="top")
        axis.text(8.50, y - 0.25, body, fontsize=8.7, color=GRAY, va="top", linespacing=1.22)

    axis.text(
        8.50, 0.48,
        "Final 29: 17 exact-sequence pairs; 12 have 1–5 substitutions.",
        fontsize=9.2, fontweight="bold", color=GREEN, va="top",
    )
    axis.text(
        8.50, 0.18,
        "No fold-match, Helico, GDT, RMSD, or TM-score outcome entered selection.",
        fontsize=8.6, color=GRAY, va="top",
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    save_plot_with_meta(
        fig,
        OUTPUT,
        caption=(
            "Selection of the 29-pair held-out fold-switching testbed from the "
            "93 literature-curated AF2 benchmark reference pairs. Counts are "
            "reconstructed from the exp301 premise gate and the frozen exp304 cohort."
        ),
        dpi=180,
        facecolor="white",
    )
    plt.close(fig)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
