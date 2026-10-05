# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot held-out complex loss by token role, exp343 against exp277.

The whole reason this experiment holds a shard out is that a monomer benchmark
cannot see whether a complex was modelled. This figure is that measurement, and
the split by token role is what keeps it honest: a model can score well on a
complex document by predicting only the intra-chain contacts it already knows.

Reads `data/complex_loss/aggregate.csv`, which carries one row per (label, group).
Plots whichever labels are present, so it renders with the exp277 control alone
and gains exp343's series without a rewrite.

    uv run python plot_complex_loss.py
    uv run python plot_complex_loss.py --group source_arm=pinder
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

HERE = Path(__file__).resolve().parent
AGGREGATE = HERE / "data" / "complex_loss" / "aggregate.csv"

#: The roles worth showing, in document order, with the labels a reader needs.
#: `header` and `end` are two tokens per document and carry no signal.
ROLES = (
    ("sequence", "Sequence section"),
    ("terminus", "Chain termini"),
    ("contact_intra", "Intra-chain contacts"),
    ("contact_inter", "Inter-chain contacts"),
)
#: Repo convention across MarinFold figures: the new model in crimson, the
#: previous one in navy (exp277's `plot_results.py`). exp277's exact navy
#: `#365b8c` fails a chroma floor -- it reads gray next to a gray reference mark
#: -- so this is the same navy stepped up in chroma. The pair validates on
#: lightness, chroma, CVD separation (protan dE 13.7), normal-vision separation
#: and surface contrast.
SERIES = {
    "exp343": "#b83c54",
    "exp277": "#2b5fa8",
}
OTHER = "#8a8a8a"
INK = "#1a1a1a"
MUTED = "#52514e"
GRID = "#d8d8d6"
#: A uniform distribution over the contacts-v1 vocabulary. A role scoring above
#: this line is not merely uncertain -- the model is confidently wrong there.
VOCAB_SIZE = 2845
UNIFORM = math.log(VOCAB_SIZE)


def read_aggregate(path: Path, group: str) -> dict[str, dict[str, float]]:
    """`{label: {role: nll_per_token}}` for one group, plus the overall value."""
    series: dict[str, dict[str, float]] = {}
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            if row["group"] != group:
                continue
            values = {"overall": float(row["nll_per_token"])}
            for role, _ in ROLES:
                raw = row.get(f"nll_per_token_{role}")
                if raw:
                    values[role] = float(raw)
            series[row["label"]] = values
    if not series:
        raise ValueError(f"{path} has no rows for group {group!r}")
    return series


def short(label: str) -> str:
    """`exp277-step266344` -> `exp277`, for the legend and colour lookup."""
    return label.split("-", 1)[0]


def plot(series: dict[str, dict[str, float]], group: str, documents: str) -> Path:
    order = [key for key, _ in ROLES]
    labels = sorted(series, key=lambda name: short(name) != "exp343")
    # Thin marks: with one series a full-width bar is a slab, so cap it and let
    # the group share the slot as more series arrive.
    height = min(0.30, 0.72 / max(len(labels), 1))
    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    positions = range(len(order))
    for index, label in enumerate(labels):
        colour = SERIES.get(short(label), OTHER)
        offset = (index - (len(labels) - 1) / 2) * height
        values = [series[label].get(role, 0.0) for role in order]
        # A 2px surface gap between adjacent bars, per the mark spec.
        bars = ax.barh(
            [position - offset for position in positions],
            values,
            height=height * 0.88,
            color=colour,
            label=f"{short(label)} ({series[label]['overall']:.3f} overall)",
            zorder=3,
        )
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value + 0.12,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                ha="left",
                fontsize=8.5,
                color=MUTED,
            )
    ax.axvline(UNIFORM, color=OTHER, lw=1, ls="--", zorder=2)
    ax.text(
        UNIFORM - 0.15,
        -0.62,
        f"uniform over {VOCAB_SIZE:,} tokens",
        fontsize=8,
        color=MUTED,
        va="center",
        ha="right",
    )
    ax.set_ylim(len(order) - 0.55, -0.95)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([name for _, name in ROLES], fontsize=9.5, color=INK)
    ax.set_xlabel("Held-out complex loss (nats / token) — lower is better", fontsize=9.5,
                  color=INK)
    ax.set_xlim(0, max(UNIFORM, *(v for s in series.values() for v in s.values())) + 1.4)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
    ax.tick_params(axis="y", length=0)
    # With one series there is no legend: the subtitle names the model, as it must
    # when colour carries no identity.
    if len(labels) > 1:
        ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=MUTED)
    named = ", ".join(
        f"{short(label)} {series[label]['overall']:.3f} overall"
        for label in labels
    )
    scope = "all complexes" if group == "all" else group
    ax.set_title(
        "Held-out complex loss, by what each token encodes\n"
        f"{named} · {documents} documents, {scope}",
        fontsize=10.5,
        color=INK,
        loc="left",
    )
    fig.tight_layout()
    # Two lines, plain text. `build_summary.py` does not wrap a longer caption
    # away from the plot -- a third line silently overlaps the figure title --
    # and it renders no markdown, so asterisks would print literally. The full
    # reading notes live in the README.
    caption = (
        f"Held-out complex loss by token role, {documents} documents of the #294 "
        f"corpus ({group}). Compare within a role, not across: roles differ in "
        "intrinsic difficulty. Dashed line is uniform over the 2,845-token vocabulary."
    )
    return save_plot_with_meta(
        fig, HERE / "plots" / "complex_loss_by_role.png", caption=caption, dpi=200
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="all")
    parser.add_argument("--aggregate", type=Path, default=AGGREGATE)
    arguments = parser.parse_args()
    series = read_aggregate(arguments.aggregate, arguments.group)
    with arguments.aggregate.open(newline="") as source:
        documents = next(
            row["documents"]
            for row in csv.DictReader(source)
            if row["group"] == arguments.group
        )
    path = plot(series, arguments.group, f"{int(documents):,}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
