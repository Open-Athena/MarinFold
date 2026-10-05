# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot monomer contact R-precision, exp343 against the previous model exp277.

Two panels of the same measure rather than one crowded axis: absolute
R-precision, and the **paired** per-protein difference. The paired panel is the
one that decides the question — proteins differ enormously in difficulty, so
comparing two models on the same protein removes the variance that dominates the
absolute numbers. Its interval is what says whether the gap is real.

Reads only committed CSVs: this experiment's `contact_precision_all.csv` and
`exp277_comparison.csv`, plus exp277's committed per-protein file.

    uv run python plot_monomer_rprecision.py
"""

import argparse
import csv
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from build_summary import save_plot_with_meta  # noqa: E402

HERE = Path(__file__).resolve().parent
EXP343 = HERE / "data" / "eval_rollout_v2"
EXP277 = HERE.parent / "exp277_models_single_mpnn_pilot" / "data" / "eval_rollout_v2"

#: eval-val first: it is the natural-protein working set and the number to lead
#: with. legacy 554 is for placing the checkpoint against earlier generations,
#: and eval-denovo is 19 designs — a sanity check, not a benchmark.
SUBSETS = (
    ("eval-val", "eval-val\n97 natural"),
    ("eval-denovo", "eval-denovo\n19 designs"),
    ("legacy_554", "legacy 554\nhistorical"),
)
RANGES = (("all", "All contacts"), ("long", "Long-range contacts"))
#: MarinFold figure convention: new model crimson, previous navy. exp277's exact
#: navy fails the chroma floor (reads gray); this is the same hue stepped up.
#: The pair passes lightness, chroma, CVD separation, and surface contrast.
NEW, OLD = "#b83c54", "#2b5fa8"
INK, MUTED, GRID = "#1a1a1a", "#52514e", "#d8d8d6"
RESAMPLES, SEED = 10_000, 343


def read_pairs(distance_range: str):
    """`{subset: [(exp343, exp277), ...]}` for one distance range."""
    exp343 = {}
    with (EXP343 / "contact_precision_all.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if row["cut"] == "R" and row["range"] == distance_range and row["precision"]:
                exp343[(row["dataset"], row["stem"])] = float(row["precision"])
    pairs: dict[str, list[tuple[float, float]]] = {}
    with (EXP277 / "paired_r_precision.csv").open(newline="") as source:
        for row in csv.DictReader(source):
            if row["range"] != distance_range or not row["precision_exp277"]:
                continue
            key = (row["dataset"], row["stem"])
            if key in exp343:
                pairs.setdefault(row["subset"], []).append(
                    (exp343[key], float(row["precision_exp277"]))
                )
    return pairs


def bootstrap_mean(values: list[float], seed: int) -> tuple[float, float]:
    """Percentile 95% interval for a mean, by protein resampling."""
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(RESAMPLES)
    )
    return means[int(0.025 * (RESAMPLES - 1))], means[int(0.975 * (RESAMPLES - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="plots/monomer_rprecision.png")
    arguments = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    for axis, (distance_range, range_title) in zip(axes, RANGES, strict=True):
        pairs = read_pairs(distance_range)
        positions = range(len(SUBSETS))
        height = 0.34
        for offset, (colour, index, name) in (
            (-height / 2, (NEW, 0, "exp343 (complex corpus)")),
            (+height / 2, (OLD, 1, "exp277 (previous default)")),
        ):
            means, errors = [], [[], []]
            for subset, _ in SUBSETS:
                values = [pair[index] for pair in pairs[subset]]
                mean = sum(values) / len(values)
                low, high = bootstrap_mean(values, SEED)
                means.append(mean)
                errors[0].append(mean - low)
                errors[1].append(high - mean)
            axis.barh([p - offset for p in positions], means, height=height * 0.86,
                      color=colour, label=name, zorder=3)
            axis.errorbar(means, [p - offset for p in positions], xerr=errors,
                          fmt="none", ecolor="#333", elinewidth=1.1, capsize=2.5,
                          zorder=4)
        # The paired delta is the statistic that answers the question; print it
        # beside each pair rather than leaving the reader to subtract two bars
        # whose own intervals overlap.
        for position, (subset, _) in zip(positions, SUBSETS, strict=True):
            deltas = [a - b for a, b in pairs[subset]]
            mean = sum(deltas) / len(deltas)
            low, high = bootstrap_mean(deltas, SEED)
            solid = not (low <= 0.0 <= high)
            # In the clear space to the right of every bar, in text ink -- never
            # over a fill, where it is unreadable at any colour.
            axis.text(0.985, position, f"Δ {mean:+.4f}" + ("" if solid else "  n.s."),
                      transform=axis.get_yaxis_transform(), va="center", ha="right",
                      fontsize=8.5, color=INK if solid else MUTED, zorder=5)
        axis.set_yticks(list(positions))
        axis.set_yticklabels([label for _, label in SUBSETS], fontsize=9, color=INK)
        axis.set_ylim(len(SUBSETS) - 0.5, -0.5)
        axis.set_xlim(0, 0.92)
        axis.set_xlabel("R-precision — higher is better", fontsize=9, color=INK)
        axis.set_title(range_title, fontsize=10, color=INK, loc="left")
        axis.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        axis.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(axis="x", colors=MUTED, labelsize=8.5)
        axis.tick_params(axis="y", length=0)
    axes[1].set_yticklabels([])
    # Legend above the panels: inside, it sat on top of the legacy-554 bars.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, labelcolor=MUTED,
               loc="upper right", bbox_to_anchor=(0.995, 0.99), ncol=2)
    fig.suptitle(
        "Monomer contact R-precision: exp343 is lower on every set\n"
        "five of six paired differences exclude zero",
        fontsize=11, color=INK, x=0.012, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    caption = (
        "Monomer contact R-precision under the fixed exp82 rollout recipe, 100 rollouts "
        "per unit. Bars are 95% protein-bootstrap intervals; the printed paired delta is "
        "the per-protein difference, which is the statistic that decides the comparison. "
        "n.s. = its interval covers zero."
    )
    save_plot_with_meta(fig, HERE / arguments.out, caption=caption, dpi=200)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
