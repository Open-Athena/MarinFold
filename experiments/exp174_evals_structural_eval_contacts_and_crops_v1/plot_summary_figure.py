# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The one figure that summarizes exp174.

Three panels, chosen so each answers a different question and none needs the
others to be legible:

* **(a) Accuracy by condition** — lDDT and TM-score together, with the
  oracle-document ceiling marked. The point of putting them on one axis (both
  are 0–1 fractions, so this is one scale, not a dual axis) is that the *gap
  between the two bars* is the finding: Plan F reaches the ceiling on lDDT and a
  third of it on TM-score, because a local metric rewards a well-refined wrong
  fold and a global one does not.
* **(b) Median CA-RMSD** — the same conditions on a distance scale, where the
  result is starkly bimodal: everything with correct coarse boxes lands at
  ~4 Å, everything working from the model's own boxes at ~16.5 Å, regardless of
  how much inference was spent.
* **(c) lDDT vs chain length** — the single-document ceiling collapses with
  length because the token budget is fixed while the atom count is not; Plan F
  re-prompts and so runs above it on long chains.

Colours are categorical slots 1 and 2 of the validated default palette (blue /
orange), which clear the CVD and normal-vision gates as a pair. Ink stays in
text colours rather than series colours.

Usage::

    uv run python plot_summary_figure.py --scores data/scores_all.csv --out plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

# Categorical slots 1 and 2 of the validated palette; text tokens for ink.
SERIES_1 = "#2a78d6"   # lDDT
SERIES_2 = "#eb6834"   # TM-score
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d8d8d5"

LENGTH_BINS = [0, 100, 200, 400, 10_000]
LENGTH_LABELS = ["≤100", "101–200", "201–400", ">400"]

CEILING = "oracle-doc"

# (run, display label, whether the COARSE BOXES are correct). That flag is the
# real explanatory variable — panel (b) colours by it, because the RMSD split
# tracks it and nothing else.
CONDITIONS = [
    (CEILING, "oracle document\n(format ceiling)", True),
    ("e2-cc1mix5-step50000", "E2 · true Pass-1 boxes", True),
    ("e1b-cc1mix5-step50000", "E1 · true contacts", False),
    ("e1-cc1mix5-step50000", "E1 · true contacts", False),
    ("f-cc1mix5-step50000", "F · iterative refinement", False),
    ("c-cc1mix5-step50000", "C · one forced sweep", False),
    ("a-cc1mix5-step50000", "A · one document", False),
]

# Length curves: the ceiling plus the two ends of the de-novo range.
CURVES = [
    (CEILING, "oracle document", SERIES_1, "--"),
    ("f-cc1mix5-step50000", "F · iterative refinement", SERIES_2, "-"),
    ("a-cc1mix5-step50000", "A · one document", TEXT_SECONDARY, "-"),
]


def present_conditions(scores: pd.DataFrame):
    """Conditions that exist in the data, de-duplicated by display label.

    The corrected E1 (``e1b``) supersedes the first attempt, which sampled its
    50 forced contacts from the unfiltered contact list; whichever is present
    wins, and if both are, the corrected one does.
    """
    have = set(scores.run.unique())
    out, seen = [], set()
    for run, label, boxes_correct in CONDITIONS:
        if run in have and label not in seen:
            out.append((run, label, boxes_correct))
            seen.add(label)
    return out


def _bars(axis, rows, values, colour, offset, height, label):
    positions = [i + offset for i in range(len(rows))]
    axis.barh(positions, values, height=height, color=colour, label=label,
              linewidth=0)
    return positions


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", type=Path, default=Path("data/scores_all.csv"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args(argv)

    scores = pd.read_csv(args.scores)
    scores["length_bin"] = pd.cut(scores.L, LENGTH_BINS, labels=LENGTH_LABELS)
    rows = present_conditions(scores)
    runs = [r for r, _, _ in rows]
    labels = [l for _, l, _ in rows]
    boxes_correct = [ok for _, _, ok in rows]

    stats = scores.groupby("run").agg(
        lddt=("lddt_all", "mean"),
        tm=("tm_score", "mean"),
        rmsd=("rmsd_ca", "median"),
    ).reindex(runs)

    fig = plt.figure(figsize=(15.5, 4.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.85, 1.0], wspace=0.42)
    a, b, c = (fig.add_subplot(grid[i]) for i in range(3))

    # ---- (a) accuracy, two metrics on one 0-1 axis ----
    _bars(a, rows, stats.lddt, SERIES_1, -0.20, 0.38, "lDDT (local)")
    _bars(a, rows, stats.tm, SERIES_2, 0.20, 0.38, "TM-score (global)")
    for value, colour in ((stats.lddt[CEILING], SERIES_1),
                          (stats.tm[CEILING], SERIES_2)):
        a.axvline(value, color=colour, ls=":", lw=1.2, zorder=0)
    for i, (lddt, tm) in enumerate(zip(stats.lddt, stats.tm)):
        a.text(lddt + 0.008, i - 0.20, f"{lddt:.3f}", va="center", fontsize=7.5,
               color=TEXT_SECONDARY)
        a.text(tm + 0.008, i + 0.20, f"{tm:.3f}", va="center", fontsize=7.5,
               color=TEXT_SECONDARY)
    a.set_xlim(0, 0.66)
    a.set_xlabel("mean score over 554 proteins", fontsize=9, color=TEXT_SECONDARY)
    a.set_title("(a)  Local accuracy reaches the ceiling; global accuracy does not",
                fontsize=10, color=TEXT_PRIMARY, loc="left")
    a.legend(fontsize=8, loc="lower right", frameon=False,
             bbox_to_anchor=(1.0, -0.02))
    f_row = runs.index("f-cc1mix5-step50000") if "f-cc1mix5-step50000" in runs else None
    if f_row is not None:
        a.annotate(
            "same structure —\nlDDT at the ceiling,\nTM a third of it",
            xy=(stats.lddt[runs[f_row]] + 0.005, f_row - 0.20),
            xytext=(0.40, f_row - 1.05), fontsize=7.5, color=TEXT_PRIMARY,
            ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=0.9, color=TEXT_SECONDARY,
                            connectionstyle="arc3,rad=0.25"),
        )

    # ---- (b) the distance scale, where the split is bimodal ----
    # Colour by whether the coarse boxes were correct — the variable the split
    # actually tracks — so the bimodality is in the mark, not only in the prose.
    colours = [SERIES_1 if ok else SERIES_2 for ok in boxes_correct]
    b.barh(range(len(rows)), stats.rmsd, height=0.55, color=colours, linewidth=0)
    for i, value in enumerate(stats.rmsd):
        b.text(value + 0.4, i, f"{value:.1f}", va="center", fontsize=7.5,
               color=TEXT_SECONDARY)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=SERIES_1, label="coarse boxes correct"),
        plt.Rectangle((0, 0), 1, 1, color=SERIES_2, label="boxes from the model"),
    ]
    # Upper right: the two short bars leave that corner empty, so the legend
    # never sits on a mark or its value label.
    b.legend(handles=handles, fontsize=7.5, loc="upper right", frameon=False)
    b.set_xlim(0, 21)
    b.set_xlabel("median CA-RMSD (Å) — lower is better", fontsize=9,
                 color=TEXT_SECONDARY)
    b.set_title("(b)  The fold is right or it is not", fontsize=10,
                color=TEXT_PRIMARY, loc="left")

    for axis in (a, b):
        axis.set_yticks(range(len(rows)))
        axis.invert_yaxis()
        axis.grid(axis="x", color=GRID, lw=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(GRID)
        axis.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    a.set_yticklabels(labels, fontsize=8.5, color=TEXT_PRIMARY)
    b.set_yticklabels([])

    # ---- (c) length: F escapes the fixed token budget ----
    for run, label, colour, style in CURVES:
        if run not in set(scores.run):
            continue
        binned = scores[scores.run == run].groupby("length_bin", observed=True)[
            "lddt_all"
        ].mean()
        c.plot(range(len(binned)), binned.values, marker="o", ms=6, lw=2,
               ls=style, color=colour, label=label)
    c.set_xticks(range(len(LENGTH_LABELS)))
    c.set_xticklabels(LENGTH_LABELS, fontsize=8)
    c.set_xlabel("chain length (residues)", fontsize=9, color=TEXT_SECONDARY)
    c.set_ylabel("lDDT", fontsize=9, color=TEXT_SECONDARY)
    c.set_title("(c)  Re-prompting beats the one-document ceiling on long chains",
                fontsize=10, color=TEXT_PRIMARY, loc="left")
    c.grid(color=GRID, lw=0.7, alpha=0.8)
    c.set_axisbelow(True)
    for side in ("top", "right"):
        c.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        c.spines[side].set_color(GRID)
    c.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    c.legend(fontsize=8, frameon=False, loc="upper right")
    c.annotate("one document runs out\nof token budget here", xy=(2.95, 0.062),
               xytext=(1.25, 0.185), fontsize=7.5, color=TEXT_PRIMARY,
               arrowprops=dict(arrowstyle="->", lw=0.9, color=TEXT_SECONDARY,
                               connectionstyle="arc3,rad=-0.2"))

    fig.suptitle(
        "contacts-and-crops-v1 · 1.5B · 554-protein eval — refinement works, the coarse fold does not",
        fontsize=11.5, color=TEXT_PRIMARY, x=0.008, ha="left", y=1.02,
    )
    save_plot_with_meta(
        fig,
        args.out / "summary_figure.png",
        caption=(
            "exp174 summary. (a) Plan F matches the format's one-document "
            "ceiling on lDDT while reaching only a third of it on TM-score — a "
            "local metric rewards a well-refined wrong fold. (b) Median CA-RMSD "
            "splits cleanly: ~4 Å whenever the coarse boxes are correct, ~16.5 Å "
            "whenever they come from the model, regardless of inference spend. "
            "(c) The one-document ceiling collapses with chain length because the "
            "8192-token budget is fixed; Plan F re-prompts and runs above it."
        ),
        dpi=200,
    )
    print(f"[plot] wrote {args.out / 'summary_figure.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
