# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the model results against the format's ceiling.

Three panels, all reading ``data/scores_all.csv``:

* **left** — lDDT by plan, with the oracle-document ceiling drawn as a line.
  The gap to that line, not the distance from 1.0, is what a model result
  means.
* **middle** — the same by sequence length. Coverage falls with length by
  construction (fixed 8192-token budget, growing atom count), so every
  comparison has to hold length fixed.
* **right** — coverage vs accuracy per protein, which shows whether a plan is
  limited by how much it emits or by how well it places what it emits.

Usage::

    uv run python plot_results.py --scores data/scores_all.csv --out plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_summary import save_plot_with_meta

LENGTH_BINS = [0, 100, 200, 400, 10_000]
LENGTH_LABELS = ["<=100", "101-200", "201-400", ">400"]

# Report order and display names. Anything not listed still plots, at the end.
RUN_LABELS = {
    "oracle-doc": "oracle document (1-doc ceiling)",
    "oracle-document": "oracle document (1-doc ceiling)",
    "e2-cc1mix5-step50000": "E2 oracle boxes",
    "e1-cc1mix5-step50000": "E1 oracle contacts",
    "f-cc1mix5-step50000": "F  mix5",
    "c-cc1mix5-step50000": "C  mix5",
    "a-cc1mix5-step50000": "A  mix5",
    "f-3way-step20000": "F  3way",
    "a-3way-step20000": "A  3way",
}


def ordered_runs(scores: pd.DataFrame) -> list[str]:
    """The runs to plot, in report order.

    Only the model runs and the oracle ceiling; the model-free quantization
    baselines live in ``plots/ceiling.png`` and would crowd this figure.
    """
    present = set(scores.run.unique())
    return [r for r in RUN_LABELS if r in present]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scores", type=Path, default=Path("data/scores_all.csv"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    ap.add_argument("--ceiling-run", default="oracle-doc")
    args = ap.parse_args(argv)

    scores = pd.read_csv(args.scores)
    scores["length_bin"] = pd.cut(
        scores.L, LENGTH_BINS, labels=LENGTH_LABELS, right=True
    )
    runs = ordered_runs(scores)
    label = lambda r: RUN_LABELS.get(r, r)  # noqa: E731

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(15, 4.4))

    # --- left: lDDT and TM by plan ---
    means = scores.groupby("run")[["lddt_all", "tm_score"]].mean().reindex(runs)
    positions = range(len(runs))
    left.barh([p - 0.19 for p in positions], means.lddt_all, height=0.36,
              color="#1f77b4", label="lDDT")
    left.barh([p + 0.19 for p in positions], means.tm_score, height=0.36,
              color="#d62728", label="TM-score")
    if args.ceiling_run in means.index:
        for value, colour in ((means.loc[args.ceiling_run, "lddt_all"], "#1f77b4"),
                              (means.loc[args.ceiling_run, "tm_score"], "#d62728")):
            left.axvline(value, color=colour, ls="--", lw=1)
    left.set_yticks(list(positions))
    left.set_yticklabels([label(r) for r in runs], fontsize=8)
    left.invert_yaxis()
    left.set_xlabel("mean score over 554 proteins")
    left.set_title("Accuracy by plan (dashed = ceiling)")
    left.grid(alpha=0.25, axis="x")
    left.legend(fontsize=8)

    # --- middle: lDDT by length ---
    for run in runs:
        group = scores[scores.run == run]
        binned = group.groupby("length_bin", observed=True)["lddt_all"].mean()
        style = "--" if run == args.ceiling_run else "-"
        middle.plot(binned.index.astype(str), binned.values, marker="o",
                    ls=style, lw=1.6, label=label(run))
    middle.set_xlabel("sequence length")
    middle.set_ylabel("lDDT")
    middle.set_title("Accuracy vs length")
    middle.grid(alpha=0.25)
    middle.tick_params(axis="x", rotation=20)
    middle.legend(fontsize=7)

    # --- right: coverage vs lDDT, per protein ---
    for run in runs:
        group = scores[(scores.run == run) & (scores.status == "ok")]
        right.scatter(group.atom_coverage, group.lddt_all, s=6, alpha=0.35,
                      label=label(run))
    right.set_xlabel("atom coverage")
    right.set_ylabel("lDDT")
    right.set_title("Is it coverage-limited or accuracy-limited?")
    right.grid(alpha=0.25)
    right.legend(fontsize=7, markerscale=2)

    fig.tight_layout()
    save_plot_with_meta(
        fig,
        args.out / "results.png",
        caption=(
            "contacts-and-crops-v1 structure prediction on the 554-protein eval "
            "set. Left: mean lDDT and TM-score by inference plan, with the "
            "oracle-document ceiling dashed. Middle: the same by length — "
            "coverage falls with length by construction. Right: per-protein "
            "coverage against lDDT, separating coverage-limited from "
            "accuracy-limited failure."
        ),
        dpi=160,
    )
    print(f"[plot] wrote {args.out / 'results.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
