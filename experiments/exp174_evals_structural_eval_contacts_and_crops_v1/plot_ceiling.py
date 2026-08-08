# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the model-free ceiling: what a *perfect* contacts-and-crops-v1 document scores.

Two panels, both read straight out of ``data/baseline_ceiling.csv``:

* **left** — accuracy vs coverage, for ground truth kept at 0.1 Å but thinned.
  The reference curves (coverage² for lDDT, coverage for TM-score) are drawn
  on top: an lDDT contact needs both of its atoms, a TM-score residue needs
  only itself, and the measured points sit on those curves.
* **right** — accuracy vs the fraction of atoms Pass 2 refines, with Pass 1
  boxing everything. This is the curve an inference plan moves along, and the
  marked point is what a single realistic document reaches.

Usage::

    uv run python plot_ceiling.py --ceiling data/baseline_ceiling.csv --out plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_summary import save_plot_with_meta

LDDT_COLOR = "#1f77b4"
TM_COLOR = "#d62728"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ceiling", type=Path, default=Path("data/baseline_ceiling.csv"))
    ap.add_argument("--out", type=Path, default=Path("plots"))
    args = ap.parse_args(argv)

    table = pd.read_csv(args.ceiling)
    overall = table[table["stratum"] == "all"].set_index("baseline")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- left: accuracy vs coverage, at full 0.1 Å resolution ---
    grid = np.linspace(0.0, 1.0, 200)
    left.plot(grid, grid**2, color=LDDT_COLOR, ls=":", lw=1.2, label="coverage$^2$")
    left.plot(grid, grid, color=TM_COLOR, ls=":", lw=1.2, label="coverage")
    for keep_mode, marker in (("atom", "o"), ("box", "s")):
        rows = overall[
            (overall["mode"] == "tenths") & (overall["keep_mode"] == keep_mode)
        ].sort_values("mean_atom_coverage")
        # The full-coverage `tenths` row has keep_mode 'atom' by convention;
        # include it in both series so each curve reaches 1.0.
        full = overall.loc[["tenths"]]
        rows = pd.concat([rows, full]).sort_values("mean_atom_coverage")
        left.plot(
            rows["mean_atom_coverage"], rows["mean_lddt_all"],
            marker=marker, color=LDDT_COLOR, lw=1.4,
            label=f"lDDT ({keep_mode}-wise dropout)",
        )
        left.plot(
            rows["mean_atom_coverage"], rows["mean_tm_score"],
            marker=marker, color=TM_COLOR, lw=1.4,
            label=f"TM-score ({keep_mode}-wise dropout)",
        )
    left.set_xlabel("atom coverage")
    left.set_ylabel("score")
    left.set_title("Coverage penalty at 0.1 Å resolution")
    left.set_xlim(0, 1.02)
    left.set_ylim(0, 1.02)
    left.grid(alpha=0.25)
    left.legend(fontsize=7, loc="upper left")

    # --- right: accuracy vs refined fraction, Pass 1 boxing everything ---
    crops = overall[overall["mode"] == "crops"].copy()
    crops = crops[crops["keep_frac"] == 1.0].sort_values("fine_frac")
    # fine_frac 0 is the all-boxes structure; fine_frac 1 is the all-tenths one.
    fine = np.concatenate([[0.0], crops["fine_frac"].to_numpy(), [1.0]])
    lddt = np.concatenate(
        [[overall.loc["box10", "mean_lddt_all"]],
         crops["mean_lddt_all"].to_numpy(),
         [overall.loc["tenths", "mean_lddt_all"]]]
    )
    tm = np.concatenate(
        [[overall.loc["box10", "mean_tm_score"]],
         crops["mean_tm_score"].to_numpy(),
         [overall.loc["tenths", "mean_tm_score"]]]
    )
    right.plot(fine, lddt, marker="o", color=LDDT_COLOR, lw=1.4, label="lDDT")
    right.plot(fine, tm, marker="o", color=TM_COLOR, lw=1.4, label="TM-score")

    single = overall.loc["crops-single-doc"]
    right.scatter(
        [single["fine_frac"]], [single["mean_lddt_all"]],
        marker="*", s=200, color=LDDT_COLOR, zorder=5, edgecolor="k", linewidth=0.5,
    )
    right.scatter(
        [single["fine_frac"]], [single["mean_tm_score"]],
        marker="*", s=200, color=TM_COLOR, zorder=5, edgecolor="k", linewidth=0.5,
    )
    right.annotate(
        "one realistic document\n(65% boxed, 25% refined)",
        xy=(single["fine_frac"], single["mean_tm_score"]),
        xytext=(0.34, 0.80), fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )
    right.set_xlabel("fraction of atoms refined to 0.1 Å by Pass 2")
    right.set_ylabel("score")
    right.set_title("Pass-1 boxes everywhere + Pass-2 refinement")
    right.set_xlim(-0.02, 1.02)
    right.set_ylim(0, 1.02)
    right.grid(alpha=0.25)
    right.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    save_plot_with_meta(
        fig,
        args.out / "ceiling.png",
        caption=(
            "Model-free ceiling on the 554-protein eval set: ground truth "
            "degraded to each contacts-and-crops-v1 resolution tier, scored by "
            "the same harness. Left: lDDT falls as coverage^2, TM-score "
            "linearly. Right: the Pass-2 refined fraction is the whole "
            "ballgame; a single document reaches lDDT 0.17 / TM 0.41."
        ),
        dpi=160,
    )
    print(f"[plot] wrote {args.out / 'ceiling.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
