# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the full model-free ceiling grid and collect it into one table.

Drives ``baseline_predictions.py`` + ``score_structures.py`` over the
resolution tiers and coverage fractions that bracket what a contacts-and-crops-v1
document can express, and writes a tidy CSV every later plot and README table
reads from.

The grid answers three questions that have to be settled before any model
number means anything:

* **Resolution ceiling** — ``exact`` (harness identity check), ``tenths``
  (an atom Pass 2 refined to convergence), ``box10`` (a Pass-1-only atom).
* **Coverage ceiling** — ``tenths`` at 100 / 50 / 30 / 15 % of atoms kept,
  dropped both independently (``keep_mode=atom``) and a whole 10 Å box at a
  time (``keep_mode=box``, how a document's coverage is really distributed).
  The SPEC's coverage table says a 150–500-residue protein gets 30–70 % of its
  atoms boxed and 12–40 % at full precision, so this is the range a real
  document actually lives in.
* **What a perfect document scores** — ``crops-fine-*``: Pass-1 boxes for every
  atom plus Pass-2 refinement of 50 / 30 / 15 % of them. This is the number a
  trained model is actually competing against, and it is a long way below 1.0.

Usage::

    uv run python run_baselines.py --gt-dir _scratch/gt \\
        --work-dir _scratch --out data/baseline_ceiling.csv
"""

import argparse
from pathlib import Path

import pandas as pd

import baseline_predictions
import score_structures

# (label, mode, keep_frac, keep_mode, fine_frac). Order is report order.
GRID: tuple[tuple[str, str, float, str, float], ...] = (
    # Resolution ceiling at full coverage.
    ("exact", "exact", 1.0, "atom", 0.0),
    ("tenths", "tenths", 1.0, "atom", 0.0),
    ("box10", "box10", 1.0, "atom", 0.0),
    # Coverage ceiling, atoms dropped independently (near worst case).
    ("tenths-atom-50pct", "tenths", 0.50, "atom", 0.0),
    ("tenths-atom-30pct", "tenths", 0.30, "atom", 0.0),
    ("tenths-atom-15pct", "tenths", 0.15, "atom", 0.0),
    # Coverage ceiling, whole 10 Å boxes dropped (how documents really cover).
    ("tenths-box-50pct", "tenths", 0.50, "box", 0.0),
    ("tenths-box-30pct", "tenths", 0.30, "box", 0.0),
    ("tenths-box-15pct", "tenths", 0.15, "box", 0.0),
    # What a *perfect document* yields: Pass-1 boxes everywhere, Pass-2 crops
    # refining part of the structure to 0.1 Å.
    ("crops-fine-50pct", "crops", 1.0, "atom", 0.50),
    ("crops-fine-30pct", "crops", 1.0, "atom", 0.30),
    ("crops-fine-15pct", "crops", 1.0, "atom", 0.15),
    # A single realistic document: the SPEC's coverage table says a
    # 150-500-residue protein gets 30-70 % of its atoms boxed by Pass 1 and
    # 12-40 % refined by Pass 2, so take the middle of both ranges. This is
    # the number a one-document-per-protein inference plan is shooting at.
    ("crops-single-doc", "crops", 0.65, "box", 0.25),
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="scratch root for the generated prediction trees and per-record CSVs",
    )
    ap.add_argument("--out", type=Path, required=True, help="combined summary CSV")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)

    summaries = []
    for label, mode, keep_frac, keep_mode, fine_frac in GRID:
        pred_dir = args.work_dir / "pred" / label
        scores_path = args.work_dir / "scores" / f"{label}.csv"
        baseline_predictions.main(
            [
                "--gt-dir", str(args.gt_dir),
                "--out-dir", str(pred_dir),
                "--mode", mode,
                "--keep-frac", str(keep_frac),
                "--keep-mode", keep_mode,
                "--fine-frac", str(fine_frac),
                "--seed", str(args.seed),
            ]
        )
        score_structures.main(
            [
                "--gt-dir", str(args.gt_dir),
                "--pred-dir", str(pred_dir),
                "--model-name", f"baseline-{label}",
                "--out", str(scores_path),
                "--jobs", str(args.jobs),
            ]
        )
        summary = pd.read_csv(scores_path.with_suffix(".summary.csv"))
        summary.insert(0, "baseline", label)
        summary.insert(1, "mode", mode)
        summary.insert(2, "keep_frac", keep_frac)
        summary.insert(3, "keep_mode", keep_mode)
        summary.insert(4, "fine_frac", fine_frac)
        summaries.append(summary)

    combined = pd.concat(summaries, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out, index=False)
    print(f"\n[baselines] combined summary -> {args.out}")

    overall = combined[combined["stratum"] == "all"]
    columns = [
        "baseline",
        "mean_atom_coverage",
        "mean_lddt_all",
        "mean_lddt_ca",
        "mean_tm_score",
        "mean_lddt_all_covered",
        "mean_rmsd_all",
    ]
    print(overall[columns].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
