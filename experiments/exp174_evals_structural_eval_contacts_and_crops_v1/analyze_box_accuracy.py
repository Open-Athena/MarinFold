# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Why the de-novo plans fail: at what *scale* is the error?

The aggregate metrics say the coarse fold is wrong but not by how much, and
that distinction decides whether more inference could ever help. Pass-2 crops
emit only ones + tenths — the ``<crop>`` header supplies hundreds + tens — so
refinement is confined to the named 10 Å cell **by construction**. If the
typical atom is misplaced by more than a box, no amount of within-box
refinement can reach it.

So: superimpose each prediction on its ground truth and bin the per-atom
displacement against the box scale.

* ``< 5 Å``  — the atom is in (or on the edge of) the right 10 Å box; refinement
  is the operation that can fix it.
* ``< 10 Å`` — at worst an adjacent box.
* beyond that, the error is a fold error and refinement is the wrong tool.

Usage::

    uv run python analyze_box_accuracy.py --gt-dir _scratch/gt \\
        --pred-dir _scratch/pred --out data/box_accuracy.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from biotite.structure import superimpose

import canonical_pdb

# The format's coarse cell is 10 Å, so "within 5 Å" is the natural proxy for
# "in the right box" (half-width), and 10 Å for "no worse than adjacent".
BOX_A = 10.0

DEFAULT_RUNS = [
    "oracle-doc",
    "e2-cc1mix5-step50000",
    "e1-cc1mix5-step50000",
    "f-cc1mix5-step50000",
    "c-cc1mix5-step50000",
    "a-cc1mix5-step50000",
    "a-3way-step20000",
    "f-3way-step20000",
]


def per_atom_error(gt, pred) -> np.ndarray | None:
    """Displacement of every common atom after a least-squares superposition."""
    gt_rows = {
        key: i for i, key in enumerate(zip(gt.res_id.tolist(), gt.atom_name.tolist()))
    }
    pred_index, gt_index = [], []
    for i, key in enumerate(zip(pred.res_id.tolist(), pred.atom_name.tolist())):
        if key in gt_rows:
            pred_index.append(i)
            gt_index.append(gt_rows[key])
    if len(pred_index) < 3:
        return None
    fitted, _ = superimpose(gt[gt_index], pred[pred_index])
    return np.linalg.norm(fitted.coord - gt[gt_index].coord, axis=1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-dir", type=Path, default=Path("_scratch/gt"))
    ap.add_argument("--pred-dir", type=Path, default=Path("_scratch/pred"))
    ap.add_argument("--out", type=Path, default=Path("data/box_accuracy.csv"))
    ap.add_argument("--runs", default=None, help="comma-separated; default: all known")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    records = [json.loads(line) for line in (args.gt_dir / "gt_index.jsonl").open()]
    if args.limit:
        records = records[: args.limit]
    runs = args.runs.split(",") if args.runs else DEFAULT_RUNS

    rows = []
    for run in runs:
        root = args.pred_dir / run
        if not root.is_dir():
            continue
        within_half, within_box, medians, n_atoms = [], [], [], 0
        for record in records:
            gt_path = (
                args.gt_dir / "gt_structures" / record["dataset"] / f"{record['stem']}.pdb"
            )
            pred_path = root / record["dataset"] / f"{record['stem']}.pdb"
            if not pred_path.exists():
                continue
            error = per_atom_error(
                canonical_pdb.read_structure(gt_path),
                canonical_pdb.read_structure(pred_path),
            )
            if error is None:
                continue
            within_half.append(float((error < BOX_A / 2).mean()))
            within_box.append(float((error < BOX_A).mean()))
            medians.append(float(np.median(error)))
            n_atoms += len(error)
        if not medians:
            continue
        rows.append(
            {
                "run": run,
                "n_proteins": len(medians),
                "n_atoms": n_atoms,
                "frac_within_5a_right_box": float(np.mean(within_half)),
                "frac_within_10a_adjacent_box": float(np.mean(within_box)),
                "median_atom_error_a": float(np.median(medians)),
            }
        )
        print(f"  {run}: {rows[-1]['frac_within_5a_right_box']:.1%} in the right box", flush=True)

    frame = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\n[box] {args.out}")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
