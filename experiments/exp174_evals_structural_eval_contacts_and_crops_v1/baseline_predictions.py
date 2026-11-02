# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference "predictors" that need no model: the format's information ceiling.

A contacts-and-crops-v1 document cannot express an atom's position better than
its resolution tier allows, whatever the model. An atom seen only in Pass 1 is
localized to a **10 Å box**; an atom refined by Pass-2 crops converges on
**0.1 Å** (ones + tenths). And because documents are budget-filling and ~96%
are truncated, a document mentions only a fraction of the atoms at all.

So before reading any model's score, we need to know what a *perfect* model
would score. This script produces exactly that, by degrading the ground truth
to each tier and emitting it as a prediction directory the scorer can eat:

* ``exact`` — the ground truth itself. The harness's identity check: lDDT and
  TM-score must come back 1.0 and RMSD 0.0.
* ``tenths`` — every coordinate rounded to 0.1 Å. The ceiling for an atom that
  Pass 2 has refined to convergence.
* ``box10`` — every coordinate replaced by the center of its 10 Å box. The
  ceiling for a Pass-1-only atom.
* ``crops`` — the composite a *perfect document* actually yields: every atom at
  box resolution (Pass 1), with the atoms of a randomly chosen set of boxes
  upgraded to 0.1 Å (Pass 2). ``--fine-frac`` sets how much of the structure
  those crops cover.

``--keep-frac`` drops a fraction of atoms on top, which traces the *coverage*
penalty curve — the second axis of the ceiling, and the one that decides how
much of a model's deficit is the format's truncation rather than the model's
error. ``--keep-mode`` chooses how they are dropped, and the choice matters:

* ``atom`` — uniformly at random. lDDT then falls roughly as coverage², since
  a reference contact survives only if *both* of its atoms do. This is close
  to the worst case for a given coverage.
* ``box`` — whole 10 Å boxes are kept or dropped together, which is how a real
  document's Pass-2 coverage is actually distributed. Contacts inside a kept
  box survive, so lDDT falls much more slowly than coverage².

Note that the box-center placement in ``box10`` / ``crops`` is *a* choice for
what to do with an atom the document localizes but never refines — the same
choice ``PLANS.md`` flags as open (box center vs centroid of refined
neighbors vs exclude). It is the minimax-optimal point estimate given only the
box, so it is the natural reference point, but a plan that does better on
box-only atoms would beat this baseline.

**On the coordinate frame.** Real documents quantize in a randomly rotated and
translated frame, on a grid that is axis-aligned in *that* frame. This script
quantizes on an axis-aligned grid in the ground truth's own frame. The
quantization error distribution is identical either way — a random rigid
transform is exactly what randomizes the grid offset — and every metric here
is frame-invariant, so nothing is lost by skipping the transform.

Usage::

    uv run python baseline_predictions.py --gt-dir _scratch/gt \\
        --out-dir _scratch/pred/box10 --mode box10
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

import canonical_pdb

# Per-mode standard deviation of the positional error, written to the B-factor
# column so the scorer's refined-vs-coarse split has something to act on. A
# value quantized to a grid of width w has error ~Uniform(-w/2, w/2), whose
# standard deviation is w/sqrt(12).
BOX_WIDTH_A = 10.0
TENTH_WIDTH_A = 0.1
TENTH_SIGMA = TENTH_WIDTH_A / math.sqrt(12.0)
BOX_SIGMA = BOX_WIDTH_A / math.sqrt(12.0)
MODE_SIGMA = {
    "exact": 0.0,
    "tenths": TENTH_SIGMA,
    "box10": BOX_SIGMA,
    # `crops` writes a per-atom sigma (fine atoms get TENTH_SIGMA, the rest
    # BOX_SIGMA); this entry is the fallback for an all-coarse structure.
    "crops": BOX_SIGMA,
}


def box_index(coord: np.ndarray) -> np.ndarray:
    """The 10 Å cell each coordinate falls in, as an ``(n, 3)`` integer array."""
    return np.floor(coord / BOX_WIDTH_A).astype(np.int64)


def degrade(coord: np.ndarray, mode: str) -> np.ndarray:
    """Return ``coord`` at the resolution of one contacts-and-crops-v1 tier."""
    if mode == "exact":
        return coord.copy()
    if mode in ("tenths", "crops"):
        # The format's own quantizer: round(v * 10) read back as tenths.
        return np.round(coord * 10.0) / 10.0
    if mode == "box10":
        # Pass 1 localizes an atom to a 10 Å cell and says nothing more; the
        # cell center is the minimax-optimal point estimate for it.
        return (box_index(coord) + 0.5) * BOX_WIDTH_A
    raise ValueError(f"unknown mode {mode!r}")


def select_boxes(coord: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask keeping whole 10 Å boxes until ``frac`` of atoms is reached.

    Boxes are visited in a random order and taken whole, which is how a
    document's fine coverage is really distributed: Pass 2 reveals a *box* at
    a time, not scattered individual atoms. Selection stops at the first box
    that would overshoot the target, so the realized fraction is at or just
    below ``frac``.
    """
    if frac >= 1.0:
        return np.ones(len(coord), dtype=bool)
    cells = box_index(coord)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = rng.permutation(int(inverse.max()) + 1) if len(inverse) else np.empty(0, int)
    target = frac * len(coord)
    keep = np.zeros(len(coord), dtype=bool)
    taken = 0
    for box in order:
        members = inverse == box
        if taken + int(members.sum()) > target:
            continue
        keep |= members
        taken += int(members.sum())
    return keep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=sorted(MODE_SIGMA), required=True)
    ap.add_argument(
        "--keep-frac",
        type=float,
        default=1.0,
        help="keep this fraction of atoms, to trace the coverage penalty curve",
    )
    ap.add_argument(
        "--keep-mode",
        choices=("atom", "box"),
        default="atom",
        help="drop atoms independently at random ('atom', near worst case for "
        "a given coverage) or a whole 10 Å box at a time ('box', how a real "
        "document's coverage is distributed)",
    )
    ap.add_argument(
        "--fine-frac",
        type=float,
        default=0.3,
        help="mode=crops only: fraction of atoms whose box is refined to 0.1 Å",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not 0.0 < args.keep_frac <= 1.0:
        raise ValueError(f"--keep-frac must be in (0, 1], got {args.keep_frac}")
    if not 0.0 <= args.fine_frac <= 1.0:
        raise ValueError(f"--fine-frac must be in [0, 1], got {args.fine_frac}")

    rng = np.random.default_rng(args.seed)
    records = [json.loads(line) for line in (args.gt_dir / "gt_index.jsonl").open()]
    sigma = MODE_SIGMA[args.mode]

    n_written = 0
    for record in records:
        gt_path = (
            args.gt_dir / "gt_structures" / record["dataset"] / f"{record['stem']}.pdb"
        )
        array = canonical_pdb.read_structure(gt_path)
        true_coord = array.coord.astype(np.float64)
        if args.mode == "crops":
            # Pass 1 boxes everything; Pass 2 upgrades whole boxes to tenths.
            coord = degrade(true_coord, "box10")
            fine = select_boxes(true_coord, args.fine_frac, rng)
            coord[fine] = degrade(true_coord[fine], "tenths")
            array.b_factor = np.where(fine, TENTH_SIGMA, BOX_SIGMA)
        else:
            coord = degrade(true_coord, args.mode)
            array.b_factor = np.full(len(array), sigma)
        array.coord = coord.astype(np.float32)
        if args.keep_frac < 1.0:
            keep = (
                select_boxes(true_coord, args.keep_frac, rng)
                if args.keep_mode == "box"
                else rng.random(len(array)) < args.keep_frac
            )
            array = array[keep]
            if len(array) == 0:
                continue
        out_path = args.out_dir / record["dataset"] / f"{record['stem']}.pdb"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_pdb.write_structure(array, out_path)
        n_written += 1

    print(
        f"[baseline] mode={args.mode} keep_frac={args.keep_frac} "
        f"keep_mode={args.keep_mode} fine_frac={args.fine_frac} "
        f"-> {n_written} structures in {args.out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
