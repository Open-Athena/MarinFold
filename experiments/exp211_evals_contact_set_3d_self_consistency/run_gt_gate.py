# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step B (issue #211) — the calibration gate, on ground truth alone.

Before spending any GPU time generating rollouts, prove the metric works on data
whose answer is known. For every one of the 554 eval proteins, score:

* the **ground-truth** contact set — which came off a real structure, so it *is*
  realizable in 3D and must score ~0; and
* a **separation-matched random** set of the same size and the same ``|i - j|``
  profile — which must score worse.

The issue's stated gate is: GT scores ~0 on >= 95% of proteins, and random scores
strictly worse than GT on essentially all of them. If this fails, the bounds or
the optimizer are wrong and no amount of rollout data will fix it.

This also produces the **null distribution** the real arms are read against, and
it is the only part of the pipeline that can be run before the rollouts land.

Run (after ``calibrate_bounds.py``)::

    uv run python run_gt_gate.py --gt-dir _scratch/gt --bounds data/bounds.json \
        --out data/gt_gate.csv

**Chain breaks.** ~0.1% of consecutive-index CA pairs in the bundle are not 3.8 A
apart (the pooled bond distribution runs p99.9 = 5.62 A with a 31.3 A tail):
deposited structures with disordered or mis-numbered stretches. The scorer
imposes a continuous 3.8 A chain because that is what a contacts-v1 document
asserts, so for those proteins the ground truth genuinely is not embeddable as a
continuous chain. They are flagged in a ``has_chain_break`` column rather than
dropped, so the gate can report them instead of being quietly dragged down.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from arms import decoy_protein, ground_truth, separation_matched_random
from calibrate_bounds import MIN_CONTACT_DEGREE, MIN_SEP, load_bundle
from consistency import Bounds, contact_matrix, embed_residual, packing_score, triangle_violations

# A consecutive-index CA pair further apart than this is a chain break, not a
# peptide bond (pooled p99.9 = 5.62 A, so this sits above the honest tail).
CHAIN_BREAK_A = 5.0


def bounds_from_json(path: Path) -> Bounds:
    d = json.loads(path.read_text())
    return Bounds(
        bond=d["bond"],
        u_contact=d["u_contact"],
        l_noncontact=d["l_noncontact"],
        d_min=d["d_min"],
        min_sep=d["min_sep"],
    )


def gt_pairs(raw_contacts) -> list[tuple[int, int]]:
    """The contacts-v1 ground-truth contact set: degree and separation filtered."""
    return ground_truth(
        (i, j)
        for i, j, degree in raw_contacts
        if degree >= MIN_CONTACT_DEGREE and abs(j - i) >= MIN_SEP
    )


def chain_break_count(xyz: np.ndarray) -> int:
    """Consecutive resolved CA pairs that are too far apart to be bonded."""
    d = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    return int(np.sum(np.isfinite(d) & (d > CHAIN_BREAK_A)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", type=Path, default=Path("_scratch/gt"))
    ap.add_argument("--bounds", type=Path, default=Path("data/bounds.json"))
    ap.add_argument("--out", type=Path, default=Path("data/gt_gate.csv"))
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--n-random", type=int, default=3,
                    help="separation-matched random draws per protein")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    bounds = bounds_from_json(args.bounds)
    print(f"[gate] bounds: {bounds}")

    # Load everything first: the decoy arm needs a donor of comparable length,
    # which means knowing all the proteins before scoring any of them.
    proteins = []
    for n, (record_id, meta, xyz, raw) in enumerate(load_bundle(args.gt_dir)):
        if args.limit and n >= args.limit:
            break
        proteins.append(
            {
                "record_id": record_id,
                "dataset": meta["dataset"],
                "L": int(meta["L"]),
                "gt": gt_pairs(raw),
                "n_chain_breaks": chain_break_count(xyz),
            }
        )
    proteins.sort(key=lambda p: p["L"])
    print(f"[gate] {len(proteins)} proteins loaded")

    rows, t0 = [], time.time()
    for k, p in enumerate(proteins):
        length, gt = p["L"], p["gt"]
        if len(gt) < 3:
            continue
        rng = np.random.default_rng(abs(hash(p["record_id"])) % (2**31))

        sets = {"gt": gt}
        for r in range(args.n_random):
            sets[f"random_{r}"] = separation_matched_random(gt, length, rng)
        # Donor for the decoy arm: the nearest protein by length that is not this
        # one. Sorting by L above makes the neighbour the natural choice.
        donor = proteins[k + 1] if k + 1 < len(proteins) else proteins[k - 1]
        sets["decoy"] = decoy_protein(donor["gt"], length, len(gt), rng)

        names = [n for n, s in sets.items() if len(s) >= 3]
        masks = np.stack([contact_matrix(sets[n], length) for n in names])
        emb = embed_residual(
            masks, bounds, n_restarts=args.n_restarts, iters=args.iters, seed=k
        )
        for name, mask, e in zip(names, masks, emb):
            rows.append(
                {
                    "record_id": p["record_id"],
                    "dataset": p["dataset"],
                    "L": length,
                    "arm": name,
                    "n_chain_breaks": p["n_chain_breaks"],
                    "has_chain_break": p["n_chain_breaks"] > 0,
                    **packing_score(mask),
                    **triangle_violations(mask, bounds),
                    **e,
                }
            )
        if (k + 1) % 50 == 0:
            el = time.time() - t0
            print(f"[gate] {k + 1}/{len(proteins)} proteins  {el / 60:.1f} min  "
                  f"({el / (k + 1):.1f} s/protein)", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    # ---- the gate ----
    wide = df.pivot_table(index="record_id", columns="arm", values="contact_excess_per_contact")
    rand_cols = [c for c in wide.columns if c.startswith("random_")]
    rand_best = wide[rand_cols].min(axis=1)
    clean = df[~df["has_chain_break"]]["record_id"].unique()
    wide_clean = wide.loc[wide.index.intersection(clean)]
    rb_clean = rand_best.loc[wide_clean.index]

    gt_near_zero = float((wide_clean["gt"] < 0.01).mean())
    gt_beats_random = float((wide_clean["gt"] < rb_clean).mean())

    print(f"\n=== calibration gate ({len(wide_clean)} chain-break-free proteins of "
          f"{len(wide)}) ===")
    print(f"  GT contact excess/contact: median {wide_clean['gt'].median():.4f}  "
          f"p90 {wide_clean['gt'].quantile(0.9):.4f}")
    print(f"  best-of-{len(rand_cols)} random:      median {rb_clean.median():.4f}  "
          f"p10 {rb_clean.quantile(0.1):.4f}")
    if "decoy" in wide_clean:
        print(f"  decoy protein:             median "
              f"{wide_clean['decoy'].median():.4f}")
    print(f"\n  GT < 0.01 per contact:  {100 * gt_near_zero:5.1f}%  (gate: >= 95%)")
    print(f"  GT < best random:       {100 * gt_beats_random:5.1f}%")
    print(f"\n  proteins with a chain break: {df.groupby('record_id')['has_chain_break'].first().sum()}")
    print(f"\nwrote {args.out}  ({len(df)} rows, {(time.time() - t0) / 60:.1f} min)")

    return 0 if (gt_near_zero >= 0.95 and gt_beats_random >= 0.95) else 1


if __name__ == "__main__":
    raise SystemExit(main())
