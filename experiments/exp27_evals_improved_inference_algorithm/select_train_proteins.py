# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Pick the frozen 10-protein train set for exp27.

Uniform random sample of 10 rows from the Protenix manifest filtered
to ``n_residues <= max_n_residues``, with
``random.Random(27).sample(...)``. Writes the chosen rows verbatim to
``data/train_proteins.csv`` so the train set is in git and never
drifts. We DO NOT look at the other 90 proteins for the duration of
this experiment (per the issue).

The length cap exists because we run baseline+experiments locally on
V100s in fp32 — sm_70 has no native bf16 / no FlashAttention, so the
naive distogram readout costs ~9 hours wall-clock for a single 738 aa
protein. Capping at 400 aa keeps iteration tractable while still
sampling across the easy/medium length range of FoldBench monomers.
The seed is unchanged from the original draw; the cap simply trims
the manifest pool before sampling.

Idempotent — overwriting ``data/train_proteins.csv`` is fine; the
seed + cap pin the sample. If the manifest itself ever changes (it
won't, the exp12 dataset is frozen on HF), the seed alone is no
longer enough — that's why we also commit the resolved CSV.
"""

import argparse
import csv
import random
from pathlib import Path


_THIS = Path(__file__).resolve().parent
_MANIFEST = (
    _THIS / "protenix_data" / "data" / "protenix-foldbench-monomers" / "manifest.csv"
)
_OUT = _THIS / "data" / "train_proteins.csv"
_SEED = 27
_N = 10
_MAX_N_RESIDUES = 400


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_MANIFEST)
    parser.add_argument("--out", type=Path, default=_OUT)
    parser.add_argument("--seed", type=int, default=_SEED)
    parser.add_argument("--n", type=int, default=_N)
    parser.add_argument(
        "--max-n-residues",
        type=int,
        default=_MAX_N_RESIDUES,
        help=(
            "Cap the sampling pool to proteins with at most this many "
            "residues. Default 400 — keeps baseline tractable on V100 fp32. "
            "Pass 9999 to disable the cap."
        ),
    )
    args = parser.parse_args()

    with args.manifest.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    pool = [r for r in rows if int(r["n_residues"]) <= args.max_n_residues]
    if len(pool) < args.n:
        raise ValueError(
            f"Manifest filtered to n_residues<={args.max_n_residues} has "
            f"{len(pool)} rows; cannot sample {args.n}."
        )

    rng = random.Random(args.seed)
    chosen = rng.sample(pool, args.n)
    # Sort by n_residues so the printed summary is readable. The TSV
    # row order doesn't affect downstream code (everything keys by stem).
    chosen.sort(key=lambda r: int(r["n_residues"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chosen)

    print(
        f"Wrote {args.out} with {len(chosen)} proteins "
        f"(seed={args.seed}, max_n_residues={args.max_n_residues}, "
        f"pool size after cap={len(pool)}/{len(rows)}):"
    )
    print(f"  {'stem':<10} {'n_residues':>10}")
    total = 0
    for row in chosen:
        n = int(row["n_residues"])
        total += n
        print(f"  {row['stem']:<10} {n:>10}")
    print(f"  {'TOTAL':<10} {total:>10}")
    print(
        f"  min={min(int(r['n_residues']) for r in chosen)} "
        f"max={max(int(r['n_residues']) for r in chosen)} "
        f"mean={total / len(chosen):.1f}"
    )


if __name__ == "__main__":
    main()
