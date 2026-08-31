# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 5 — freeze the low-MSA-depth evaluation set.

The 29 natural eval proteins whose ColabFold MSA holds fewer than 10 sequences
are the regime a single-sequence model exists for, and they are now a standing
reporting cut (see the ``eval-checkpoint`` skill). Membership has to be a file
rather than a filter applied fresh each time: the depths come off Modal volumes
that only a few people can read, and a set that is recomputed per evaluation is
a set that quietly changes underneath a comparison.

The 29 span **both** eval universes — 5 FoldBench monomers (all in `eval-test`)
and 24 CAMEO-hard / CASP-FM targets that only exist in the legacy 554 — so any
run that wants to report this cut has to score both.

    uv run python build_low_depth_set.py
"""

import argparse
import json

import pandas as pd
import upstream as U

#: The depth cut this set is defined by, and the counts it must produce.
DEPTH_THRESHOLD = 10
EXPECTED_TOTAL = 29
EXPECTED_BY_DATASET = {"cameo_hard": 16, "casp_fm": 8, "foldbench_monomer": 5}

COLUMNS = [
    "dataset",
    "stem",
    "subset",
    "eval_set",
    "L",
    "msa_depth",
    "msa_neff",
    "msa_volume",
    "is_viral",
    "kingdom",
]


def build() -> pd.DataFrame:
    """Return one row per protein in the low-MSA-depth set."""

    frame = pd.read_csv(U.DATA / "per_protein_depth.csv")
    proteins = frame.drop_duplicates(["dataset", "stem"])[COLUMNS]
    low = proteins[
        (proteins.subset != "foldbench_designed")
        & (proteins.msa_depth < DEPTH_THRESHOLD)
    ].sort_values(["dataset", "stem"], ignore_index=True)

    if len(low) != EXPECTED_TOTAL:
        raise ValueError(f"expected {EXPECTED_TOTAL} proteins, got {len(low)}")
    counts = low.dataset.value_counts().to_dict()
    if counts != EXPECTED_BY_DATASET:
        raise ValueError(f"membership changed: {counts} != {EXPECTED_BY_DATASET}")
    return low


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(U.DATA / "low_msa_depth_set.csv"))
    args = parser.parse_args()
    low = build()
    low.to_csv(args.out, index=False)
    print(
        json.dumps(
            {
                "n": len(low),
                "by_dataset": low.dataset.value_counts().to_dict(),
                "by_subset": low.subset.value_counts().to_dict(),
                "depth_range": [int(low.msa_depth.min()), int(low.msa_depth.max())],
                "median_length": float(low.L.median()),
                "out": args.out,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
