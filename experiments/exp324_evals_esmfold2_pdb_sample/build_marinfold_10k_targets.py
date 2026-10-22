# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the 10k PDB-deduped target parquet consumed by MarinFold rollout workers."""

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/sample_10000_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/sample_10000_targets.parquet"))
    parser.add_argument("--dataset", default="pdb_deduped_10k")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    targets = pd.DataFrame(
        {
            "dataset": args.dataset,
            "stem": manifest["stem"].astype(str),
            "L": manifest["seq_len"].astype("int32"),
            "input_seq": manifest["sequence"].astype(str),
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    targets.to_parquet(args.out, index=False)
    print(f"wrote {len(targets)} targets -> {args.out}")


if __name__ == "__main__":
    main()
