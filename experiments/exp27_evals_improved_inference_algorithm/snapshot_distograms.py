# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Snapshot outputs/<stem>/distogram.npz to a named snapshot file.

Used after running an algorithm whose distograms we want to keep as a
prior for downstream experiments (e.g. baseline_naive → snapshot →
seed all future seeded variants from the snapshot).

Usage:
    uv run python snapshot_distograms.py --to distogram_baseline_naive.npz
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

_THIS = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv", type=Path,
        default=_THIS / "data" / "train_proteins.csv",
    )
    parser.add_argument("--out", type=Path, default=_THIS / "outputs")
    parser.add_argument(
        "--from-name", default="distogram.npz",
        help="Source file inside outputs/<stem>/",
    )
    parser.add_argument(
        "--to", required=True,
        help="Destination filename inside outputs/<stem>/",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing snapshot.",
    )
    args = parser.parse_args()

    with args.train_csv.open() as f:
        rows = list(csv.DictReader(f))

    copied = 0
    skipped = 0
    missing: list[str] = []
    for r in rows:
        stem = r["stem"]
        src = args.out / stem / args.from_name
        dst = args.out / stem / args.to
        if not src.exists():
            missing.append(stem)
            continue
        if dst.exists() and not args.force:
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1

    if missing:
        print(f"ERROR: missing {args.from_name} for: {', '.join(missing)}")
        sys.exit(1)
    print(f"Copied {copied}, skipped {skipped} (already existed; use --force to overwrite).")


if __name__ == "__main__":
    main()
