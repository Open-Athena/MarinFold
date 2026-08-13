# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the decontamination reference and the drop lists to the public bucket.

The point of pinning a reference is that the *next* corpus filters against the
same one, and that only works if it is fetchable without a checkout of this
experiment's scratch disk. Three things go up, under one versioned prefix:

* ``eval_queries.fasta`` + ``eval_structures.csv`` — the reference itself (also
  committed to git, since they are small; the bucket copy is what a cluster job
  reads).
* ``eval_structures.tar.gz`` — the 554 single-chain mmCIFs. 96 MB, so git is
  out; the committed manifest's sha256 column is what makes this copy
  verifiable.
* ``droplist_sequence.parquet`` / ``droplist_structure_afdb.parquet`` — the
  derived drop lists, so a filtering job does not have to re-run MMseqs2 or
  Foldseek to rebuild them.

Writing needs an **open-athena-scoped** token (``hf auth whoami`` must list the
org); reading needs nothing.

    uv run python publish_reference.py --dry-run
    uv run python publish_reference.py
"""
from __future__ import annotations

import argparse
import subprocess
import tarfile
import tempfile
from pathlib import Path

from decontam_lib import REFERENCE_VERSION

HERE = Path(__file__).resolve().parent
BUCKET = "hf://buckets/open-athena/MarinFold"
PREFIX = f"data/decontamination/contacts_v1_eval_reference/{REFERENCE_VERSION}"


def upload(local: Path, remote: str, *, dry_run: bool) -> None:
    cmd = ["hf", "buckets", "cp", str(local), f"{BUCKET}/{remote}"]
    print("  $", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--work", type=Path, default=Path("/data/exp225_decontam"))
    ap.add_argument("--structures", type=Path,
                    default=Path("/data/exp225_decontam/eval_structures"))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the uploads without performing them")
    args = ap.parse_args()

    print(f"[publish] {BUCKET}/{PREFIX}/", flush=True)
    for name in ("eval_queries.fasta", "eval_structures.csv", "reference.provenance.json"):
        upload(HERE / "data/reference" / name, f"{PREFIX}/{name}", dry_run=args.dry_run)

    with tempfile.TemporaryDirectory() as tmp:
        tarball = Path(tmp) / "eval_structures.tar.gz"
        if not args.dry_run:
            with tarfile.open(tarball, "w:gz") as tf:
                for path in sorted(args.structures.glob("*.cif")):
                    tf.add(path, arcname=f"eval_structures/{path.name}")
            print(f"[publish] tarball {tarball.stat().st_size / 1e6:.0f} MB", flush=True)
        upload(tarball, f"{PREFIX}/eval_structures.tar.gz", dry_run=args.dry_run)

    for name in ("droplist_sequence.parquet", "droplist_structure_afdb.parquet"):
        path = args.work / name
        if not path.exists():
            print(f"[publish] skipping {name} (not built yet)", flush=True)
            continue
        upload(path, f"{PREFIX}/{name}", dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
