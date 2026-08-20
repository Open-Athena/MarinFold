# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the ground-truth bundle to the public ``open-athena/MarinFold`` bucket.

The bundle is checkpoint-independent and expensive-ish to rebuild (it needs the
exp78 checkout's staged third-party structures, which are not in this repo, plus
a pyconfind run per protein). Publishing it once means every later scoring run —
and anyone outside the cluster — can pull it instead.

Uploads three objects to
``hf://buckets/open-athena/MarinFold/data/exp174-structural-eval/gt/``:

* ``gt_structures.tar.gz`` — the 554 canonical PDBs (62 MB raw, ~13 MB gzipped;
  one archive rather than 554 objects, which uploads and fetches far faster)
* ``gt_index.jsonl`` — per-record lengths, atom counts, alignment quality, strata
* ``gt_contacts.jsonl`` — every degree>0 pyconfind contact, input-seq coordinates

Fetching it back::

    hf buckets cp hf://buckets/open-athena/MarinFold/data/exp174-structural-eval/gt/gt_structures.tar.gz .
    hf buckets cp hf://buckets/open-athena/MarinFold/data/exp174-structural-eval/gt/gt_index.jsonl .
    hf buckets cp hf://buckets/open-athena/MarinFold/data/exp174-structural-eval/gt/gt_contacts.jsonl .
    mkdir -p _scratch/gt && tar -xzf gt_structures.tar.gz -C _scratch/gt

Needs the ``hf`` CLI (>= 1.5, for ``hf buckets``) on ``PATH`` and a token with
**open-athena** write scope — ``hf auth whoami`` must list the org. The
workstation default token may be personal-only; see the root ``AGENTS.md``.
"""

import argparse
import subprocess
import tarfile
from pathlib import Path

BUCKET_PREFIX = (
    "hf://buckets/open-athena/MarinFold/data/exp174-structural-eval/gt"
)


def build_archive(gt_dir: Path, archive: Path) -> Path:
    """Tar + gzip the ``gt_structures`` tree, preserving the dataset layout."""
    structures = gt_dir / "gt_structures"
    if not structures.is_dir():
        raise FileNotFoundError(f"{structures} does not exist — run prepare_gt_structures.py first")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(structures, arcname="gt_structures")
    return archive


def upload(local: Path, prefix: str, *, dry_run: bool) -> None:
    """``hf buckets cp`` one file into the bucket prefix."""
    command = ["hf", "buckets", "cp", str(local), f"{prefix}/{local.name}"]
    print("  $ " + " ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, check=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-dir", type=Path, required=True, help="prepare_gt_structures.py output")
    ap.add_argument("--prefix", default=BUCKET_PREFIX)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    archive = build_archive(args.gt_dir, args.gt_dir / "gt_structures.tar.gz")
    print(f"[publish] archive {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

    for name in ("gt_structures.tar.gz", "gt_index.jsonl", "gt_contacts.jsonl"):
        upload(args.gt_dir / name, args.prefix, dry_run=args.dry_run)

    print(f"[publish] done -> {args.prefix}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
