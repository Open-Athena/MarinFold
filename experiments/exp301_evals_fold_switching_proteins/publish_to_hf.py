#!/usr/bin/env python
# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish exp301's artifacts to the public open-athena/MarinFold bucket.

The small tables live in git; this puts the things git should not hold -- the
per-rollout parquets and the eval universe -- somewhere a reader outside the
cluster can fetch without credentials, which is what the notebook reads.

Destination: ``data/contacts-v1-fold-switching-exp301/``

Needs an **open-athena-scoped** write token (`hf auth whoami` must list the org;
the bare workstation token may be timodonnell-only). Reads are anonymous.

    uv run --with "huggingface_hub>=1.5" python publish_to_hf.py --dry-run
    uv run --with "huggingface_hub>=1.5" python publish_to_hf.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SCORES = Path("/data/exp301/scores/exp277")
DEST = "hf://buckets/open-athena/MarinFold/data/contacts-v1-fold-switching-exp301"

#: Committed tables that a reader needs alongside the bulk artifacts.
TABLES = [
    "foldswitch_universe.jsonl",
    "premise_gate.csv",
    "noise_floor.csv",
    "eval_targets.parquet",
    "training_hits.tsv",
    "training_fold_labels.csv",
    "fold_preference.csv",
    "bimodality.csv",
    "delta_nll.csv",
    "memorization.csv",
    "calibration.csv",
]


def plan() -> list[tuple[Path, str]]:
    """(local, remote) pairs to upload."""
    items: list[tuple[Path, str]] = []
    for name in TABLES:
        path = DATA / name
        if path.exists():
            items.append((path, f"{DEST}/{name}"))
    for kind in ("rollouts", "votes", "nll"):
        for path in sorted((SCORES / kind).glob("*.parquet")):
            items.append((path, f"{DEST}/scores/{kind}/{path.name}"))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    items = plan()
    total = sum(p.stat().st_size for p, _ in items)
    print(f"{len(items)} files, {total / 2**20:.1f} MiB -> {DEST}")
    for path, remote in items[:12]:
        print(f"   {path.stat().st_size / 2**10:9.1f} KiB  {remote.rsplit('/', 2)[-2]}/{path.name}")
    if len(items) > 12:
        print(f"   ... and {len(items) - 12} more")
    if args.dry_run:
        print("dry run — nothing uploaded")
        return 0

    import fsspec

    fs, _ = fsspec.core.url_to_fs(DEST)
    for n, (path, remote) in enumerate(items, 1):
        fs.put_file(str(path), remote.replace("hf://", ""))
        if n % 25 == 0 or n == len(items):
            print(f"   uploaded {n}/{len(items)}", flush=True)
    print(f"done -> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
