# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the analysis tables beside the scores they were computed from.

The CoreWeave driver already pushed the raw evaluation outputs to
``data/contacts-v1-msa-depth-exp260/<run-id>/``. This adds the depth
measurements and the joined tables under ``.../analysis/`` so the whole chain —
per-protein scores, per-protein depth, tiered means — is readable anonymously
from a notebook without a CoreWeave credential or a Modal account.

    uv run python publish_to_hf.py            # push
    uv run python publish_to_hf.py --dry-run  # list what would be pushed

Writing needs an ``open-athena``-scoped token (``hf auth whoami`` must list the
org); reading needs nothing.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import upstream as U

PUBLISHED = (
    "universe.csv",
    "msa_depth.csv",
    "depth_consistency.json",
    "per_protein_depth.csv",
    "depth_tiers.csv",
    "paired_deltas.csv",
    "tier_counts.csv",
    "low_msa_depth_set.csv",
)
DESTINATION = (
    f"hf://buckets/open-athena/MarinFold/{U.PUBLISH_PREFIX}/{U.RUN_ID}/analysis"
)


def hf_binary() -> str:
    """Return an ``hf`` CLI new enough to have the ``buckets`` subcommand."""

    candidate = Path(sys.executable).with_name("hf")
    if candidate.exists():
        return str(candidate)
    found = shutil.which("hf")
    if found is None:
        raise RuntimeError("no `hf` CLI available")
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    binary = hf_binary()
    for name in PUBLISHED:
        source = U.DATA / name
        if not source.exists():
            raise FileNotFoundError(f"{source} has not been built yet")
        target = f"{DESTINATION}/{name}"
        print(f"{source} -> {target}")
        if not args.dry_run:
            subprocess.run([binary, "buckets", "cp", str(source), target], check=True)


if __name__ == "__main__":
    main()
