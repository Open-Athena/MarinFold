# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish exp277 rollout-v2 result assets to the public MarinFold HF bucket.

Run from this experiment directory:

    uv run --frozen python publish_to_hf.py --dry-run
    uv run --frozen python publish_to_hf.py
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data" / "eval_rollout_v2"
DESTINATION = (
    "hf://buckets/open-athena/MarinFold/data/exp277-models-single-mpnn-pilot/"
    "evals/rollout-v2/2026-09-13/v2-01/results"
)
PUBLISHED = tuple(
    sorted(path.name for path in SOURCE.iterdir() if path.suffix in {".csv", ".json"})
)


def hf_binary() -> str:
    """Return an ``hf`` CLI that supports bucket uploads."""

    candidates = [Path(sys.executable).with_name("hf")]
    discovered = shutil.which("hf")
    if discovered is not None:
        candidates.append(Path(discovered))
    for candidate in candidates:
        if not candidate.exists():
            continue
        result = subprocess.run(
            [str(candidate), "buckets", "--help"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return str(candidate)
    raise RuntimeError("no `hf` CLI with bucket support is available")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    binary = hf_binary()

    for name in PUBLISHED:
        source = SOURCE / name
        if not source.exists():
            raise FileNotFoundError(source)
        target = f"{DESTINATION}/{name}"
        print(f"{source} -> {target}")
        if not args.dry_run:
            subprocess.run([binary, "buckets", "cp", str(source), target], check=True)

    for source in sorted((HERE / "plots").iterdir()):
        if source.suffix not in {".png", ".svg", ".pdf", ".json"}:
            continue
        target = f"{DESTINATION.rsplit('/', 1)[0]}/plots/{source.name}"
        print(f"{source} -> {target}")
        if not args.dry_run:
            subprocess.run([binary, "buckets", "cp", str(source), target], check=True)


if __name__ == "__main__":
    main()
