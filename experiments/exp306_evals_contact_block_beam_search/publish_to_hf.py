#!/usr/bin/env python
"""Publish exp306 tables, figures, and raw rollouts to the public HF bucket."""

import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEST = "hf://buckets/open-athena/MarinFold/data/contact-block-beam-exp306"


def main() -> None:
    """Sync reproducible small outputs and auditable per-rollout parquets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sources = (
        (HERE / "data", "tables"),
        (HERE / "plots", "plots"),
        (HERE / "_cache" / "beam4" / "eval-val", "raw/beam4/eval-val"),
        (HERE / "_cache" / "beam4" / "foldswitch", "raw/beam4/foldswitch"),
        (HERE / "_cache" / "dev-beam4" / "foldswitch", "raw/beam4/foldswitch-dev"),
        (HERE / "_cache" / "beam8" / "foldswitch", "raw/beam8/foldswitch-dev"),
        (HERE / "_cache" / "beam8" / "eval-val", "raw/beam8/eval-val-pilot"),
    )
    for source, suffix in sources:
        if not source.exists():
            raise FileNotFoundError(source)
        command = ["hf", "buckets", "sync", str(source), f"{DEST}/{suffix}"]
        if args.dry_run:
            command.append("--dry-run")
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
