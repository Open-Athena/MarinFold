"""Publish exp326 raw rollout and timing parquets to the public HF bucket."""

import argparse
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESTINATION = (
    "hf://buckets/open-athena/MarinFold/data/exp326/contact-cluster-branching-v1"
)


def bucket_cli() -> list[str]:
    """Use an installed bucket-capable CLI or an isolated current release."""
    executable = os.environ.get("HF_CLI", "hf")
    probe = subprocess.run(
        [executable, "buckets", "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode == 0:
        return [executable]
    return [
        "uvx",
        "--from",
        "huggingface-hub==1.5.0",
        "--with",
        "click>=8",
        "hf",
    ]


def main() -> None:
    """Sync raw parquets without deleting unrelated bucket objects."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=HERE / "_cache")
    parser.add_argument("--destination", default=DESTINATION)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    command = [
        *bucket_cli(),
        "buckets",
        "sync",
        str(args.source),
        args.destination,
        "--include",
        "*/*/*.parquet",
    ]
    if args.dry_run:
        command.append("--dry-run")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
