# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit `export_results.py` to CoreWeave so the run's results reach the bucket.

The results prefix is only readable from inside the cluster, so the export has
to run there. This is a 4-CPU job that finishes in under a minute.

    HF_TOKEN=... uv run python submit_export.py --run-id fbmono-20260818-01
"""
import argparse
import os
import subprocess
from pathlib import Path

from checkpoint_specs import MARIN_PREFIX
from submit_coreweave import DEFAULT_IRIS, TARGET_CLUSTER


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--subdir", default="results")
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    parser.add_argument("--user", default=os.environ.get("IRIS_USER", "eczech"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required")

    # The job's pinned huggingface_hub is <1 and has no bucket API, so the
    # export runs under `uv run --with` rather than the workspace environment.
    command = [
        args.iris_bin, "--cluster=marin", "job", "run",
        "--target-cluster", TARGET_CLUSTER,
        "--priority", "batch",
        "--enable-extra-resources",
        "--user", args.user,
        "--job-name", f"exp245-export-{args.run_id}-{args.subdir}",
        "--cpu", "4", "--memory", "16GB", "--disk", "32GB",
        "--max-retries", "2", "--timeout", "3600", "--no-wait",
        "-e", "MARIN_PREFIX", MARIN_PREFIX,
        "-e", "HF_TOKEN", token,
        "--",
        "bash", "-lc",
        "uv run --with 'huggingface_hub>=1.5' --with 's3fs==2026.1.0' "
        "--with 'fsspec==2026.1.0' --with 'aiobotocore==2.26.0' "
        f"python export_results.py --run-id {args.run_id} --subdir {args.subdir}",
    ]
    print(f"Submitting export for {args.run_id}/{args.subdir}")
    if args.dry_run:
        return
    subprocess.run(command, cwd=Path(__file__).parent, check=True)


if __name__ == "__main__":
    main()
