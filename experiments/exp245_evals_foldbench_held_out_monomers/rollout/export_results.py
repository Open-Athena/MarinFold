# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy one evaluation run's results out of CoreWeave S3 and onto the HF bucket.

The results land in ``s3://marin-us-east-02a/...`` and only jobs inside that
region have credentials for it, so the analysis on the workstation cannot read
them. This runs as a small CPU job in the cluster, reads the run's ``results/``
prefix, and pushes each file to the public bucket, where the analysis (and
anyone else) can fetch it anonymously.

Submit it with ``submit_export.py`` after the evaluation finishes; the driver
job's own environment cannot do this because its pinned ``huggingface_hub`` is
older than the ``buckets`` API.

    python export_results.py --run-id fbmono-20260818-01
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

import fsspec
from huggingface_hub import HfApi

from checkpoint_specs import BUCKET_PREFIX, run_root

BUCKET_REPO = "open-athena/MarinFold"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--subdir", default="results",
                        help="prefix under the run root to export")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is required to write to the bucket")

    root = f"{run_root(args.run_id)}/{args.subdir}"
    filesystem, path = fsspec.core.url_to_fs(root)
    files = [entry for entry in filesystem.find(path)]
    if not files:
        raise SystemExit(f"no files under {root}")

    api = HfApi(token=token)
    destination = f"{BUCKET_PREFIX}/runs/{args.run_id}"
    exported = []
    with tempfile.TemporaryDirectory() as scratch:
        for remote in files:
            name = remote[len(path.rstrip("/")) + 1:]
            local = Path(scratch) / name
            local.parent.mkdir(parents=True, exist_ok=True)
            filesystem.get_file(remote, str(local))
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=f"{destination}/{name}",
                repo_id=BUCKET_REPO,
                repo_type="bucket",
            )
            exported.append({"name": name, "bytes": local.stat().st_size})
            print(json.dumps({"event": "exported", **exported[-1]}), flush=True)
    print(json.dumps({"event": "complete", "run_id": args.run_id,
                      "destination": destination, "files": len(exported)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
