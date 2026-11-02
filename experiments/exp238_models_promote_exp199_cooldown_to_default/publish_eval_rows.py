# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the cooldown's per-protein eval rows to the public bucket — issue #238.

#234 kept the cooldown evaluation's per-protein table on CoreWeave S3 and
checked only manifests and summaries into git. That is enough to quote a mean,
and not enough to draw #180's MarinFold-vs-Protenix scatter, which needs one
row per protein — so the figure that says *which proteins* the headline number
is made of cannot be re-pointed at the new frontier model without this.

The same rule as every other published artifact applies: a reader outside the
cluster, and every notebook, must be able to get it with no authentication.

Destination follows the layout #204 established for the earlier checkpoints:

    data/contacts-v1-model-eval-exp199/replicates/<run-id>/derived/
        <run-name>/step-<step>/<name>_rows.csv.gz

Gzipped with mtime 0 so the same input always produces the same bytes, and
`ROWS_SHA256` in exp180's plot_vs_protenix.py stays a usable drift check
instead of changing on every re-upload.

    uv run python publish_eval_rows.py --submit    # from the workstation
    uv run python publish_eval_rows.py             # on the pod
"""

import argparse
import gzip
import hashlib
import io
import os
import subprocess
import sys
import time
from pathlib import Path

from publish_cooldown import (
    BUCKET_ID,
    DEFAULT_IRIS,
    RUN_NAME,
    STEP,
    TARGET_CLUSTER,
    hf_token,
    log,
    s3_filesystem,
)

EVAL_RUN_ID = "cooldown-v2-20260815-01"
SOURCE_URI = (
    "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
    "exp199_optimize_contacts_v1_afdb_esm/evals/rollout_v2/"
    f"{EVAL_RUN_ID}/results/contact_precision_all.csv"
)
BUCKET_PATH = (
    f"data/contacts-v1-model-eval-exp199/replicates/{EVAL_RUN_ID}/derived/"
    f"{RUN_NAME}/step-{STEP}/contact_eval_cw_p06_cool_step{STEP}_rows.csv.gz"
)

# The universe #234 scored. 577 (dataset, stem) units x 4 ranges x 5 cuts.
EXPECTED_ROWS = 577 * 20


def fetch(filesystem) -> bytes:
    """Read the per-protein metric table out of the evaluation's S3 outputs."""
    with filesystem.open(SOURCE_URI.removeprefix("s3://"), "rb") as handle:
        payload = handle.read()
    header, *rows = payload.decode().splitlines()
    if len(rows) != EXPECTED_ROWS:
        raise SystemExit(
            f"FATAL: {SOURCE_URI} has {len(rows)} rows, expected {EXPECTED_ROWS} "
            f"(577 units x 4 ranges x 5 cuts). Header: {header}"
        )
    log(f"read {len(rows)} metric rows ({len(payload) / 2**20:.1f} MiB)")
    return payload


def compress(payload: bytes) -> bytes:
    """Gzip deterministically — same input, same bytes, same sha256."""
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def run() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN with open-athena write scope is required")
    from huggingface_hub import HfFileSystem

    started = time.time()
    compressed = compress(fetch(s3_filesystem()))
    digest = hashlib.sha256(compressed).hexdigest()
    destination = f"buckets/{BUCKET_ID}/{BUCKET_PATH}"
    with HfFileSystem(token=token).open(destination, "wb") as handle:
        handle.write(compressed)
    log(f"put {BUCKET_PATH} ({len(compressed) / 2**20:.1f} MiB, "
        f"{time.time() - started:.0f}s)")
    print(f"sha256 {digest}", flush=True)
    return 0


def submit(iris_bin: str) -> int:
    dirty = [
        line for line in subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True,
            check=True,
        ).stdout.splitlines() if not line.startswith("??")
    ]
    if dirty:
        raise SystemExit("refusing to submit with uncommitted changes here:\n  "
                         + "\n  ".join(dirty))
    argv = [
        iris_bin, "--cluster=marin", "job", "run",
        "--target-cluster", TARGET_CLUSTER,
        "--job-name", f"exp238-publish-eval-rows-step{STEP}",
        "--priority", "batch", "--enable-extra-resources", "--no-wait",
        "--cpu", "2", "--memory", "8GB", "--disk", "16GB",
        "--max-retries", "3", "--timeout", "3600",
        "-e", "HF_TOKEN", hf_token(),
        "--", "python", "publish_eval_rows.py",
    ]
    if subprocess.run(argv, cwd=Path(__file__).resolve().parent).returncode != 0:
        raise SystemExit("iris job run failed")
    log(f"submitted; logs: {iris_bin} --cluster=marin job logs "
        f"/bizon/exp238-publish-eval-rows-step{STEP}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--iris-bin", default=os.environ.get("IRIS_BIN", DEFAULT_IRIS))
    args = parser.parse_args()
    return submit(args.iris_bin) if args.submit else run()


if __name__ == "__main__":
    sys.exit(main())
