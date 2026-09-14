# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch the HF-bucket publish to CoreWeave.

CPU-only and CoreWeave-side on purpose. The corpus is 64 GB in CoreWeave
object storage; the workstation uplink is ~2.5 MB/s, so a round trip through
here would take days. A pod streams shard-by-shard straight to
huggingface.co.

Runs on **cw-us-east-02a**, whose `cpu-genoa` pool is co-located with the
`marin-us-east-02a` bucket and had ~735 idle vCPU. It deliberately does *not*
take a 1xH100 on rno2a: the job is pure I/O and would leave the GPU idle for
hours, free or not.

The HF token goes through `create_environment(env_vars=...)`, not the
bootstrap string, so it does not end up in the job's command line.

    HF_TOKEN=$(...) uv run python dispatch_publish_cw.py --dry-run
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import os
from pathlib import Path

from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

IRIS_PRIORITY_BAND_BATCH = 3

assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; submit from a fresh marin checkout."
)

IMAGE = os.environ.get("EXP266_PUBLISH_IMAGE", "python:3.12-slim")
S3_PREFIX = os.environ.get("EXP266_CW_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp266")
DOCUMENTS_GLOB = os.environ.get("EXP266_PUBLISH_DOCS", f"{S3_PREFIX}/documents/*.parquet")
REPO_PATH = os.environ.get(
    "EXP266_PUBLISH_REPO_PATH",
    "data/document_structures/contacts_v1_mpnn_redesign/train",
)
JOB_NAME = os.environ.get("EXP266_PUBLISH_JOB", "exp266-publish")

WORK_DIR = "/tmp/exp266"
WORKER_FILES = ("publish_to_hf.py",)

FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def _encoded_sources() -> str:
    here = Path(__file__).resolve().parent
    return "\n".join(
        f'echo {base64.b64encode((here / n).read_bytes()).decode()} | base64 -d > {WORK_DIR}/{n}'
        for n in WORKER_FILES
    )


def build_bootstrap(*, workers: int) -> str:
    return f"""
set -euo pipefail
echo "[exp266-publish] host=$(hostname) image={IMAGE}"

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}
echo "[exp266-publish] HF_TOKEN=${{HF_TOKEN:+present}} FSSPEC_S3=${{FSSPEC_S3:+present}}"

mkdir -p {WORK_DIR}
{_encoded_sources()}

PY=python
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet "huggingface_hub>=1.5" fsspec s3fs boto3

export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec $PY {WORK_DIR}/publish_to_hf.py \\
    --documents-glob "{DOCUMENTS_GLOB}" \\
    --repo-path "{REPO_PATH}" \\
    --workers {workers}
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--ram", default="32g")
    ap.add_argument("--disk", default="64g")
    ap.add_argument("--workers", type=int, default=4,
                    help="Upload threads. exp139 saw sustained 429s from HF's "
                         "bucket endpoint at 16 and found 4 both stable and "
                         "faster overall.")
    ap.add_argument("--cluster", default="cw-us-east-02a")
    ap.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        raise SystemExit("HF_TOKEN required (must be open-athena scoped)")

    req = JobRequest(
        name=JOB_NAME,
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(workers=args.workers)]),
        resources=ResourceConfig(cpu=args.cpu, ram=args.ram, disk=args.disk,
                                 image=IMAGE, preemptible=False),
        # Token via env_vars, never the bootstrap string: the command line shows
        # up in job metadata and logs.
        environment=create_environment(docker_image=IMAGE,
                                       env_vars={"HF_TOKEN": token or ""},
                                       setup_scripts=[]),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH if args.priority == "batch" else 0,
        processes_per_task=1,
        max_retries_failure=3,
        # Not preemptible and idempotent-ish: HF upload_file overwrites, so a
        # retry re-uploads rather than corrupting.
        max_retries_preemption=20,
    )

    if args.dry_run:
        print(f"[exp266-publish] DRY RUN -> {REPO_PATH}")
        print(req.entrypoint.binary_entrypoint.args[1])
        return

    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    with open_iris_client(cluster_name=args.cluster, workspace=None) as iris_client:
        FrayIrisClient.from_iris_client(iris_client).submit(req)
        print(f"  submitted {req.name} to {args.cluster}")


if __name__ == "__main__":
    raise SystemExit(main())
