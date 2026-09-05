# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch ESM-Atlas backbone staging to CoreWeave.

CPU-only: the work is HF read + gemmi parse + encode, no model. Runs on
**cw-us-east-02a**, whose `cpu-genoa` pool is co-located with the
`marin-us-east-02a` bucket it writes to.

No GCP stage, unlike the AFDB arm: ESM-Atlas structures are inline
`cif_content` in a public HF bucket, so a CoreWeave pod reads them directly.

    uv run python dispatch_stage_esm_cw.py --shards 24 --dry-run
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

IMAGE = os.environ.get("EXP266_ESM_IMAGE", "python:3.12-slim")
S3_PREFIX = os.environ.get("EXP266_CW_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp266")
OUT_PREFIX = os.environ.get("EXP266_ESM_OUT", f"{S3_PREFIX}/esm_backbones")
JOB_PREFIX = os.environ.get("EXP266_ESM_JOB_PREFIX", "exp266-esm-stage")
MARINFOLD_GIT = os.environ.get(
    "EXP266_CW_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold",
)

WORK_DIR = "/tmp/exp266"
WORKER_FILES = ("backbone.py", "stage_rows.py", "stage_esm_atlas_cw.py")

FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def _encoded_sources() -> str:
    here = Path(__file__).resolve().parent
    return "\n".join(
        f'echo {base64.b64encode((here / n).read_bytes()).decode()} | base64 -d > {WORK_DIR}/{n}'
        for n in WORKER_FILES
    )


def build_bootstrap(*, shard_i: int, num_shards: int, limit: str) -> str:
    return f"""
set -euo pipefail
echo "[exp266-esm-stage] host=$(hostname) shard={shard_i}/{num_shards}"

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}

for attempt in 1 2 3; do
  apt-get update -qq && apt-get install -y -qq --no-install-recommends git && break
  echo "[exp266-esm-stage] apt attempt $attempt failed; retrying" >&2
  sleep $((attempt * 10))
done
if ! command -v git >/dev/null; then
  echo "[exp266-esm-stage] FATAL: git missing after 3 apt attempts" >&2
  exit 4
fi

mkdir -p {WORK_DIR}
{_encoded_sources()}

PY=python
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet gemmi "pyconfind[fast]" "numpy<2" fsspec s3fs boto3 \\
    pyarrow "huggingface_hub>=1.5"
$PY -m pip install --quiet --no-deps "{MARINFOLD_GIT}"

# pyconfind's numba backend must not fan out: several tasks share a node.
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec $PY {WORK_DIR}/stage_esm_atlas_cw.py \\
    --out-prefix "{OUT_PREFIX}" \\
    --shard {shard_i}/{num_shards}{limit}
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=24)
    ap.add_argument("--cpu", type=int, default=4)
    ap.add_argument("--ram", default="24g")
    ap.add_argument("--disk", default="32g")
    ap.add_argument("--max-shards", type=int, default=None, help="Smoke cap per task.")
    ap.add_argument("--only", default=None)
    ap.add_argument("--cluster", default="cw-us-east-02a")
    ap.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    limit = f" \\\n    --max-shards {args.max_shards}" if args.max_shards else ""
    priority = IRIS_PRIORITY_BAND_BATCH if args.priority == "batch" else 0
    wanted = ({int(x) for x in args.only.split(",")} if args.only
              else set(range(args.shards)))

    reqs = [
        JobRequest(
            name=f"{JOB_PREFIX}-s{i}of{args.shards}",
            entrypoint=Entrypoint.from_binary(
                "bash", ["-lc", build_bootstrap(shard_i=i, num_shards=args.shards,
                                                limit=limit)]),
            resources=ResourceConfig(cpu=args.cpu, ram=args.ram, disk=args.disk,
                                     image=IMAGE, preemptible=True),
            environment=create_environment(docker_image=IMAGE, env_vars={},
                                           setup_scripts=[]),
            replicas=1, priority=priority, processes_per_task=1,
            max_retries_failure=3, max_retries_preemption=100,
        )
        for i in sorted(wanted)
    ]

    if args.dry_run:
        print(f"[exp266-esm-stage] DRY RUN — {len(reqs)} jobs -> {OUT_PREFIX}")
        print(reqs[0].entrypoint.binary_entrypoint.args[1])
        return

    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    with open_iris_client(cluster_name=args.cluster, workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for r in reqs:
            client.submit(r)
            print(f"  submitted {r.name}")


if __name__ == "__main__":
    raise SystemExit(main())
