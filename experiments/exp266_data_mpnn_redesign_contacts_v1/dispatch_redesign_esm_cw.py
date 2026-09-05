# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch the ESM-Atlas redesign to CoreWeave rno-2a.

Same 1xH100 no-gang fan-out as `dispatch_redesign_cw.py`, pointed at
`redesign_esm_cw.py`. Differences from the AFDB arm:

* **One stage, not three.** The worker reads source cif from the public HF
  bucket itself, so there is no staged-backbone input to prepare.
* **2 designs, not 8** (`--temperatures 0.1 0.2`). exp266's AFDB run measured
  the 8-slot ladder to span almost nothing, and ESM-Atlas's distinctive value
  is 65 M non-redundant backbones rather than more sequences per backbone.
* Reads ~2.08 TB from HF across the whole fan-out, one pass, spread over tasks.

    uv run python dispatch_redesign_esm_cw.py --shards 1 --max-shards 1 --dry-run
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

IMAGE = os.environ.get("EXP266_CW_IMAGE", "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
MARINFOLD_GIT = os.environ.get(
    "EXP266_CW_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold",
)
S3_PREFIX = os.environ.get("EXP266_CW_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp266")
OUT_PREFIX = os.environ.get("EXP266_ESM_OUT", f"{S3_PREFIX}/esm_documents")
JOB_PREFIX = os.environ.get("EXP266_ESM_JOB_PREFIX", "exp266-esm")

WORK_DIR = "/tmp/exp266"
WORKER_FILES = ("backbone.py", "redesign.py", "generate_rows.py", "stage_rows.py",
                "redesign_esm_cw.py")

FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def _encoded_sources() -> str:
    here = Path(__file__).resolve().parent
    return "\n".join(
        f'echo {base64.b64encode((here / n).read_bytes()).decode()} | base64 -d > {WORK_DIR}/{n}'
        for n in WORKER_FILES
    )


def build_bootstrap(*, shard_i: int, num_shards: int, cpu_workers: int,
                    temperatures: list[float], max_batch_residues: int,
                    limit: str) -> str:
    temps = " ".join(str(t) for t in temperatures)
    return f"""
set -euo pipefail
echo "[exp266-esm] host=$(hostname) shard={shard_i}/{num_shards} image={IMAGE}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}

for attempt in 1 2 3; do
  apt-get update -qq && apt-get install -y -qq --no-install-recommends git && break
  echo "[exp266-esm] apt attempt $attempt failed; retrying" >&2
  sleep $((attempt * 10))
done
if ! command -v git >/dev/null; then
  echo "[exp266-esm] FATAL: git missing after 3 apt attempts" >&2
  exit 4
fi

mkdir -p {WORK_DIR}
{_encoded_sources()}

PY=python
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet fsspec s3fs boto3 pyarrow gemmi "pyconfind[fast]" \\
    "huggingface_hub>=1.5"
$PY -m pip install --quiet --no-deps proteinmpnn
$PY -m pip install --quiet --no-deps "{MARINFOLD_GIT}"
$PY -c "from marinfold.document_structures.contacts_v1 import generate_document; \\
        import proteinmpnn, torch; \\
        print('[exp266-esm] deps OK, cuda:', torch.cuda.is_available())"

export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec $PY {WORK_DIR}/redesign_esm_cw.py \\
    --out-prefix "{OUT_PREFIX}" \\
    --shard {shard_i}/{num_shards} \\
    --temperatures {temps} \\
    --device cuda \\
    --cpu-workers {cpu_workers} \\
    --max-batch-residues {max_batch_residues}{limit}
""".strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=112)
    ap.add_argument("--cpu", type=int, default=15)
    ap.add_argument("--ram", default="96g")
    ap.add_argument("--disk", default="64g")
    ap.add_argument("--cpu-workers", type=int, default=14)
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.1, 0.2])
    ap.add_argument("--max-batch-residues", type=int, default=100_000)
    ap.add_argument("--max-shards", type=int, default=None, help="Smoke cap per task.")
    ap.add_argument("--only", default=None)
    ap.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    ap.add_argument("--cluster", default="cw-rno2a")
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
                "bash", ["-lc", build_bootstrap(
                    shard_i=i, num_shards=args.shards, cpu_workers=args.cpu_workers,
                    temperatures=args.temperatures,
                    max_batch_residues=args.max_batch_residues, limit=limit)]),
            resources=ResourceConfig.with_gpu("H100", count=1, image=IMAGE,
                                              cpu=args.cpu, ram=args.ram,
                                              disk=args.disk),
            environment=create_environment(docker_image=IMAGE, env_vars={},
                                           setup_scripts=[]),
            replicas=1, priority=priority, processes_per_task=1,
            max_retries_failure=3, max_retries_preemption=100,
        )
        for i in sorted(wanted)
    ]

    if args.dry_run:
        print(f"[exp266-esm] DRY RUN — {len(reqs)} jobs, "
              f"{len(args.temperatures)} designs -> {OUT_PREFIX}")
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
