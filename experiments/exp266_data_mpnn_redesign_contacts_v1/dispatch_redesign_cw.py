# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Batch-priority Fray dispatch of the exp266 redesign to CoreWeave rno-2a.

Fans `redesign_worker_cw.py` out as N independent **1xH100** jobs (no gang) over
the idle part of rno2a's prepaid H100 fleet. Follows the root `AGENTS.md`
"Single-GPU inference fan-out" recipe; the differences from exp82 are:

* **No vLLM.** A stock PyTorch CUDA image plus four pip installs is enough, so
  there is no `VLLM_PORT` collision to dodge — nothing here binds a fixed port.
* **The task is CPU-heavy too.** Each task asks for 15 vCPU alongside its GPU:
  ProteinMPNN is ~0.1 s per backbone on the H100 while pyconfind is ~4.2 s of
  CPU, so a GPU with only a couple of cores next to it would sit idle. 8 tasks
  x 15 vCPU = 120 of a node's 128.
* **`proteinmpnn` is installed `--no-deps`** — it pins `numpy<2`, which fights
  the image's numpy and everything else we install. Its actual numpy use is
  basic (see `pyproject.toml`), and `tests/test_redesign.py` exercises the model
  under numpy 2.

Submit from the workstation (needs a <14-day marin-iris client, so run it from
the marin checkout's env)::

    uv run --project /home/bizon/git/marin python dispatch_redesign_cw.py \\
        --shards 28 --priority batch

Dry-run locally (build + print the JobRequests, no submit)::

    python dispatch_redesign_cw.py --shards 4 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import os
import sys
from pathlib import Path

from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

# iris PriorityBand enum (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3). fray maps
# JobRequest.priority straight to the iris band.
IRIS_PRIORITY_BAND_BATCH = 3

# Same guard as exp82/exp108/exp163: the frozen 0.99.dev fray has no `priority`
# field, so priority=3 would be silently dropped into the interactive band —
# and a data job of this size does not belong in the interactive band.
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line. Submit from a fresh marin checkout."
)

IMAGE = os.environ.get("EXP266_CW_IMAGE", "pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime")
MARINFOLD_GIT = os.environ.get(
    "EXP266_CW_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold",
)

S3_PREFIX = os.environ.get("EXP266_CW_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp266")
BACKBONES_GLOB = os.environ.get("EXP266_CW_BACKBONES", f"{S3_PREFIX}/backbones/*.parquet")
OUT_PREFIX = os.environ.get("EXP266_CW_OUT", f"{S3_PREFIX}/documents")
JOB_PREFIX = os.environ.get("EXP266_CW_JOB_PREFIX", "exp266-redesign")

WORK_DIR = "/tmp/exp266"

# Shipped into the pod verbatim; no workspace bundle, so these travel base64 in
# the bootstrap rather than through a `uv sync`.
WORKER_FILES = ("backbone.py", "redesign.py", "generate_rows.py", "redesign_worker_cw.py")

# iris injects CoreWeave's endpoint + credentials as an FSSPEC_S3 blob that only
# fsspec/s3fs reads; CoreWeave buckets use virtual-hosted addressing.
FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    'export FSSPEC_S3_ADDRESSING_STYLE="${FSSPEC_S3_ADDRESSING_STYLE:-virtual}"'
)


def _encoded_sources() -> str:
    here = Path(__file__).resolve().parent
    lines = []
    for name in WORKER_FILES:
        blob = base64.b64encode((here / name).read_bytes()).decode()
        lines.append(f'echo {blob} | base64 -d > {WORK_DIR}/{name}')
    return "\n".join(lines)


def build_bootstrap(*, shard_i: int, num_shards: int, cpu_workers: int,
                    max_batch_residues: int, limit_args: str) -> str:
    return f"""
set -euo pipefail
echo "[exp266-cw] host=$(hostname) shard={shard_i}/{num_shards} image={IMAGE}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}
echo "[exp266-cw] AWS_ENDPOINT_URL=${{AWS_ENDPOINT_URL:-unset}} iris_FSSPEC_S3=${{FSSPEC_S3:+present}}"

mkdir -p {WORK_DIR}
{_encoded_sources()}

# The pytorch *runtime* image ships no git, and pip needs it for the `git+`
# marinfold install ("Error [Errno 2] No such file or directory: 'git'"). The
# repo tarball is not an alternative: it carries an absolute symlink that pip
# refuses ("is a link to an absolute path"), and it is 134 MB.
apt-get update -qq && apt-get install -y -qq --no-install-recommends git

PY=python
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet fsspec s3fs boto3 pyarrow gemmi "pyconfind[fast]"
# --no-deps: proteinmpnn pins numpy<2 and would drag the image's stack backwards.
# Its numpy use is basic and the model is tested under numpy 2 (see pyproject).
$PY -m pip install --quiet --no-deps proteinmpnn
# marinfold likewise --no-deps: contacts_v1's generator needs only fsspec+numpy,
# and a full install repins transformers for no reason.
$PY -m pip install --quiet --no-deps "{MARINFOLD_GIT}"
$PY -c "from marinfold.document_structures.contacts_v1 import generate_document; \\
        import proteinmpnn, torch; \\
        print('[exp266-cw] deps OK, cuda:', torch.cuda.is_available())"

# pyconfind's numba backend must not fan out across the node: every co-located
# pod would spawn a full thread pool and they would fight for the same cores.
export NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec $PY {WORK_DIR}/redesign_worker_cw.py \\
    --input-glob "{BACKBONES_GLOB}" \\
    --out-prefix "{OUT_PREFIX}" \\
    --shard {shard_i}/{num_shards} \\
    --device cuda \\
    --cpu-workers {cpu_workers} \\
    --max-batch-residues {max_batch_residues}{limit_args}
""".strip()


def build_request(*, shard_i: int, num_shards: int, cpu: int, ram: str, disk: str,
                  cpu_workers: int, max_batch_residues: int, limit_args: str,
                  priority: int) -> JobRequest:
    return JobRequest(
        name=f"{JOB_PREFIX}-s{shard_i}of{num_shards}",
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(
                shard_i=shard_i, num_shards=num_shards, cpu_workers=cpu_workers,
                max_batch_residues=max_batch_residues, limit_args=limit_args)]),
        resources=ResourceConfig.with_gpu(
            "H100", count=1, image=IMAGE, cpu=cpu, ram=ram, disk=disk),
        # setup_scripts=[] disables iris's default `uv sync` setup step: we submit
        # with no workspace bundle, so there is no pyproject to sync and the step
        # would fail before the entrypoint ever runs.
        environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
        replicas=1,
        priority=priority,
        processes_per_task=1,
        max_retries_failure=3,
        # Batch band is preemptible; the worker resumes at the first output file
        # it has not written yet.
        max_retries_preemption=100,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=28,
                    help="Number of 1xH100 tasks. 8 per node, so 28 ~ 3.5 nodes; "
                         "check live idle GPUs before raising it.")
    ap.add_argument("--cpu", type=int, default=15,
                    help="vCPU per task. 8 tasks x 15 = 120 of a node's 128.")
    ap.add_argument("--ram", default="96g")
    ap.add_argument("--disk", default="64g")
    ap.add_argument("--cpu-workers", type=int, default=14,
                    help="Document processes per task; one core is left for the "
                         "GPU feeder.")
    ap.add_argument("--max-batch-residues", type=int, default=100_000)
    ap.add_argument("--max-files", type=int, default=None,
                    help="Smoke cap: staged files per task.")
    ap.add_argument("--only", default=None,
                    help="Comma-separated shard indices to (re)submit; default all.")
    ap.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    ap.add_argument("--cluster", default="cw-rno2a")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    limit_args = f" --max-files {args.max_files}" if args.max_files else ""
    priority = IRIS_PRIORITY_BAND_BATCH if args.priority == "batch" else 0
    wanted = (
        {int(x) for x in args.only.split(",")} if args.only
        else set(range(args.shards))
    )
    reqs = [
        build_request(shard_i=i, num_shards=args.shards, cpu=args.cpu, ram=args.ram,
                      disk=args.disk, cpu_workers=args.cpu_workers,
                      max_batch_residues=args.max_batch_residues,
                      limit_args=limit_args, priority=priority)
        for i in sorted(wanted)
    ]

    if args.dry_run:
        print(f"[exp266-cw] DRY RUN — {len(reqs)} JobRequests built, not submitting.")
        for r in reqs[:3]:
            # `device` is the CpuConfig|GpuConfig|TpuConfig union in current
            # fray; there is no flat `device_count`.
            print(f"  {r.name}: priority={r.priority} image={r.resources.image} "
                  f"cpu={r.resources.cpu} device={r.resources.device}")
        print("\n--- bootstrap for shard 0 ---")
        print(reqs[0].entrypoint.binary_entrypoint.args[1])
        return

    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    print(f"[exp266-cw] submitting {len(reqs)} jobs via the {args.cluster} "
          f"controller tunnel", flush=True)
    with open_iris_client(cluster_name=args.cluster, workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        jobs = [client.submit(r) for r in reqs]
        for job, req in zip(jobs, reqs, strict=True):
            print(f"  {req.name}: {getattr(job, 'id', job)}")
    print(f"[exp266-cw] submitted {len(jobs)} jobs; they are root jobs, so this "
          f"launcher can exit. Watch: iris --cluster={args.cluster} job list",
          file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
