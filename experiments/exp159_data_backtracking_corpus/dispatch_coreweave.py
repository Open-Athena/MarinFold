# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fan the ESM-Atlas backtracking generation out over CoreWeave H100s (#159).

Runs *inside* a tiny CPU driver job on iris; submits one **independent 1-GPU
job per worker** and waits for them all.

Three things here are load-bearing and easy to get wrong:

1. **Batch priority must be set on the child JobRequest.** ``iris job run
   --priority batch`` sets the band of the *driver* only — it does not
   propagate to jobs the driver submits. Every child therefore carries
   ``priority=IRIS_PRIORITY_BAND_BATCH`` explicitly. (Standing instruction:
   everything we run on a CoreWeave GPU cluster goes at batch priority.)
2. **Independent jobs, not a gang.** ``replicas=N`` on a GPU ResourceConfig
   turns the submission into a co-scheduled gang with all-or-nothing Kueue
   admission (and is unreliable past ~4 nodes). Our shards are embarrassingly
   parallel, so we submit N separate ``replicas=1`` jobs that can start,
   preempt and retry independently.
3. **The driver must wait.** Child jobs are children of this driver; if it
   exits, iris finalizes (kills) them.

Batch band is preemptible, so ``max_retries_preemption`` is high and each
worker skips shards whose output parquet already exists — a preempted worker
resumes rather than redoing finished shards.

Submit (from the workstation, with the FRESH iris client — the default
/home/bizon/git/marin checkout fails the 14-day client-freshness gate)::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    KUBECONFIG=~/.kube/coreweave-iris-rno2a \\
    /home/bizon/git/marin-freshiris/.venv/bin/iris --cluster=cw-rno2a job run \\
        --no-wait --priority batch --cpu=2 --memory=8GB --disk=32GB \\
        -e EXP159_NUM_WORKERS 64 -e EXP159_SHARDS 0-255 \\
        -- python -m dispatch_coreweave
"""

from __future__ import annotations

import dataclasses
import os

# Import everything from fray.types / fray.current_client: some fray builds
# export nothing at the `fray` top level, so `from fray import ResourceConfig`
# is not portable across the dev line.
from fray.current_client import current_client
from fray.types import (
    Entrypoint,
    JobRequest,
    ResourceConfig,
    create_environment,
)

# iris PriorityBand enum value (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3).
IRIS_PRIORITY_BAND_BATCH = 3

# CoreWeave S3 is auto-injected into task pods (AWS_* + FSSPEC_S3 via the
# iris-task-env secret), so workers write s3:// paths with no extra config.
DEFAULT_OUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp159_backtracking_esm_atlas/documents"
)


def build_request(worker_id: int, num_workers: int, args: dict) -> JobRequest:
    """One 1-GPU worker over its stripe of the shard range, at batch priority."""
    # count=1 (not a whole 8-GPU node): the k8s request is a plain
    # nvidia.com/gpu, so 8 of these pack onto one node. disk is explicit —
    # CoreWeave pods default to 5Gi ephemeral, far too small for a 1.5B
    # checkout plus parquet staging.
    resources = ResourceConfig.with_gpu(
        "H100", count=1, replicas=1, cpu=16, ram="128g", disk="200g",
    )
    environment = create_environment(
        env_vars={
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "8",
            # uv's cache and the venv are on different filesystems in the pod;
            # copy mode avoids a hardlink warning per package.
            "UV_LINK_MODE": "copy",
        },
        extras=["gpu"],
    )
    # `uv run` (not bare `python`): the workspace env is a project .venv that a
    # plain `bash -lc python` does not resolve to, so the worker would start on
    # the system interpreter and fail on the first project import.
    command = (
        "uv run python gen_esm_atlas_worker.py "
        f"--shards {args['shards']} "
        f"--worker-id {worker_id} --num-workers {num_workers} "
        f"--docs-per-shard {args['docs_per_shard']} "
        f"--chunk-docs {args['chunk_docs']} "
        f"--batch {args['batch']} "
        f"--noise-prob {args['noise_prob']} "
        f"--flush {args['flush']} "
        f"--force-true-prob {args['force_true_prob']} "
        f"--out {args['out']}"
    )
    return JobRequest(
        name=f"exp159-bt-{args['flush']}-p{int(args['force_true_prob'] * 100):02d}"
             f"-{worker_id:03d}",
        entrypoint=Entrypoint.from_binary("bash", ["-lc", command]),
        resources=resources,
        environment=environment,
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=10,
        # `max_task_failures` is a SEPARATE field that defaults to 0 — without
        # it a worker dies on its FIRST failure regardless of
        # max_retries_failure. On this cluster GPU reclamation arrives as a
        # SIGTERM (exit 143) recorded as a *failure*, not a preemption, so
        # max_retries_preemption never applies and every reclaimed worker was
        # being killed outright. Retry generously; a resumed worker skips the
        # parts it already wrote.
        max_task_failures=30,
        # Batch band is preemptible; workers resume by skipping finished parts.
        max_retries_preemption=100,
    )


def main() -> None:
    assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
        "This fray build lacks JobRequest.priority; batch-band dispatch needs "
        "the 0.2.x.dev fray line, not a frozen marin-*-latest build."
    )

    num_workers = int(os.environ.get("EXP159_NUM_WORKERS", "64"))
    args = {
        "shards": os.environ.get("EXP159_SHARDS", "0-255"),
        "docs_per_shard": int(os.environ.get("EXP159_DOCS_PER_SHARD", "4000")),
        "chunk_docs": int(os.environ.get("EXP159_CHUNK_DOCS", "250")),
        "batch": int(os.environ.get("EXP159_BATCH", "48")),
        "noise_prob": float(os.environ.get("EXP159_NOISE_PROB", "0.05")),
        # Closing-flush mode. "none" is the arm; "shuffled" is the control that
        # isolates "the flush's ordering" from "the flush at all". Both must be
        # dispatched to DIFFERENT --out prefixes: the worker's resume logic
        # skips parts that already exist, so a shared prefix would have each arm
        # silently adopt the other's documents.
        "flush": os.environ.get("EXP159_FLUSH", "none"),
        "force_true_prob": float(os.environ.get("EXP159_FORCE_TRUE", "0.0")),
        "out": os.environ.get("EXP159_OUT", DEFAULT_OUT),
    }
    print(f"dispatching {num_workers} workers: {args}", flush=True)

    client = current_client()
    jobs = []
    for worker_id in range(num_workers):
        job = client.submit(build_request(worker_id, num_workers, args))
        jobs.append(job)
        print(f"submitted worker {worker_id} -> {job}", flush=True)

    print(f"waiting on {len(jobs)} workers (driver must not exit)", flush=True)
    failures = 0
    for worker_id, job in enumerate(jobs):
        try:
            job.wait(raise_on_failure=True)
            print(f"worker {worker_id} finished", flush=True)
        except Exception as exc:  # keep waiting on the rest; report at the end
            failures += 1
            print(f"worker {worker_id} FAILED: {exc}", flush=True)
    print(f"all workers done ({failures} failed)", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
