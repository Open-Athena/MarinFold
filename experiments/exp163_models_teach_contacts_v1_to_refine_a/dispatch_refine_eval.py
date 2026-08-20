# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
"""Direct batch-priority Fray dispatch of the exp163 Phase-3 eval (issue #163).

Fans ``eval_refiner_worker.py`` out over single-H100 iris jobs on cw-rno2a — one
job per (model, shard) — at **batch** priority. Same foreign-container recipe as
``dispatch_rollouts.py``: the ``vllm/vllm-openai`` CUDA image already carries
torch + transformers, the worker is base64-inlined into a bash bootstrap (pods
have no repo checkout), and ``JobRequest.priority=3`` puts it in the batch band.

Unlike the rollout dispatcher this runs **HF transformers**, not vLLM — the eval
needs teacher-forced logits (``score_matrix``), not generation. The image is
reused only because it already has a CUDA torch stack.

Why not run it locally: the workstation's single A5000 is often occupied, and the
eval is ~3 teacher-forced passes per protein over 554 proteins × 2 models. On
H100s at 4 shards/model it is minutes.

Models evaluated (override with EXP163_EVAL_MODELS as ``name=uri,name=uri``):

* ``base``    — the E8 export the rollouts were generated from (the control)
* ``refiner`` — the Phase-2 1e-4 checkpoint's auto-written HF export

Run as a tiny CPU driver job::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \\
        --enable-extra-resources --cpu=2 --memory=6GB --disk=16GB \\
        -- python -m dispatch_refine_eval --shards 4

Dry-run locally (build + print the JobRequests, no submit)::

    EXP163_DRY_RUN=1 python -m dispatch_refine_eval --shards 2
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import logging
import os
from pathlib import Path

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, JobStatus, ResourceConfig, create_environment

logger = logging.getLogger(__name__)

IRIS_PRIORITY_BAND_BATCH = 3
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line, not the frozen 0.99.dev build."
)

IMAGE = os.environ.get("EXP163_IMAGE", "vllm/vllm-openai:v0.9.2")
S3_PREFIX = os.environ.get("EXP163_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp163")
EVAL_PREFIX = os.environ.get("EXP163_EVAL_PREFIX", f"{S3_PREFIX}/eval554")

DEFAULT_MODELS = {
    "base": f"{S3_PREFIX}/model/step-35679",
    "refiner": f"{S3_PREFIX}/checkpoints/plm-exp163-refine-cv1-1_5b-lr1e-4-e1-cos/hf/step-51",
}

WORKER = Path(__file__).with_name("eval_refiner_worker.py")
WORK_DIR = "/tmp/exp163_eval"
WORKER_LOCAL = f"{WORK_DIR}/eval_refiner_worker.py"

FSSPEC_VIRTUAL = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def models() -> dict[str, str]:
    spec = os.environ.get("EXP163_EVAL_MODELS")
    if not spec:
        return dict(DEFAULT_MODELS)
    return dict(kv.split("=", 1) for kv in spec.split(",") if kv.strip())


def build_bootstrap(*, name: str, uri: str, shard_i: int, shards: int, ks: str,
                    limit: int | None, fmt: str = "candidate", mode_id: int | None = None,
                    k0_only: bool = False) -> str:
    worker_b64 = base64.b64encode(WORKER.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    mode_arg = f" --mode-id {mode_id}" if mode_id is not None else ""
    k0_arg = " --k0-only" if k0_only else ""
    return f"""
set -euo pipefail
echo "[exp163-eval] host=$(hostname) model={name} shard={shard_i}/{shards}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL}

mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

# Locate the image's python that has torch, and install only the worker's I/O deps.
PY=""
for _py in /usr/bin/python3 /app/.venv/bin/python /usr/local/bin/python /opt/venv/bin/python python3 python; do
  if "$_py" -c "import torch, transformers" >/dev/null 2>&1; then PY="$_py"; break; fi
done
echo "[exp163-eval] torch python: ${{PY:-NONE}}"
if [ -z "$PY" ]; then echo "[exp163-eval] FATAL: no python imports torch+transformers"; exit 3; fi
uv pip install --python "$PY" --quiet fsspec pyarrow s3fs boto3 || \
  "$PY" -m pip install --quiet fsspec pyarrow s3fs boto3

exec "$PY" {WORKER_LOCAL} \\
    --model {uri} \\
    --targets {EVAL_PREFIX}/targets.parquet \\
    --prompts {EVAL_PREFIX}/prompts \\
    --rollouts {EVAL_PREFIX}/runs/rollout_metrics \\
    --out {EVAL_PREFIX}/scores/{name} \\
    --shard {shard_i}/{shards} \\
    --ks {ks} \\
    --format {fmt}{mode_arg}{k0_arg}{limit_arg}
""".strip()


def build_request(*, name: str, uri: str, shard_i: int, shards: int, ks: str,
                  limit: int | None, fmt: str = "candidate", mode_id: int | None = None,
                  k0_only: bool = False) -> JobRequest:
    resources = ResourceConfig.with_gpu("H100", count=1, image=IMAGE, cpu=16,
                                        ram="128g", disk="256g")
    bootstrap = build_bootstrap(name=name, uri=uri, shard_i=shard_i, shards=shards,
                                ks=ks, limit=limit, fmt=fmt, mode_id=mode_id,
                                k0_only=k0_only)
    environment = create_environment(docker_image=IMAGE, env_vars={})
    return JobRequest(
        name=f"exp163-eval-{name}-shard{shard_i}-of{shards}",
        entrypoint=Entrypoint.from_binary("bash", ["-lc", bootstrap]),
        resources=resources,
        environment=environment,
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=3,
        max_retries_preemption=100,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Dispatch exp163 Phase-3 eval shards (batch band).")
    ap.add_argument("--shards", type=int, default=int(os.environ.get("EXP163_EVAL_SHARDS", "4")))
    ap.add_argument("--ks", default="2,4,8,16",
                    help="candidate-block counts to sweep. K x n_cap must stay inside "
                         "the TRAINING candidate-context distribution (max 1,282 contacts "
                         "/ 3,862 tokens); past it the context is out of distribution and "
                         "scores below random.")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N proteins per shard")
    ap.add_argument("--format", choices=["candidate", "multi-draft"], default="candidate",
                    help="MUST match the corpus the checkpoints were trained on")
    ap.add_argument("--k0-only", action="store_true",
                    help="forwarded to the worker: no-draft prefixes only (~4x cheaper)")
    ap.add_argument("--mode-id", type=int, default=None,
                    help="v3 refine-mode doc-type sentinel id (7); also records base-mode K0")
    a = ap.parse_args()

    mods = models()
    requests = [
        build_request(name=n, uri=u, shard_i=i, shards=a.shards, ks=a.ks, limit=a.limit,
                      fmt=a.format, mode_id=a.mode_id, k0_only=a.k0_only)
        for n, u in mods.items()
        for i in range(a.shards)
    ]
    print(f"[exp163-eval] {len(mods)} model(s) x {a.shards} shard(s) = {len(requests)} jobs | "
          f"ks={a.ks} format={a.format} mode_id={a.mode_id} limit={a.limit} image={IMAGE}")
    for n, u in mods.items():
        print(f"    {n:>8}: {u}")
    print(f"    out: {EVAL_PREFIX}/scores/<model>/")

    if os.environ.get("EXP163_DRY_RUN"):
        print("[exp163-eval] DRY RUN -- JobRequests built, not submitting.")
        for r in requests:
            print(f"  {r.name}: priority={r.priority} gpu={r.resources.device.variant}"
                  f"x{r.resources.device.count} image={r.resources.image}")
        return

    client = current_client()
    handles = []
    for req in requests:
        job = client.submit(req)
        handles.append((req.name, job))
        print(f"[exp163-eval] submitted {req.name} (job_id={job.job_id})", flush=True)

    results: dict[str, JobStatus] = {}
    for name, job in handles:
        results[name] = job.wait(raise_on_failure=False)
        print(f"[exp163-eval] {name}: {results[name]}", flush=True)

    failed = [n for n, st in results.items() if st != JobStatus.SUCCEEDED]
    if failed:
        raise SystemExit(f"[exp163-eval] {len(failed)}/{len(handles)} job(s) failed: {failed}")
    print(f"[exp163-eval] all {len(handles)} job(s) SUCCEEDED -> {EVAL_PREFIX}/scores/")


if __name__ == "__main__":
    main()
