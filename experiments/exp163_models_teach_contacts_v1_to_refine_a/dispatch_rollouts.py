# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct batch-priority Fray dispatch of the exp163 vLLM rollout worker (issue #163).

Fan out the contacts-v1 rollout generator across ``N`` single-H100 iris jobs on the
CoreWeave **rno-2a** cluster -- one job per shard -- at **batch** priority. This is
the CUDA/vLLM analogue of exp112's foreign-container recipe:

  * exp112 ran the NeMo NGC image + a ``torchrun`` binary entrypoint (training);
  * exp163 runs the **vLLM CUDA** image + a bash bootstrap that generates contact
    rollouts with vLLM on one H100 per shard (inference).

Novelty vs exp112: ``N`` independent single-GPU jobs (a fan-out) rather than one
gang, and the payload is generation, not training. Everything else is the same
foreign-container trick:

  * **image override** -> ``ResourceConfig.image`` = the vLLM CUDA image. In this
    fray build ``convert_environment`` does NOT forward ``EnvironmentConfig.docker_image``
    to iris; the pod image comes solely from ``task_image = request.resources.image``
    (``iris_backend.FrayIrisClient.submit``). So the load-bearing override is the
    ``image=`` kwarg on ``with_gpu`` -- WITHOUT it the pod would boot the cluster
    default (TPU/JAX) image and vLLM/CUDA would be absent. We ALSO pass
    ``create_environment(docker_image=...)`` (not ``workspace``) so create_environment
    does not default ``workspace`` to the launcher dir and try to ``uv sync`` our
    pyproject INTO the vLLM image (exp112 README design note);
  * ``Entrypoint.from_binary("bash", ["-lc", bootstrap])`` -- the vLLM container has
    no repo checkout, so the bootstrap **base64-inlines** BOTH
    ``gen_rollouts_worker_exp163.py`` and its ``rollout_metrics.py`` import, decodes
    them to ``/tmp/exp163``, ``pip install``s only the worker's I/O deps (vLLM + torch
    are already baked into the image), and runs the worker for this shard;
  * **batch band** via ``JobRequest.priority=3`` (iris ``PRIORITY_BAND_BATCH``),
    asserted at import exactly like exp108/exp112 so the frozen ``0.99.dev`` fray
    (whose ``JobRequest`` has no ``priority`` field) cannot silently drop us into the
    interactive band -- which would disrupt the very interactive users batch protects.

**Image tag.** The issue text suggested ``vllm/vllm-openai:v0.6.6``, but the exp163
model is **Qwen3** (``config.json`` ``architectures=["Qwen3ForCausalLM"]``,
``model_type="qwen3"``) and vLLM only added Qwen3 support in ~v0.8.5. v0.6.6 (Dec
2024) would abort with "architecture not supported", so we pin a recent stable
Qwen3-capable CUDA tag. Override with ``EXP163_IMAGE``.

**CoreWeave S3 addressing (critical).** CoreWeave AI Object Storage rejects
path-style requests (``PathStyleRequestNotAllowed``); s3fs must use virtual-hosted
addressing. iris injects the pod's S3 connection settings (endpoint + creds, and for
``cw-*`` the ``config_kwargs`` virtual-addressing block) via the ``iris-task-env``
secret / an ``FSSPEC_S3`` env blob. As belt-and-suspenders the bootstrap also exports
``FSSPEC_S3_CONFIG_KWARGS`` -- fsspec's ``FSSPEC_<PROTO>_<KWARG>`` env form
(``fsspec.config.set_conf_env``) -- which sets the s3 filesystem's ``config_kwargs``
kwarg; s3fs forwards it to botocore's ``AioConfig(s3={"addressing_style":"virtual"})``.
This surgically sets only ``conf["s3"]["config_kwargs"]``, so it does NOT clobber the
endpoint/creds iris injects (those live under other keys of the same ``FSSPEC_S3``
blob). No worker code change is needed -- see ``build_bootstrap``.

Run (as a tiny CPU **driver** job, like exp108/exp112 -- ``current_client`` then
resolves to the in-cluster controller and submits the N GPU shard jobs)::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    WK=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
    uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \
        --cpu=2 --memory=6GB --disk=16GB \
        -e WANDB_API_KEY "$WK" -e EXP163_NUM_SHARDS 2 \
        -- python -m dispatch_rollouts

Smoke first (one shard, first 4 targets of that shard)::

    uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \
        --cpu=2 --memory=6GB --disk=16GB \
        -e WANDB_API_KEY "$WK" -e EXP163_NUM_SHARDS 1 \
        -- python -m dispatch_rollouts --limit 4

Dry-run locally (build + print the JobRequests + shard-0 bootstrap, no submit)::

    EXP163_DRY_RUN=1 python -m dispatch_rollouts --num-shards 2
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import logging
import os
from pathlib import Path

from fray.types import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, JobStatus, create_environment

logger = logging.getLogger(__name__)

# iris PriorityBand enum value (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3). fray
# maps JobRequest.priority (int) straight to the iris band (iris_backend.submit():
# priority_band=request.priority).
IRIS_PRIORITY_BAND_BATCH = 3

# Fail loudly on the frozen 0.99.dev fray, whose JobRequest has no `priority` field,
# so `priority=3` would be silently dropped -> interactive band (which would disrupt
# the very interactive users batch priority protects). Same guard as exp108/exp112.
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line (exp112's pins), not the frozen 0.99.dev build."
)

# ---------------------------------------------------------------------------
# Container: the official vLLM CUDA image already ships CUDA vLLM + torch. Pinned
# to a recent stable Qwen3-capable tag (see the module docstring for why NOT
# v0.6.6). Override with EXP163_IMAGE.
# ---------------------------------------------------------------------------
VLLM_IMAGE = os.environ.get("EXP163_IMAGE", "vllm/vllm-openai:v0.9.2")

# ---------------------------------------------------------------------------
# I/O -- all on CoreWeave S3, readable/writable from the pods via iris-task-env.
# Mirrors the issue's exact worker command; every leg is env-overridable.
# ---------------------------------------------------------------------------
EXP163_S3_PREFIX = os.environ.get("EXP163_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp163")
MODEL_S3 = os.environ.get("EXP163_MODEL", f"{EXP163_S3_PREFIX}/model/step-35679")
VAL_PREFIX = os.environ.get("EXP163_VAL_PREFIX", f"{EXP163_S3_PREFIX}/val10k")
TARGETS_S3 = os.environ.get("EXP163_TARGETS", f"{VAL_PREFIX}/targets.parquet")
PROMPTS_S3 = os.environ.get("EXP163_PROMPTS", f"{VAL_PREFIX}/prompts")
OUT_S3 = os.environ.get("EXP163_OUT", f"{VAL_PREFIX}/runs")

# Generation knobs (issue defaults): 24 rollouts/target, top-k disabled (-1, the
# vLLM convention + the #142 under-generation fix), one H100 per shard so
# tensor-parallel size 1.
N_ROLLOUTS = int(os.environ.get("EXP163_N_ROLLOUTS", "24"))
TOP_K = int(os.environ.get("EXP163_TOP_K", "-1"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("EXP163_TP", "1"))

# The two worker files inlined into every pod's bootstrap (no repo checkout in the
# vLLM container). rollout_metrics.py is imported by the worker.
WORKER_SCRIPT = Path(__file__).with_name("gen_rollouts_worker_exp163.py")
METRICS_SCRIPT = Path(__file__).with_name("rollout_metrics.py")

# In-pod layout.
WORK_DIR = "/tmp/exp163"
WORKER_LOCAL = f"{WORK_DIR}/gen_rollouts_worker_exp163.py"
METRICS_LOCAL = f"{WORK_DIR}/rollout_metrics.py"

# Belt-and-suspenders virtual-hosted S3 addressing (see the module docstring). This
# is a NON-f string on purpose: the JSON braces are literal. fsspec's set_conf_env
# reads FSSPEC_S3_CONFIG_KWARGS as the s3 filesystem's `config_kwargs` kwarg and s3fs
# forwards it to botocore's AioConfig(s3={"addressing_style":"virtual"}). It sets ONLY
# config_kwargs, so it never clobbers the endpoint/creds iris puts in FSSPEC_S3.
FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def build_bootstrap(*, shard_i: int, num_shards: int, limit: int | None) -> str:
    """Bash bootstrap for one vLLM rollout pod (shard ``shard_i``/``num_shards``).

    Decodes the inlined worker + its ``rollout_metrics`` import to ``/tmp/exp163``,
    installs only the worker's fsspec/pyarrow/s3fs/boto3 I/O deps (vLLM + torch are
    baked into the image), forces virtual-hosted S3 addressing, then execs the worker
    for this shard. ``--limit`` is appended only for a smoke.
    """
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    metrics_b64 = base64.b64encode(METRICS_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    return f"""
set -euo pipefail
echo "[exp163-bootstrap] host=$(hostname) shard={shard_i}/{num_shards} image={VLLM_IMAGE}"
nvidia-smi -L || true

# CoreWeave object storage rejects path-style S3 -> force virtual-hosted addressing
# for the worker's plain fsspec/s3fs. Endpoint + creds still come from iris
# (iris-task-env / the FSSPEC_S3 blob); this only augments the config_kwargs sub-key.
{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}
echo "[exp163-bootstrap] AWS_ENDPOINT_URL=${{AWS_ENDPOINT_URL:-unset}} iris_FSSPEC_S3=${{FSSPEC_S3:+present}}"

mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORKER_LOCAL}
echo {metrics_b64} | base64 -d > {METRICS_LOCAL}
echo "[exp163-bootstrap] decoded worker ($(wc -l < {WORKER_LOCAL}) lines) + rollout_metrics"

# vLLM + torch are already in the image; add only the worker's I/O deps. `python -m
# pip` targets the same interpreter that runs the worker.
# Locate the image's python that has vLLM (image layout varies), install the
# worker's I/O deps into THAT python, and run the worker with it.
VLLM_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$_py"; break; fi
done
echo "[exp163-bootstrap] candidate pythons: $(which -a python python3 2>/dev/null | tr '\n' ' ')"
echo "[exp163-bootstrap] vLLM python: ${{VLLM_PY:-NONE}}"
if [ -z "$VLLM_PY" ]; then echo "[exp163-bootstrap] FATAL: no python imports vllm"; find / -maxdepth 7 -name vllm -type d 2>/dev/null | head; exit 3; fi
uv pip install --python "$VLLM_PY" --quiet fsspec pyarrow s3fs boto3 || "$VLLM_PY" -m pip install --quiet fsspec pyarrow s3fs boto3

# The worker does `import rollout_metrics`; running the script already puts its dir on
# sys.path[0], but pin PYTHONPATH too as belt-and-suspenders.
export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}

echo "[exp163-bootstrap] launching vLLM rollout worker for shard {shard_i}/{num_shards}"
exec "$VLLM_PY" {WORKER_LOCAL} \\
    --model {MODEL_S3} \\
    --targets {TARGETS_S3} \\
    --prompts {PROMPTS_S3} \\
    --out {OUT_S3} \\
    --shard {shard_i}/{num_shards} \\
    --n-rollouts {N_ROLLOUTS} \\
    --top-k {TOP_K} \\
    --tensor-parallel-size {TENSOR_PARALLEL_SIZE}{limit_arg}
""".strip()


def build_request(
    *,
    shard_i: int,
    num_shards: int,
    limit: int | None,
    env_vars: dict[str, str] | None = None,
) -> JobRequest:
    """One batch-band single-H100 iris JobRequest for shard ``shard_i``/``num_shards``."""
    # ONE H100 per shard. `image` here sets iris `task_image` (the actual pod image);
    # this is the load-bearing override of the cluster's TPU/JAX default -- see the
    # module docstring. cpu/ram/disk per the issue spec.
    resources = ResourceConfig.with_gpu(
        "H100", count=1, image=VLLM_IMAGE, cpu=16, ram="128g", disk="256g",
    )
    bootstrap = build_bootstrap(shard_i=shard_i, num_shards=num_shards, limit=limit)
    # docker_image (NOT workspace) so create_environment does not default workspace to
    # the launcher dir and uv-sync our pyproject INTO the vLLM image. In this fray
    # build docker_image is dropped at the fray->iris boundary (the pod image is
    # resources.image, above); passing it only steers create_environment's own
    # workspace-XOR-docker_image guard. Mirrors exp112.
    environment = create_environment(docker_image=VLLM_IMAGE, env_vars=env_vars or {})

    return JobRequest(
        name=f"exp163-val10k-rollouts-shard{shard_i}-of{num_shards}",
        entrypoint=Entrypoint.from_binary("bash", ["-lc", bootstrap]),
        resources=resources,
        environment=environment,
        replicas=1,                            # one single-H100 pod per shard job
        priority=IRIS_PRIORITY_BAND_BATCH,     # -> iris BATCH band (the whole point)
        processes_per_task=1,                  # single vLLM process (tensor-parallel=1)
        # The rollout worker is resume-safe: on restart it reads this shard's existing
        # part files and skips their entry_ids, so a retried preemption/failure just
        # resumes rather than redoing work.
        max_retries_failure=5,
        max_retries_preemption=100,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Dispatch exp163 vLLM contact-rollout shards to cw-rno2a (batch band)."
    )
    ap.add_argument(
        "--num-shards", type=int,
        default=int(os.environ.get("EXP163_NUM_SHARDS", "2")),
        help="number of single-H100 shard jobs to submit (default $EXP163_NUM_SHARDS or 2)",
    )
    ap.add_argument(
        "--limit", type=int, default=None,
        help="smoke: only the first N targets of each shard (forwarded as the worker's --limit)",
    )
    a = ap.parse_args()
    n = a.num_shards
    if n < 1:
        raise SystemExit(f"--num-shards must be >= 1, got {n}")

    # WANDB is not strictly needed for generation; forward it if present (create_environment
    # also picks it up from the driver's env, but pass it explicitly to be sure).
    env_vars: dict[str, str] = {}
    if os.environ.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    requests = [
        build_request(shard_i=i, num_shards=n, limit=a.limit, env_vars=env_vars)
        for i in range(n)
    ]

    print(
        f"[exp163] dispatch: {n} shard(s) x 1xH100 batch-band | image={VLLM_IMAGE} | "
        f"n_rollouts={N_ROLLOUTS} top_k={TOP_K} tp={TENSOR_PARALLEL_SIZE} limit={a.limit}\n"
        f"         model={MODEL_S3}\n"
        f"         out={OUT_S3}"
    )

    if os.environ.get("EXP163_DRY_RUN"):
        print("[exp163] DRY RUN -- JobRequests built, not submitting.")
        for req in requests:
            bs = req.entrypoint.binary_entrypoint.args[1]
            print(
                f"  {req.name}: priority={req.priority} replicas={req.replicas} "
                f"image={req.resources.image} gpu={req.resources.device.variant}x{req.resources.device.count} "
                f"cpu={req.resources.cpu} ram={req.resources.ram} disk={req.resources.disk} "
                f"bootstrap={len(bs)} chars"
            )
        print("\n[exp163] ----- shard-0 bootstrap -----")
        print(requests[0].entrypoint.binary_entrypoint.args[1])
        return

    # Runs as an in-cluster CPU driver job -> current_client() is the controller.
    client = current_client()
    handles = []
    for req in requests:
        job = client.submit(req)
        handles.append(job)
        print(f"[exp163] submitted {req.name} (job_id={job.job_id})", flush=True)

    # Wait for every shard, reporting per-shard status WITHOUT aborting siblings on one
    # failure (each shard is independent + resume-safe). Fail the driver at the end if
    # any shard did not succeed, so the batch result is visible in iris.
    results: dict[str, JobStatus] = {}
    for job in handles:
        status = job.wait(raise_on_failure=False)
        results[job.job_id] = status
        print(f"[exp163] {job.job_id}: {status}", flush=True)

    failed = [jid for jid, st in results.items() if st != JobStatus.SUCCEEDED]
    if failed:
        raise SystemExit(f"[exp163] {len(failed)}/{n} shard(s) did not succeed: {failed}")
    print(f"[exp163] all {n} shard(s) SUCCEEDED -> output under {OUT_S3}/")


if __name__ == "__main__":
    main()
