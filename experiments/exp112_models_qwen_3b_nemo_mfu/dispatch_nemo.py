# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Direct batch-priority Fray dispatch of the exp112 NeMo benchmark (issue #112).

Submits ONE 8xH100 iris job at **batch** priority whose container is the NeMo
NGC image and whose entrypoint is a ``torchrun`` **binary** command — NOT a
Python callable, and NOT the marin executor. This is the novelty vs exp108:

  * exp108 dispatched a *levanter* callable (``run_levanter_train_lm``) into the
    default marin iris-task image;
  * exp112 dispatches a *bash/torchrun* binary into the ``nvcr.io/nvidia/nemo``
    image, which has no repo checkout and no fray task harness.

Because the container has no checkout, we **base64-inline** the in-container
training script (``nemo_train_qwen3.py``) into the bootstrap bash, which decodes
it and runs ``torchrun --standalone`` over the node's 8 GPUs. Single node ->
``--standalone`` handles rendezvous locally, sidestepping iris's missing
multi-node torchrun coordinator API.

Batch band is set exactly as exp108: ``JobRequest.priority=3`` (iris
``PRIORITY_BAND_BATCH``). We assert at import that this fray build actually has
``JobRequest.priority`` (the frozen ``0.99.dev`` build silently drops it ->
interactive band, which would disrupt the users batch priority protects).

Run (as a tiny CPU **driver** job, like exp108/grug — ``current_client`` then
resolves to the in-cluster controller)::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    WK=$(python -c "import netrc; print(netrc.netrc().authenticators('api.wandb.ai')[2])")
    uv run iris --cluster=cw-rno2a job run --no-wait --priority batch \
        --cpu=2 --memory=6GB --disk=16GB \
        -e WANDB_API_KEY "$WK" -e EXP112_MAX_STEPS 300 \
        -- python -m dispatch_nemo

Smoke first: add ``-e EXP112_MAX_STEPS 50``. Dry-run locally (build + print the
JobRequest, no submit): ``EXP112_DRY_RUN=1 python -m dispatch_nemo``.
"""

from __future__ import annotations

import base64
import dataclasses
import logging
import os
from pathlib import Path

from fray import ResourceConfig
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment

from common import (
    EXP112_S3_PREFIX,
    GLOBAL_BATCH_SIZE,
    MODEL,
    NEMO_IMAGE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    h100_resources,
)

logger = logging.getLogger(__name__)

# iris PriorityBand enum value (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3).
IRIS_PRIORITY_BAND_BATCH = 3

assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line (exp112's pins), not the frozen 0.99.dev build."
)

# Perf/runtime env forwarded from the driver onto the GPU task (iris tasks don't
# inherit the submitter's shell; the task pod is separate). Same idea as exp108.
_FORWARD_ENV_PREFIXES = ("NCCL_", "TORCH_", "NVTE_", "CUDA_", "TRANSFORMERS_", "HF_")
_FORWARD_ENV_EXCLUDE = ("CUDA_VISIBLE_DEVICES",)


def _forwarded_env() -> dict[str, str]:
    return {
        k: v
        for k, v in os.environ.items()
        if k.startswith(_FORWARD_ENV_PREFIXES) and k not in _FORWARD_ENV_EXCLUDE
    }


TRAIN_SCRIPT = Path(__file__).with_name("nemo_train_qwen3.py")


def build_bootstrap(*, nproc_per_node: int, run_name: str, results_s3: str, train_args: list[str]) -> str:
    """Bash bootstrap for the NeMo pod: decode the inlined script, torchrun it,
    then best-effort upload the MFU summary JSON to S3 (so it survives the pod).
    """
    script_b64 = base64.b64encode(TRAIN_SCRIPT.read_bytes()).decode()
    results_local = "/tmp/exp112_mfu.json"
    train_argline = " ".join(train_args)
    # NOTE: keep this POSIX-sh friendly; the NeMo image's /bin/bash is fine.
    return f"""
set -euo pipefail
echo "[exp112-bootstrap] host=$(hostname) python=$(command -v python) torchrun=$(command -v torchrun)"
echo "[exp112-bootstrap] nvidia-smi:"; nvidia-smi -L || true
mkdir -p /tmp/exp112
echo {script_b64} | base64 -d > /tmp/exp112/nemo_train_qwen3.py
echo "[exp112-bootstrap] decoded train script ($(wc -l < /tmp/exp112/nemo_train_qwen3.py) lines); launching torchrun"
export EXP112_RESULTS_JSON={results_local}
export EXP112_RUN_NAME={run_name}
set +e
torchrun --standalone --nnodes=1 --nproc-per-node={nproc_per_node} \
    /tmp/exp112/nemo_train_qwen3.py {train_argline}
rc=$?
set -e
echo "[exp112-bootstrap] torchrun exited rc=$rc"
# Best-effort: push the MFU summary to S3 (creds injected by iris). Logs already
# carry the summary, so failure here is non-fatal.
if [ -f {results_local} ]; then
  python - <<'PYEOF' || echo "[exp112-bootstrap] S3 upload skipped/failed (summary is still in logs)"
import os, boto3
from botocore.config import Config
src = "{results_local}"
uri = "{results_s3}"
assert uri.startswith("s3://")
bucket, key = uri[5:].split("/", 1)
s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                  config=Config(s3={{"addressing_style": "virtual"}}))
s3.upload_file(src, bucket, key)
print("[exp112-bootstrap] uploaded", uri)
PYEOF
fi
exit $rc
""".strip()


def build_train_args(*, max_steps: int, micro_batch_size: int, global_batch_size: int,
                     data: str, run_name: str) -> list[str]:
    return [
        f"--max-steps={max_steps}",
        f"--micro-batch-size={micro_batch_size}",
        f"--global-batch-size={global_batch_size}",
        f"--seq-length={MODEL['seq_length']}",
        f"--num-layers={MODEL['num_layers']}",
        f"--hidden-size={MODEL['hidden_size']}",
        f"--ffn-hidden-size={MODEL['ffn_hidden_size']}",
        f"--num-attention-heads={MODEL['num_attention_heads']}",
        f"--num-query-groups={MODEL['num_query_groups']}",
        f"--kv-channels={MODEL['kv_channels']}",
        f"--data={data}",
        f"--wandb-project={WANDB_PROJECT}",
        f"--wandb-entity={WANDB_ENTITY}",
        f"--wandb-name={run_name}",
    ]


def build_request(
    *,
    run_name: str,
    max_steps: int,
    micro_batch_size: int,
    global_batch_size: int = GLOBAL_BATCH_SIZE,
    data: str = "mock",
    image: str = NEMO_IMAGE,
    env_vars: dict[str, str] | None = None,
) -> JobRequest:
    resources = h100_resources(image=image)
    nproc = resources.device.count  # 8 GPUs on the node

    results_s3 = f"{EXP112_S3_PREFIX}/results/{run_name}.json"
    train_args = build_train_args(
        max_steps=max_steps, micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size, data=data, run_name=run_name,
    )
    bootstrap = build_bootstrap(
        nproc_per_node=nproc, run_name=run_name, results_s3=results_s3, train_args=train_args,
    )

    task_env = {**_forwarded_env(), **(env_vars or {})}
    # docker_image (NOT workspace) so create_environment does NOT bundle/sync the
    # launcher's pyproject INTO the NeMo container (which would break the image);
    # the container IS the environment. Mirror it on resources.image too.
    environment = create_environment(docker_image=image, env_vars=task_env)

    return JobRequest(
        name=run_name,
        entrypoint=Entrypoint.from_binary("bash", ["-lc", bootstrap]),
        resources=resources,
        environment=environment,
        priority=IRIS_PRIORITY_BAND_BATCH,   # -> iris BATCH band
        processes_per_task=1,                # torchrun forks the 8 ranks itself
        max_retries_failure=0,
    )


def main() -> None:
    run_name = os.environ.get("EXP112_RUN_NAME", "plm-exp112-cv1-3b-nemo-bench")
    max_steps = int(os.environ.get("EXP112_MAX_STEPS", "300"))
    micro_batch = int(os.environ.get("EXP112_MICRO_BATCH", "1"))
    global_batch = int(os.environ.get("EXP112_GLOBAL_BATCH", str(GLOBAL_BATCH_SIZE)))
    data = os.environ.get("EXP112_DATA", "mock")

    env_vars = {"WANDB_ENTITY": WANDB_ENTITY, "WANDB_PROJECT": WANDB_PROJECT}
    if os.environ.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    request = build_request(
        run_name=run_name, max_steps=max_steps, micro_batch_size=micro_batch,
        global_batch_size=global_batch, data=data, env_vars=env_vars,
    )

    print(f"[exp112] dispatch: {run_name} | image={NEMO_IMAGE} | 8xH100 batch-band | "
          f"gbs{global_batch} mbs{micro_batch} steps{max_steps} data={data}")

    if os.environ.get("EXP112_DRY_RUN"):
        print("[exp112] DRY RUN — JobRequest built, not submitting.")
        print(f"  priority={request.priority} image={request.resources.image} "
              f"nproc={request.resources.device.count} "
              f"entrypoint=bash -lc <{len(request.entrypoint.binary_entrypoint.args[1])} char bootstrap>")
        return

    job = current_client().submit(request)
    print(f"[exp112] submitted {run_name}; awaiting completion (driver must outlive the gang).")
    job.wait(raise_on_failure=True)
    print(f"[exp112] {run_name}: SUCCEEDED — MFU summary in the job logs + {EXP112_S3_PREFIX}/results/")


if __name__ == "__main__":
    main()
