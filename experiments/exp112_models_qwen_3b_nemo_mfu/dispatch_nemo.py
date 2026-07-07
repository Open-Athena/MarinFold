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
    EXP112_CKPT_S3_BASE,
    EXP112_DATA_S3,
    EXP112_RDZV_S3_BASE,
    EXP112_S3_PREFIX,
    GLOBAL_BATCH_SIZE,
    LEARNING_RATE,
    MODEL,
    NEMO_IMAGE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WARMUP_FRACTION,
    WEIGHT_DECAY,
    h100_resources,
    num_train_steps,
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


DATA_LOCAL = "/tmp/exp112/data"
MASTER_PORT = int(os.environ.get("EXP112_MASTER_PORT", "29500"))


def build_bootstrap(*, nproc_per_node: int, replicas: int, run_name: str, results_s3: str,
                    data_mode: str, train_args: list[str]) -> str:
    """Bash bootstrap for the NeMo pod(s). Single node -> torchrun --standalone.
    Multi-node -> S3 rendezvous (rank-0 publishes IRIS_ADVERTISE_HOST to an
    attempt-scoped key; peers poll) + static torchrun. Real data -> each node
    downloads the bin/idx first. Benchmark (mock) -> upload the MFU summary after.
    """
    script_b64 = base64.b64encode(TRAIN_SCRIPT.read_bytes()).decode()
    results_local = "/tmp/exp112_mfu.json"
    train_argline = " ".join(train_args)
    rdzv_prefix = f"{EXP112_RDZV_S3_BASE}/{run_name}"
    files = "train_document.bin train_document.idx val_document.bin val_document.idx"
    return f"""
set -euo pipefail
TID="${{IRIS_TASK_ID:-/x/0:0}}"
NR="${{TID%%:*}}"; export NODE_RANK="${{NR##*/}}"; export ATTEMPT="${{TID##*:}}"
export NNODES="${{IRIS_NUM_TASKS:-1}}"
export MASTER_PORT="{MASTER_PORT}"
echo "[exp112-bootstrap] host=$(hostname) task=$TID node_rank=$NODE_RANK nnodes=$NNODES ip=${{IRIS_ADVERTISE_HOST:-?}}"
nvidia-smi -L || true
mkdir -p /tmp/exp112 {DATA_LOCAL}
echo {script_b64} | base64 -d > /tmp/exp112/nemo_train_qwen3.py
echo "[exp112-bootstrap] decoded train script ($(wc -l < /tmp/exp112/nemo_train_qwen3.py) lines)"

if [ "$NNODES" -gt 1 ]; then
  echo "[exp112-bootstrap] multi-node rendezvous (attempt $ATTEMPT) via S3"
  python - <<PYEOF
import os, time, boto3
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                  config=Config(s3={{"addressing_style": "virtual"}}))
uri = "{rdzv_prefix}/%s/master" % os.environ["ATTEMPT"]
b, k = uri[5:].split("/", 1)
nr = int(os.environ["NODE_RANK"]); ip = os.environ.get("IRIS_ADVERTISE_HOST", "127.0.0.1")
if nr == 0:
    s3.put_object(Bucket=b, Key=k, Body=ip.encode()); m = ip
    print("[exp112] published MASTER_ADDR", ip)
else:
    m = None
    for _ in range(600):
        try:
            m = s3.get_object(Bucket=b, Key=k)["Body"].read().decode(); break
        except Exception:
            time.sleep(2)
    if not m:
        raise SystemExit("[exp112] rendezvous timeout waiting for rank-0 IP")
    print("[exp112] got MASTER_ADDR", m)
open("/tmp/exp112/master_addr", "w").write(m)
PYEOF
  MASTER_ADDR="$(cat /tmp/exp112/master_addr)"
  LAUNCH="torchrun --nnodes=$NNODES --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR --master-port=$MASTER_PORT --nproc-per-node={nproc_per_node}"
else
  LAUNCH="torchrun --standalone --nnodes=1 --nproc-per-node={nproc_per_node}"
fi

if [ "{data_mode}" = "real" ]; then
  echo "[exp112-bootstrap] downloading bin/idx -> {DATA_LOCAL}"
  python - <<PYEOF
import os, boto3
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                  config=Config(s3={{"addressing_style": "virtual"}}))
base = "{EXP112_DATA_S3}"; b, pre = base[5:].split("/", 1)
for f in "{files}".split():
    dst = "{DATA_LOCAL}/" + f
    if not os.path.exists(dst) or os.path.getsize(dst) == 0:
        s3.download_file(b, pre + "/" + f, dst); print("[exp112] downloaded", f, os.path.getsize(dst))
PYEOF
fi

export EXP112_RESULTS_JSON={results_local}
export EXP112_RUN_NAME={run_name}
echo "[exp112-bootstrap] launching: $LAUNCH"
set +e
$LAUNCH /tmp/exp112/nemo_train_qwen3.py {train_argline}
rc=$?
set -e
echo "[exp112-bootstrap] torchrun exited rc=$rc"
# Benchmark only: push the MFU summary (real runs persist via checkpoints).
if [ "$NODE_RANK" = "0" ] && [ -f {results_local} ]; then
  python - <<PYEOF || echo "[exp112-bootstrap] summary upload skipped (also in logs)"
import os, boto3
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"),
                  config=Config(s3={{"addressing_style": "virtual"}}))
b, k = "{results_s3}"[5:].split("/", 1)
s3.upload_file("{results_local}", b, k); print("[exp112-bootstrap] uploaded {results_s3}")
PYEOF
fi
exit $rc
""".strip()


def build_train_args(*, max_steps: int, micro_batch_size: int, global_batch_size: int,
                     data: str, run_name: str, devices: int, num_nodes: int,
                     checkpoint_s3: str = "", checkpoint_every: int = 500,
                     val_check_interval: int = 1000, limit_val_batches: int = 50) -> list[str]:
    args = [
        f"--max-steps={max_steps}",
        f"--micro-batch-size={micro_batch_size}",
        f"--global-batch-size={global_batch_size}",
        f"--devices={devices}",
        f"--num-nodes={num_nodes}",
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
    if data == "real":
        args += [
            f"--train-data-prefix={DATA_LOCAL}/train_document",
            f"--val-data-prefix={DATA_LOCAL}/val_document",
            f"--checkpoint-s3={checkpoint_s3}",
            f"--checkpoint-every={checkpoint_every}",
            f"--val-check-interval={val_check_interval}",
            f"--limit-val-batches={limit_val_batches}",
            f"--lr={LEARNING_RATE}",
            f"--weight-decay={WEIGHT_DECAY}",
            f"--warmup-fraction={WARMUP_FRACTION}",
            "--resume",  # idempotent: no S3 checkpoint yet => fresh start
        ]
    return args


def build_request(
    *,
    run_name: str,
    max_steps: int,
    micro_batch_size: int,
    global_batch_size: int = GLOBAL_BATCH_SIZE,
    data: str = "mock",
    replicas: int = 1,
    image: str = NEMO_IMAGE,
    checkpoint_every: int = 500,
    val_check_interval: int = 1000,
    env_vars: dict[str, str] | None = None,
) -> JobRequest:
    resources = h100_resources(image=image, replicas=replicas)
    nproc = resources.device.count  # 8 GPUs per node

    results_s3 = f"{EXP112_S3_PREFIX}/results/{run_name}.json"
    checkpoint_s3 = f"{EXP112_CKPT_S3_BASE}/{run_name}"
    train_args = build_train_args(
        max_steps=max_steps, micro_batch_size=micro_batch_size,
        global_batch_size=global_batch_size, data=data, run_name=run_name,
        devices=nproc, num_nodes=replicas, checkpoint_s3=checkpoint_s3,
        checkpoint_every=checkpoint_every, val_check_interval=val_check_interval,
    )
    bootstrap = build_bootstrap(
        nproc_per_node=nproc, replicas=replicas, run_name=run_name, results_s3=results_s3,
        data_mode=data, train_args=train_args,
    )

    task_env = {**_forwarded_env(), **(env_vars or {})}
    # docker_image (NOT workspace) so create_environment does NOT sync the
    # launcher's pyproject INTO the NeMo container; the container IS the environment.
    environment = create_environment(docker_image=image, env_vars=task_env)

    return JobRequest(
        name=run_name,
        entrypoint=Entrypoint.from_binary("bash", ["-lc", bootstrap]),
        resources=resources,
        environment=environment,
        replicas=replicas,                        # gang of `replicas` × 8×H100 nodes
        priority=IRIS_PRIORITY_BAND_BATCH,        # -> iris BATCH band
        processes_per_task=1,                     # torchrun forks the 8 ranks per node
        max_retries_failure=0,                    # a code bug shouldn't retry
        max_retries_preemption=100,               # but auto-restart on preemption (resume from S3)
    )


def main() -> None:
    data = os.environ.get("EXP112_DATA", "mock")
    replicas = int(os.environ.get("EXP112_REPLICAS", "1"))
    run_default = "plm-exp112-cv1-3b-nemo-e16-lr1e-3-wd0p2" if data == "real" else "plm-exp112-cv1-3b-nemo-bench"
    run_name = os.environ.get("EXP112_RUN_NAME", run_default)
    micro_batch = int(os.environ.get("EXP112_MICRO_BATCH", "1"))
    global_batch = int(os.environ.get("EXP112_GLOBAL_BATCH", str(GLOBAL_BATCH_SIZE)))
    # Full 16-epoch step count for the real run (overridable for a smoke).
    default_steps = num_train_steps(global_batch) if data == "real" else 300
    max_steps = int(os.environ.get("EXP112_MAX_STEPS", str(default_steps)))
    checkpoint_every = int(os.environ.get("EXP112_CKPT_EVERY", "500"))
    val_interval = int(os.environ.get("EXP112_VAL_INTERVAL", "1000"))

    env_vars = {"WANDB_ENTITY": WANDB_ENTITY, "WANDB_PROJECT": WANDB_PROJECT}
    if os.environ.get("WANDB_API_KEY"):
        env_vars["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    request = build_request(
        run_name=run_name, max_steps=max_steps, micro_batch_size=micro_batch,
        global_batch_size=global_batch, data=data, replicas=replicas,
        checkpoint_every=checkpoint_every, val_check_interval=val_interval, env_vars=env_vars,
    )

    print(f"[exp112] dispatch: {run_name} | image={NEMO_IMAGE} | {replicas}×8="
          f"{replicas * 8} H100 batch-band | gbs{global_batch} mbs{micro_batch} "
          f"steps{max_steps} data={data} ckpt_every={checkpoint_every}")

    if os.environ.get("EXP112_DRY_RUN"):
        print("[exp112] DRY RUN — JobRequest built, not submitting.")
        print(f"  priority={request.priority} replicas={request.replicas} "
              f"image={request.resources.image} nproc={request.resources.device.count} "
              f"max_retries_preemption={request.max_retries_preemption} "
              f"bootstrap={len(request.entrypoint.binary_entrypoint.args[1])} chars")
        return

    job = current_client().submit(request)
    print(f"[exp112] submitted {run_name} ({replicas} nodes); awaiting completion "
          f"(driver must outlive the gang).")
    job.wait(raise_on_failure=True)
    print(f"[exp112] {run_name}: SUCCEEDED — checkpoints under {EXP112_CKPT_S3_BASE}/{run_name}/")


if __name__ == "__main__":
    main()
