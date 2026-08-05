# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch exp124 raw rollout logging to CoreWeave H100s."""

import argparse
import base64
import dataclasses
import os
from pathlib import Path

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, JobStatus, ResourceConfig, create_environment


IRIS_PRIORITY_BAND_BATCH = 3
assert "priority" in {field.name for field in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the 0.2.x.dev fray line."
)

VLLM_IMAGE = os.environ.get("EXP124_ROLLOUT_IMAGE", "vllm/vllm-openai:v0.9.2")
MARINFOLD_GIT = os.environ.get(
    "EXP124_ROLLOUT_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@exp124/think-loss-masked#subdirectory=marinfold",
)

S3_PREFIX = os.environ.get("EXP124_ROLLOUT_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp124")
TARGETS_S3 = os.environ.get("EXP124_ROLLOUT_TARGETS", "s3://marin-us-east-02a/MarinFold/exp167_eval/eval_targets.parquet")
OUT_S3 = os.environ.get("EXP124_ROLLOUT_OUT", f"{S3_PREFIX}/raw_rollouts")
JOB_PREFIX = os.environ.get("EXP124_ROLLOUT_JOB_PREFIX", "exp124-rawrollout")
DEFAULT_MODEL = os.environ.get("EXP124_ROLLOUT_MODEL", f"{S3_PREFIX}/model/step-35680")
DEFAULT_LABEL = os.environ.get("EXP124_ROLLOUT_LABEL", "exp124_step35680")

N_ROLLOUTS = int(os.environ.get("EXP124_ROLLOUT_N", "100"))
TOP_K = int(os.environ.get("EXP124_ROLLOUT_TOP_K", "-1"))
TOP_P = float(os.environ.get("EXP124_ROLLOUT_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EXP124_ROLLOUT_TEMPERATURE", "1.0"))

WORKER_SCRIPT = Path(__file__).with_name("log_rollout_worker.py")
WORK_DIR = "/tmp/exp124_raw_rollouts"
WORKER_LOCAL = f"{WORK_DIR}/log_rollout_worker.py"
FSSPEC_VIRTUAL_ADDRESSING_EXPORT = """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""


def build_bootstrap(*, label: str, model: str, shard_i: int, num_shards: int, limit: int | None) -> str:
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit is not None else ""
    return f"""
set -euo pipefail
echo "[exp124-rollout] host=$(hostname) label={label} shard={shard_i}/{num_shards} image={VLLM_IMAGE}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}
echo "[exp124-rollout] AWS_ENDPOINT_URL=${{AWS_ENDPOINT_URL:-unset}} FSSPEC_S3=${{FSSPEC_S3:+present}}"

mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

VLLM_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$_py"; break; fi
done
echo "[exp124-rollout] vLLM python: ${{VLLM_PY:-NONE}}"
if [ -z "$VLLM_PY" ]; then echo "[exp124-rollout] FATAL: no python imports vllm"; exit 3; fi
uv pip install --python "$VLLM_PY" --quiet fsspec s3fs boto3 pyarrow \
  || "$VLLM_PY" -m pip install --quiet fsspec s3fs boto3 pyarrow
uv pip install --python "$VLLM_PY" --quiet --no-deps "{MARINFOLD_GIT}" \
  || "$VLLM_PY" -m pip install --quiet --no-deps "{MARINFOLD_GIT}"
"$VLLM_PY" -c "from marinfold.document_structures.contacts_v1 import build_document; print('[exp124-rollout] marinfold OK')"

export VLLM_PORT=$("$VLLM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "[exp124-rollout] VLLM_PORT=$VLLM_PORT"

export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec "$VLLM_PY" {WORKER_LOCAL} \
  --model {model} \
  --targets {TARGETS_S3} \
  --out {OUT_S3} \
  --label {label} \
  --shard {shard_i}/{num_shards} \
  --n-rollouts {N_ROLLOUTS} \
  --temperature {TEMPERATURE} \
  --top-p {TOP_P} \
  --top-k {TOP_K}{limit_arg}
""".strip()


def build_request(*, label: str, model: str, shard_i: int, num_shards: int, limit: int | None, name_suffix: str) -> JobRequest:
    resources = ResourceConfig.with_gpu("H100", count=1, image=VLLM_IMAGE, cpu=8, ram="64g", disk="128g")
    name = f"{JOB_PREFIX}-{label.replace('_', '-')}-s{shard_i}of{num_shards}{name_suffix}"
    return JobRequest(
        name=name,
        entrypoint=Entrypoint.from_binary(
            "bash",
            ["-lc", build_bootstrap(label=label, model=model, shard_i=shard_i, num_shards=num_shards, limit=limit)],
        ),
        resources=resources,
        environment=create_environment(docker_image=VLLM_IMAGE, env_vars={}, setup_scripts=[]),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=3,
        max_retries_preemption=100,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("EXP124_ROLLOUT_SHARDS", "12")))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    parser.add_argument("--shards", default=None, help="comma-separated subset; default all")
    parser.add_argument("--name-suffix", default="")
    args = parser.parse_args()

    which = [int(x) for x in args.shards.split(",")] if args.shards else list(range(args.num_shards))
    requests = [
        build_request(
            label=args.label,
            model=args.model,
            shard_i=shard,
            num_shards=args.num_shards,
            limit=args.limit,
            name_suffix=args.name_suffix,
        )
        for shard in which
    ]
    print(
        f"[exp124-rollout] {len(requests)} job(s), 1xH100 batch | image={VLLM_IMAGE}\n"
        f"  model={args.model}\n  targets={TARGETS_S3}\n  out={OUT_S3}/{args.label}\n"
        f"  n_rollouts={N_ROLLOUTS} top_k={TOP_K} top_p={TOP_P} T={TEMPERATURE} limit={args.limit}",
        flush=True,
    )

    if os.environ.get("EXP124_ROLLOUT_DRY_RUN"):
        print("[exp124-rollout] DRY RUN")
        for request in requests[:2]:
            bootstrap = request.entrypoint.binary_entrypoint.args[1]
            print(f"  {request.name}: priority={request.priority} bootstrap={len(bootstrap)} chars")
        print(requests[0].entrypoint.binary_entrypoint.args[1])
        return

    from iris.client.client import get_iris_ctx

    if get_iris_ctx() is not None:
        submit_and_wait(current_client(), requests, must_wait=True)
        return

    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    cluster = os.environ.get("EXP124_ROLLOUT_CLUSTER", "cw-rno2a")
    print(f"[exp124-rollout] workstation submit via {cluster}", flush=True)
    with open_iris_client(cluster_name=cluster, workspace=None) as iris_client:
        submit_and_wait(FrayIrisClient.from_iris_client(iris_client), requests, must_wait=False)


def submit_and_wait(client, requests: list[JobRequest], *, must_wait: bool) -> None:
    jobs = [client.submit(request) for request in requests]
    print(f"[exp124-rollout] submitted {len(jobs)} jobs", flush=True)
    for request in requests:
        print(f"    {request.name}", flush=True)
    if not must_wait and os.environ.get("EXP124_ROLLOUT_NO_WAIT"):
        print("[exp124-rollout] EXP124_ROLLOUT_NO_WAIT set; jobs are root jobs and keep running", flush=True)
        return
    results = []
    for job in jobs:
        try:
            results.append(job.wait())
        except Exception as exc:
            results.append(f"{type(exc).__name__}: {exc}")
    bad = [(request.name, status) for request, status in zip(requests, results) if status != JobStatus.SUCCEEDED]
    print(f"[exp124-rollout] finished: {len(results) - len(bad)}/{len(results)} succeeded", flush=True)
    for name, status in bad:
        print(f"  FAILED {name}: {status}", flush=True)


if __name__ == "__main__":
    main()
