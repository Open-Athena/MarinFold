# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch the FoldBench complex contact evaluation to CoreWeave rno-2a.

Six independent H100 root jobs run the exp82 100-rollout recipe with a
chain-aware prompt and parser. The checkpoint already lives in CoreWeave object
storage, and the small frozen target table is embedded in each root job request.

Run from the workstation::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    /home/bizon/git/marin-freshiris/.venv/bin/python \\
        dispatch_foldbench_contacts_cw.py

Dry-run locally (build + print the JobRequests, no submit)::

    EVAL_CW_DRY_RUN=1 /home/bizon/git/marin-freshiris/.venv/bin/python \\
        dispatch_foldbench_contacts_cw.py --num-shards 2
"""
import argparse
import base64
import dataclasses
import json
import os
from pathlib import Path

from fray.current_client import current_client
from fray.types import (
    Entrypoint,
    JobRequest,
    JobStatus,
    ResourceConfig,
    create_environment,
)

# iris PriorityBand enum (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3). fray maps
# JobRequest.priority straight to the iris band.
IRIS_PRIORITY_BAND_BATCH = 3

# Same guard as exp108/exp112/exp163: the frozen 0.99.dev fray has no `priority`
# field, so priority=3 would be silently dropped into the interactive band.
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line. Submit from /home/bizon/git/marin-freshiris."
)

VLLM_IMAGE = os.environ.get("EVAL_CW_IMAGE", "vllm/vllm-openai:v0.9.2")
MARINFOLD_GIT = os.environ.get(
    "EVAL_CW_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@"
    "339c115df33a639dd9d0c4457c39a2e1c1b4dff8#subdirectory=marinfold",
)

S3_PREFIX = os.environ.get(
    "EVAL_CW_S3_PREFIX",
    "s3://marin-us-east-02a/MarinFold/"
    "exp350_evals_complex_holdout_survival/foldbench-pair-holdout-v1",
)
OUT_S3 = os.environ.get("EVAL_CW_OUT", f"{S3_PREFIX}/rollout")
# Job names are how a run is found in `iris job list` on a shared cluster, so a
# re-use of this dispatcher from another experiment should name its jobs after
# itself (exp169 does).
JOB_PREFIX = os.environ.get("EVAL_CW_JOB_PREFIX", "exp350-foldbench-complex")

# The two checkpoints. exp75 is already on S3 from exp163; exp117 was uploaded by
# this experiment (bf16, to halve the ~2.5 MB/s workstation uplink).
HERE = Path(__file__).resolve().parent
CHECKPOINT = json.loads(
    (
        HERE.parent
        / "exp343_models_complex_corpus_training/evals/"
        "2026-09-28_rollout_v2/exp343_checkpoint.json"
    ).read_text()
)
MODEL_LABEL = CHECKPOINT["label"]
MODEL_S3 = CHECKPOINT["coreweave_uri"]
MODEL_MANIFEST_B64 = base64.b64encode(
    json.dumps(
        {
            "files": [
                {"name": row["name"], "size": row["size"]}
                for row in CHECKPOINT["checkpoint_files"]
            ]
        }
    ).encode()
).decode()

N_ROLLOUTS = int(os.environ.get("EVAL_CW_N_ROLLOUTS", "100"))
TOP_K = int(os.environ.get("EVAL_CW_TOP_K", "-1"))
TOP_P = float(os.environ.get("EVAL_CW_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EVAL_CW_TEMPERATURE", "1.0"))

WORKER_SCRIPT = Path(__file__).with_name("score_complex_rollout_worker.py")
TARGETS_FILE = Path(__file__).with_name("data") / "foldbench_complex_eval_targets.parquet"
WORK_DIR = "/tmp/exp350_complex_eval"
WORKER_LOCAL = f"{WORK_DIR}/score_complex_rollout_worker.py"
TARGETS_LOCAL = f"{WORK_DIR}/eval_targets.parquet"

# CoreWeave object storage rejects path-style S3. Literal braces on purpose.
FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def build_bootstrap(*, shard_i: int, num_shards: int, limit: int | None) -> str:
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    targets_b64 = base64.b64encode(TARGETS_FILE.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    return f"""
set -euo pipefail
echo "[eval-cw] host=$(hostname) label={MODEL_LABEL} shard={shard_i}/{num_shards} image={VLLM_IMAGE}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}
echo "[eval-cw] AWS_ENDPOINT_URL=${{AWS_ENDPOINT_URL:-unset}} iris_FSSPEC_S3=${{FSSPEC_S3:+present}}"

mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORKER_LOCAL}
echo {targets_b64} | base64 -d > {TARGETS_LOCAL}

# vLLM + torch + transformers are baked into the image. Install marinfold WITHOUT
# its dependency set so nothing repins the image's transformers out from under
# vLLM -- the contacts_v1 document generator needs only fsspec + numpy on top.
VLLM_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$_py"; break; fi
done
echo "[eval-cw] vLLM python: ${{VLLM_PY:-NONE}}"
if [ -z "$VLLM_PY" ]; then echo "[eval-cw] FATAL: no python imports vllm"; exit 3; fi
uv pip install --python "$VLLM_PY" --quiet fsspec s3fs boto3 pyarrow \
  || "$VLLM_PY" -m pip install --quiet fsspec s3fs boto3 pyarrow
uv pip install --python "$VLLM_PY" --quiet --no-deps "{MARINFOLD_GIT}" \
  || "$VLLM_PY" -m pip install --quiet --no-deps "{MARINFOLD_GIT}"
"$VLLM_PY" -c "from marinfold.document_structures.contacts_v1 import build_document; print('[eval-cw] marinfold OK')"

# 1xH100 requests pack several pods per node and those pods SHARE the node's network
# namespace, so vLLM's engine-core all pick the same default torch.distributed port
# and every loser dies with EADDRINUSE. Ask the kernel for a free port -- do NOT key
# it on $$ or $(hostname): bash is pid 1 in the container and the hostname is the
# NODE, so both are identical across co-located pods and make the collision certain.
export VLLM_PORT=$("$VLLM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "[eval-cw] VLLM_PORT=$VLLM_PORT"

export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec "$VLLM_PY" {WORKER_LOCAL} \\
    --model {MODEL_S3} \\
    --model-manifest-b64 {MODEL_MANIFEST_B64} \\
    --targets {TARGETS_LOCAL} \\
    --out {OUT_S3} \\
    --label {MODEL_LABEL} \\
    --shard {shard_i}/{num_shards} \\
    --n-rollouts {N_ROLLOUTS} \\
    --temperature {TEMPERATURE} \\
    --top-p {TOP_P} \\
    --top-k {TOP_K}{limit_arg}
""".strip()


def build_request(
    *, shard_i: int, num_shards: int, limit: int | None, name_suffix: str = ""
) -> JobRequest:
    resources = ResourceConfig.with_gpu(
        "H100", count=1, image=VLLM_IMAGE, cpu=8, ram="64g", disk="128g",
    )
    return JobRequest(
        name=f"{JOB_PREFIX}-s{shard_i}of{num_shards}{name_suffix}",
        entrypoint=Entrypoint.from_binary(
            "bash",
            [
                "-lc",
                build_bootstrap(
                    shard_i=shard_i, num_shards=num_shards, limit=limit
                ),
            ],
        ),
        resources=resources,
        # setup_scripts=[] disables iris's default `uv sync` setup step. We submit
        # from the workstation with no workspace bundle, so there is no pyproject to
        # sync (the step fails outright) — and we don't want one: the vLLM image
        # already has torch/vLLM, and the bootstrap installs the few extra wheels.
        environment=create_environment(docker_image=VLLM_IMAGE, env_vars={},
                                       setup_scripts=[]),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=3,
        max_retries_preemption=100,      # batch band is preemptible; the worker resumes
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int,
                    default=int(os.environ.get("EVAL_CW_NUM_SHARDS", "6")))
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    ap.add_argument("--shards", default=None,
                    help="comma-separated subset to (re)submit, e.g. '2'; default all")
    ap.add_argument("--name-suffix", default="",
                    help="appended to job names — iris names are unique, so a retry needs one")
    a = ap.parse_args()

    which = ([int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards)))
    reqs = [
        build_request(
            shard_i=index,
            num_shards=a.num_shards,
            limit=a.limit,
            name_suffix=a.name_suffix,
        )
        for index in which
    ]

    print(f"[eval-cw] {len(reqs)} job(s), {a.num_shards} shard(s), "
          f"1xH100 batch band | image={VLLM_IMAGE}\n"
          f"          n_rollouts={N_ROLLOUTS} top_k={TOP_K} top_p={TOP_P} T={TEMPERATURE} "
          f"limit={a.limit}\n"
          f"          targets={TARGETS_FILE} ({TARGETS_FILE.stat().st_size} bytes, embedded)"
          f"\n          out={OUT_S3}")
    print(f"          model {MODEL_LABEL}: {MODEL_S3}")

    if os.environ.get("EVAL_CW_DRY_RUN"):
        print("[eval-cw] DRY RUN — JobRequests built, not submitting.")
        for r in reqs[:2]:
            bs = r.entrypoint.binary_entrypoint.args[1]
            print(f"  {r.name}: priority={r.priority} image={r.resources.image} "
                  f"gpu={r.resources.device.variant}x{r.resources.device.count} "
                  f"cpu={r.resources.cpu} ram={r.resources.ram} disk={r.resources.disk} "
                  f"bootstrap={len(bs)} chars")
        print(reqs[0].entrypoint.binary_entrypoint.args[1])
        return

    from iris.client.client import get_iris_ctx

    if get_iris_ctx() is not None:
        # In-cluster driver job (exp163's shape). These jobs are the driver's
        # children, so it MUST outlive them — iris kills children when a parent exits.
        _submit_and_wait(current_client(), reqs, must_wait=True)
        return

    # Workstation submission. `current_client()` would fall back to LocalClient and
    # try to run 24 H100 jobs on this box, so build the iris-backed client
    # explicitly over the CLI's controller tunnel. These become ROOT jobs with no
    # iris parent, so they survive this process exiting — which also means the
    # exp82 launcher dir needs no marin/fray pyproject and no pod-side `uv sync`.
    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    cluster = os.environ.get("EVAL_CW_CLUSTER", "cw-rno2a")
    config = os.environ.get("EVAL_CW_CONFIG")
    location = config or cluster
    print(f"[eval-cw] submitting from the workstation via {location}")
    with open_iris_client(
        config_file=Path(config) if config else None,
        cluster_name=None if config else cluster,
        workspace=None,
    ) as iris_client:
        _submit_and_wait(FrayIrisClient.from_iris_client(iris_client), reqs, must_wait=False)


def _submit_and_wait(client, reqs, *, must_wait: bool) -> None:
    jobs = [client.submit(r) for r in reqs]
    print(f"[eval-cw] submitted {len(jobs)} jobs", flush=True)
    for r in reqs:
        print(f"    {r.name}")
    if not must_wait and os.environ.get("EVAL_CW_NO_WAIT"):
        print("[eval-cw] EVAL_CW_NO_WAIT set — not waiting (jobs are root jobs and keep running)")
        return
    # j.wait() RAISES on a failed job, which would abandon the remaining waits and
    # hide every other shard's outcome — catch per job so the summary is complete.
    results = []
    for j in jobs:
        try:
            results.append(j.wait())
        except Exception as e:                       # noqa: BLE001 — report, don't abort
            results.append(f"{type(e).__name__}: {e}")
    bad = [(r.name, s) for r, s in zip(reqs, results) if s != JobStatus.SUCCEEDED]
    print(f"[eval-cw] finished: {len(results) - len(bad)}/{len(results)} succeeded")
    for name, s in bad:
        print(f"  FAILED {name}: {s}")


if __name__ == "__main__":
    main()
