# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Batch-priority CoreWeave fan-out of the retraction-aware rollout eval (#160).

The TPU twin is ``dispatch_eval_tpu.py``. This exists because on 2026-07-28 the
marin TPU fleet had **no free accelerators of any family** — ``v5p-8`` in both
``us-central1-a`` and ``us-east5-a``, ``v5p-16``, and ``v6e-4`` all came back
``Insufficient TPUs (need 4, available 0)`` at interactive band, while
``cw-rno2a`` had **zero pending jobs**. Same measurement, whichever accelerator
has room; an H100 also runs this workload at ~25k tok/s against a v5p-8's
~11.5k.

Adapted from exp82's ``dispatch_rollout_eval_cw.py``, which is where every
gotcha below was paid for. Three changes:

* it runs **this** experiment's worker, so votes come from the #158 fold and
  each rollout's edit list is kept;
* ``marinfold`` is pinned to a **commit on this branch** — the ``<retract>``
  fold is not on ``main``, and a worker installed from ``main`` would parse
  retractions with a contact-only regex and silently score the backtracking arm
  as if it had never taken anything back;
* ``--contact-mult 8`` for both arms (retraction lengthens documents).

Models and targets must be in **CoreWeave S3**: CoreWeave pods cannot read GCS.
``stage_to_cw.py`` moves them cloud-side from a GCS-local marin pod, which takes
about a minute against ~16 minutes per model over the workstation uplink.

    uv run --no-project --with 'marin-fray' python dispatch_eval_cw.py --num-shards 12
    EVAL_CW_DRY_RUN=1 python dispatch_eval_cw.py --num-shards 2   # build, don't submit
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import os
import subprocess
from pathlib import Path

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

# iris PriorityBand enum (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3). fray maps
# JobRequest.priority straight to the iris band. CoreWeave GPU work is always
# batch band; the band does NOT propagate from a CLI --priority to children, so
# it has to be set on each request.
IRIS_PRIORITY_BAND_BATCH = 3

# The frozen 0.99.dev fray has no `priority` field, so priority=3 would be
# silently dropped into the interactive band — exactly the failure this assert
# exists to make loud.
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line. Submit from /home/bizon/git/marin-freshiris."
)

VLLM_IMAGE = os.environ.get("EVAL_CW_IMAGE", "vllm/vllm-openai:v0.9.2")
REPO_URL = "https://github.com/Open-Athena/MarinFold.git"

S3_PREFIX = os.environ.get(
    "EVAL_CW_S3_PREFIX",
    "s3://marin-us-east-02a/protein-structure/MarinFold/exp160_backtracking_training/eval")
TARGETS_S3 = os.environ.get("EVAL_CW_TARGETS", f"{S3_PREFIX}/eval_targets.parquet")
OUT_S3 = os.environ.get("EVAL_CW_OUT", f"{S3_PREFIX}/scores")
JOB_PREFIX = os.environ.get("EVAL_CW_JOB_PREFIX", "exp160-eval-cw")

ARMS = {
    "exp160-bt50": f"{S3_PREFIX}/models/exp160-bt50-step2058",
    "exp120-base": f"{S3_PREFIX}/models/exp120-base",
}

N_ROLLOUTS = int(os.environ.get("EXP160_N_ROLLOUTS", "100"))
TOP_K = int(os.environ.get("EXP160_TOP_K", "-1"))
TOP_P = float(os.environ.get("EXP160_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EXP160_TEMPERATURE", "1.0"))
CONTACT_MULT = int(os.environ.get("EXP160_CONTACT_MULT", "8"))

WORKER_SCRIPT = Path(__file__).with_name("score_backtracking_worker.py")
WORK_DIR = "/tmp/exp160_eval_cw"
WORKER_LOCAL = f"{WORK_DIR}/score_backtracking_worker.py"

# CoreWeave object storage rejects path-style S3. Literal braces on purpose.
FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def resolve_commit(explicit: str | None) -> str:
    """The commit the pod installs ``marinfold`` from, asserted to be on origin."""
    here = Path(__file__).resolve().parent
    sha = explicit or subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=here, check=True,
        capture_output=True, text=True).stdout.strip()
    remote = subprocess.run(
        ["git", "branch", "-r", "--contains", sha], cwd=here,
        capture_output=True, text=True)
    if remote.returncode != 0 or not remote.stdout.strip():
        raise SystemExit(
            f"commit {sha[:12]} is not on any remote branch — push it first, or the pod "
            "will install a different marinfold than the one this eval was written against"
        )
    return sha


def build_bootstrap(*, label: str, model: str, shard_i: int, num_shards: int,
                    limit: int | None, commit: str) -> str:
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    marinfold = f"marinfold @ git+{REPO_URL}@{commit}#subdirectory=marinfold"
    return f"""
set -euo pipefail
echo "[eval-cw] host=$(hostname) label={label} shard={shard_i}/{num_shards} image={VLLM_IMAGE}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}
echo "[eval-cw] AWS_ENDPOINT_URL=${{AWS_ENDPOINT_URL:-unset}} iris_FSSPEC_S3=${{FSSPEC_S3:+present}}"

mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

# vLLM + torch + transformers are baked into the image. marinfold goes in
# --no-deps so nothing repins the image's transformers out from under vLLM; the
# contacts-v1 generator + read fold need only fsspec + numpy on top.
VLLM_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$_py"; break; fi
done
echo "[eval-cw] vLLM python: ${{VLLM_PY:-NONE}}"
if [ -z "$VLLM_PY" ]; then echo "[eval-cw] FATAL: no python imports vllm"; exit 3; fi
uv pip install --python "$VLLM_PY" --quiet fsspec s3fs boto3 pyarrow \
  || "$VLLM_PY" -m pip install --quiet fsspec s3fs boto3 pyarrow
uv pip install --python "$VLLM_PY" --quiet --no-deps "{marinfold}" \
  || "$VLLM_PY" -m pip install --quiet --no-deps "{marinfold}"
"$VLLM_PY" -c "from marinfold.document_structures.contacts_v1.read import fold_statements; \
from marinfold.document_structures.contacts_v1 import build_document; \
print('[eval-cw] marinfold OK (retract fold present)')"

# 1xH100 requests pack several pods per node and those pods SHARE the node's
# network namespace, so vLLM's engine-core processes all pick the same default
# port and every loser dies with EADDRINUSE. Ask the kernel for a free one -- do
# NOT key it on $$ or $(hostname): bash is pid 1 in the container and the
# hostname is the NODE, so both are identical across co-located pods.
export VLLM_PORT=$("$VLLM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "[eval-cw] VLLM_PORT=$VLLM_PORT"

export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec "$VLLM_PY" {WORKER_LOCAL} \\
    --model {model} \\
    --targets {TARGETS_S3} \\
    --out {OUT_S3} \\
    --label {label} \\
    --shard {shard_i}/{num_shards} \\
    --n-rollouts {N_ROLLOUTS} \\
    --temperature {TEMPERATURE} \\
    --top-p {TOP_P} \\
    --top-k {TOP_K} \\
    --contact-mult {CONTACT_MULT}{limit_arg}
""".strip()


def build_request(*, label: str, model: str, shard_i: int, num_shards: int,
                  limit: int | None, commit: str, name_suffix: str = "") -> JobRequest:
    return JobRequest(
        name=f"{JOB_PREFIX}-{label}-s{shard_i}of{num_shards}{name_suffix}",
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(label=label, model=model, shard_i=shard_i,
                                            num_shards=num_shards, limit=limit,
                                            commit=commit)]),
        resources=ResourceConfig.with_gpu(
            "H100", count=1, image=VLLM_IMAGE, cpu=8, ram="64g", disk="128g"),
        # setup_scripts=[] disables iris's default `uv sync`: submitting from the
        # workstation with no workspace bundle, there is no pyproject to sync and
        # the step fails outright. The image already has torch/vLLM.
        environment=create_environment(docker_image=VLLM_IMAGE, env_vars={},
                                       setup_scripts=[]),
        replicas=1,                      # N independent jobs, never a co-scheduled gang
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=3,
        max_retries_preemption=100,      # batch band is preemptible; the worker resumes
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=12)
    ap.add_argument("--labels", default=",".join(ARMS))
    ap.add_argument("--shards", default=None, help="comma-separated subset, e.g. '0'")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    ap.add_argument("--commit", default=None, help="marinfold commit (default: local HEAD)")
    ap.add_argument("--name-suffix", default="",
                    help="appended to job names — iris names are unique, so a retry needs one")
    a = ap.parse_args()

    labels = [x for x in a.labels.split(",") if x]
    unknown = [x for x in labels if x not in ARMS]
    if unknown:
        ap.error(f"unknown label(s) {unknown}; known: {sorted(ARMS)}")
    which = ([int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards)))
    commit = resolve_commit(a.commit)

    reqs = [build_request(label=lbl, model=ARMS[lbl], shard_i=i, num_shards=a.num_shards,
                          limit=a.limit, commit=commit, name_suffix=a.name_suffix)
            for lbl in labels for i in which]

    print(f"[eval-cw] {len(reqs)} job(s) = {len(labels)} arm(s) x {len(which)} shard(s), "
          f"1xH100 batch band | image={VLLM_IMAGE} | marinfold @ {commit[:12]}\n"
          f"          n_rollouts={N_ROLLOUTS} top_k={TOP_K} top_p={TOP_P} T={TEMPERATURE} "
          f"contact_mult={CONTACT_MULT} limit={a.limit}\n"
          f"          targets={TARGETS_S3}\n          out={OUT_S3}")

    if os.environ.get("EVAL_CW_DRY_RUN"):
        print("[eval-cw] DRY RUN — JobRequests built, not submitting.")
        r = reqs[0]
        print(f"  {r.name}: priority={r.priority} image={r.resources.image} "
              f"gpu={r.resources.device.variant}x{r.resources.device.count} "
              f"disk={r.resources.disk}")
        print(r.entrypoint.binary_entrypoint.args[1])
        return

    from iris.client.client import get_iris_ctx

    if get_iris_ctx() is not None:
        # In-cluster driver job: these are its children and iris kills children
        # when a parent exits, so it must outlive them.
        _submit(current_client(), reqs, must_wait=True)
        return

    # Workstation submission. `current_client()` would fall back to LocalClient
    # and try to run every H100 job on this box, so build the iris-backed client
    # explicitly over the CLI's controller tunnel. These become ROOT jobs that
    # survive this process exiting.
    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    cluster = os.environ.get("EVAL_CW_CLUSTER", "cw-rno2a")
    print(f"[eval-cw] submitting from the workstation via the {cluster} controller tunnel")
    with open_iris_client(cluster_name=cluster, workspace=None) as iris_client:
        _submit(FrayIrisClient.from_iris_client(iris_client), reqs, must_wait=False)


def _submit(client, reqs, *, must_wait: bool) -> None:
    jobs = [client.submit(r) for r in reqs]
    print(f"[eval-cw] submitted {len(jobs)} jobs", flush=True)
    for r in reqs:
        print(f"    {r.name}")
    if not must_wait:
        return
    for job in jobs:
        try:
            job.wait()
        except Exception as e:                       # wait() RAISES on a failed job,
            print(f"[eval-cw] {job}: {e}", flush=True)   # abandoning the remaining waits


if __name__ == "__main__":
    main()
