# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Batch-priority Fray dispatch of the exp301 fold-switch scoring to CoreWeave rno-2a.

Fans ``score_foldswitch_worker_cw.py`` out over ``--num-shards`` single-H100
iris jobs at batch priority. Adapted from exp82's ``dispatch_rollout_eval_cw.py``
— same foreign-container recipe (vLLM CUDA image via ``ResourceConfig.image``,
bash bootstrap that base64-inlines the worker, ``JobRequest.priority=3``).

One difference worth knowing: exp82 staged its checkpoint into CoreWeave S3
first, to dodge the workstation's ~2.5 MB/s uplink. exp301 does not need to —
the published exp277 export is readable straight off the **public HF bucket**
over fsspec with no credentials, and the pods have egress (they already pip
install marinfold from GitHub). So ``--model`` is an ``hf://buckets/...`` URL and
there is no staging step at all. The bootstrap installs ``huggingface_hub>=1.5``
for it, which is safe here precisely because marinfold goes in ``--no-deps`` and
never gets to pin ``transformers``, which is what forces ``hub<1.0``.

Run as a tiny CPU driver job (so ``current_client`` resolves in-cluster)::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    /home/bizon/git/marin/.venv/bin/iris --cluster=cw-rno2a job run \\
        --no-wait --priority batch --enable-extra-resources \\
        --cpu=2 --memory=6GB --disk=16GB -- python -m dispatch_foldswitch_cw

or straight from the workstation (shards become root jobs and survive this
process exiting)::

    uv run --with marin-iris python dispatch_foldswitch_cw.py --num-shards 8

Dry-run (build + print the JobRequests, no submit)::

    EXP301_DRY_RUN=1 python dispatch_foldswitch_cw.py --num-shards 2
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import os
from pathlib import Path

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

# iris PriorityBand enum (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3). fray maps
# JobRequest.priority straight to the iris band.
IRIS_PRIORITY_BAND_BATCH = 3

# Same guard as exp82/exp108/exp163: the frozen 0.99.dev fray has no `priority`
# field, so priority=3 would be silently dropped into the interactive band.
assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line. Submit from an editable marin checkout."
)

VLLM_IMAGE = os.environ.get("EXP301_IMAGE", "vllm/vllm-openai:v0.9.2")
MARINFOLD_GIT = os.environ.get(
    "EXP301_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold",
)

# The MODELS.yaml default, read anonymously off the public bucket.
DEFAULT_MODEL = os.environ.get(
    "EXP301_MODEL",
    "hf://buckets/open-athena/MarinFold/checkpoints/"
    "contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344",
)
DEFAULT_LABEL = os.environ.get("EXP301_LABEL", "exp277")

S3_PREFIX = os.environ.get("EXP301_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp301")
TARGETS_S3 = os.environ.get("EXP301_TARGETS", f"{S3_PREFIX}/eval_targets.parquet")
OUT_S3 = os.environ.get("EXP301_OUT", f"{S3_PREFIX}/scores")
JOB_PREFIX = os.environ.get("EXP301_JOB_PREFIX", "exp301-foldswitch")

N_ROLLOUTS = int(os.environ.get("EXP301_N_ROLLOUTS", "100"))
N_SEEDS = int(os.environ.get("EXP301_N_SEEDS", "5"))
TOP_K = int(os.environ.get("EXP301_TOP_K", "-1"))
TOP_P = float(os.environ.get("EXP301_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EXP301_TEMPERATURE", "1.0"))

WORKER_SCRIPT = Path(__file__).with_name("score_foldswitch_worker_cw.py")
WORK_DIR = "/tmp/exp301"
WORKER_LOCAL = f"{WORK_DIR}/score_foldswitch_worker_cw.py"

# CoreWeave object storage rejects path-style S3. Literal braces on purpose.
FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def build_bootstrap(*, label: str, model: str, shard_i: int, num_shards: int,
                    limit: int | None, skip_nll: bool) -> str:
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    extra = f" --limit {limit}" if limit else ""
    extra += " --skip-nll" if skip_nll else ""
    return f"""
set -euo pipefail
echo "[exp301] host=$(hostname) label={label} shard={shard_i}/{num_shards} image={VLLM_IMAGE}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}
echo "[exp301] AWS_ENDPOINT_URL=${{AWS_ENDPOINT_URL:-unset}} iris_FSSPEC_S3=${{FSSPEC_S3:+present}}"

mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

VLLM_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$_py"; break; fi
done
echo "[exp301] vLLM python: ${{VLLM_PY:-NONE}}"
if [ -z "$VLLM_PY" ]; then echo "[exp301] FATAL: no python imports vllm"; exit 3; fi

# huggingface_hub>=1.5 is what reads hf:// BUCKET paths. Installing it before
# marinfold is deliberate: marinfold goes in with --no-deps, so it never gets to
# pin transformers (which would in turn force hub<1.0 and break the bucket read).
uv pip install --python "$VLLM_PY" --quiet fsspec s3fs boto3 pyarrow "huggingface_hub>=1.5" \
  || "$VLLM_PY" -m pip install --quiet fsspec s3fs boto3 pyarrow "huggingface_hub>=1.5"
uv pip install --python "$VLLM_PY" --quiet --no-deps "{MARINFOLD_GIT}" \
  || "$VLLM_PY" -m pip install --quiet --no-deps "{MARINFOLD_GIT}"
"$VLLM_PY" -c "from marinfold.document_structures.contacts_v1 import build_document; print('[exp301] marinfold OK')"
"$VLLM_PY" -c "import fsspec; fsspec.core.url_to_fs('hf://buckets/open-athena/MarinFold'); print('[exp301] hf bucket fs OK')"

# 1xH100 requests pack several pods per node and those pods SHARE the node's network
# namespace, so vLLM's engine-core all pick the same default torch.distributed port
# and every loser dies with EADDRINUSE. Ask the kernel for a free port -- do NOT key
# it on $$ or $(hostname): bash is pid 1 in the container and the hostname is the
# NODE, so both are identical across co-located pods and make the collision certain.
export VLLM_PORT=$("$VLLM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "[exp301] VLLM_PORT=$VLLM_PORT"

export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec "$VLLM_PY" {WORKER_LOCAL} \\
    --model {model} \\
    --targets {TARGETS_S3} \\
    --out {OUT_S3} \\
    --label {label} \\
    --shard {shard_i}/{num_shards} \\
    --n-rollouts {N_ROLLOUTS} \\
    --n-seeds {N_SEEDS} \\
    --temperature {TEMPERATURE} \\
    --top-p {TOP_P} \\
    --top-k {TOP_K}{extra}
""".strip()


def build_request(*, label: str, model: str, shard_i: int, num_shards: int,
                  limit: int | None, skip_nll: bool, name_suffix: str = "") -> JobRequest:
    return JobRequest(
        name=f"{JOB_PREFIX}-{label.replace('_', '-')}-s{shard_i}of{num_shards}{name_suffix}",
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(label=label, model=model, shard_i=shard_i,
                                            num_shards=num_shards, limit=limit,
                                            skip_nll=skip_nll)]),
        resources=ResourceConfig.with_gpu(
            "H100", count=1, image=VLLM_IMAGE, cpu=8, ram="64g", disk="128g"),
        # setup_scripts=[] disables iris's default `uv sync` step: there is no
        # workspace bundle to sync, and the bootstrap installs what it needs.
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
    ap.add_argument("--num-shards", type=int, default=int(os.environ.get("EXP301_NUM_SHARDS", "8")))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--label", default=DEFAULT_LABEL)
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N pairs per shard")
    ap.add_argument("--skip-nll", action="store_true", help="rollouts only (M1, no M2)")
    ap.add_argument("--shards", default=None,
                    help="comma-separated subset to (re)submit, e.g. '2'; default all")
    ap.add_argument("--name-suffix", default="",
                    help="appended to job names — iris names are unique, so a retry needs one")
    a = ap.parse_args()

    which = ([int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards)))
    reqs = [build_request(label=a.label, model=a.model, shard_i=i, num_shards=a.num_shards,
                          limit=a.limit, skip_nll=a.skip_nll, name_suffix=a.name_suffix)
            for i in which]

    print(f"[exp301] {len(reqs)} job(s) over {a.num_shards} shard(s), 1xH100 batch band | "
          f"image={VLLM_IMAGE}\n"
          f"         rollouts={N_ROLLOUTS} seeds={N_SEEDS} T={TEMPERATURE} top_p={TOP_P} "
          f"top_k={TOP_K} limit={a.limit} skip_nll={a.skip_nll}\n"
          f"         model={a.model}\n"
          f"         targets={TARGETS_S3}\n         out={OUT_S3}/{a.label}")

    if os.environ.get("EXP301_DRY_RUN"):
        print("[exp301] DRY RUN — JobRequests built, not submitting.")
        r = reqs[0]
        print(f"  {r.name}: priority={r.priority} image={r.resources.image} "
              f"gpu={r.resources.device.variant}x{r.resources.device.count} "
              f"cpu={r.resources.cpu} ram={r.resources.ram} disk={r.resources.disk}")
        print(r.entrypoint.binary_entrypoint.args[1])
        return

    from iris.client.client import get_iris_ctx

    if get_iris_ctx() is not None:
        # In-cluster driver job. These jobs are the driver's children, so it MUST
        # outlive them — iris finalizes children when a parent exits.
        _submit_and_wait(current_client(), reqs, must_wait=True)
        return

    # Workstation submission. `current_client()` would fall back to LocalClient and
    # try to run every H100 job on this box, so build the iris-backed client
    # explicitly over the CLI's controller tunnel.
    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    cluster = os.environ.get("EXP301_CLUSTER", "cw-rno2a")
    print(f"[exp301] submitting from the workstation via the {cluster} controller tunnel")
    with open_iris_client(cluster_name=cluster, workspace=None) as iris_client:
        _submit_and_wait(FrayIrisClient.from_iris_client(iris_client), reqs, must_wait=False)


def _submit_and_wait(client, reqs, *, must_wait: bool) -> None:
    jobs = [client.submit(r) for r in reqs]
    print(f"[exp301] submitted {len(jobs)} jobs", flush=True)
    for req, job in zip(reqs, jobs):
        print(f"           {req.name} -> {getattr(job, 'id', job)}", flush=True)
    if not must_wait:
        print("[exp301] root jobs — monitor with: "
              "iris --cluster=cw-rno2a job list | grep exp301", flush=True)
        return
    failures = []
    for req, job in zip(reqs, jobs):
        # job.wait() raises on a failed job, which would abandon every remaining
        # wait and report only the first failure. Catch per job.
        try:
            job.wait()
        except Exception as exc:  # noqa: BLE001
            failures.append((req.name, exc))
            print(f"[exp301] FAILED {req.name}: {type(exc).__name__}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)}/{len(jobs)} shards failed")
    print("[exp301] all shards finished", flush=True)


if __name__ == "__main__":
    main()
