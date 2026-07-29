# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fan the inference plans out over single-H100 CoreWeave jobs, at batch priority.

Submitted **from the workstation**, so the shards are *root* iris jobs: they
survive the launcher exiting, and the "a driver that submits child gangs must
wait on them" rule does not apply. Following the root ``AGENTS.md`` recipe for
single-GPU inference fan-out:

* build the iris-backed fray client explicitly — ``current_client()`` off-cluster
  silently falls back to ``LocalClient`` and would try to run every "H100 job" on
  the workstation;
* ``create_environment(..., setup_scripts=[])``, since with no workspace bundle
  iris's default ``uv sync`` step dies on the missing ``pyproject.toml`` before
  the entrypoint ever runs;
* ``priority=3`` (``PRIORITY_BAND_BATCH``) — CoreWeave GPU work is always batch
  band;
* interleaved sharding, done inside the worker.

The image is a plain PyTorch CUDA image rather than the vLLM one exp82 used:
this workload is HF ``transformers`` with hand-managed KV caches (two
temperatures, prefix reuse), so there is no vLLM to protect from a dependency
repin and the pod can install exactly what it needs.

Run::

    set -a; source ~/.config/marin/cw-rno2a.env; set +a
    uv run python dispatch_cw.py --plan A --model cc1mix5-step50000 --num-shards 4

Dry run (build and print the requests, submit nothing)::

    EXP174_DRY_RUN=1 uv run python dispatch_cw.py --plan A --limit 3
"""

import argparse
import base64
import dataclasses
import os
from pathlib import Path

from fray.types import (
    Entrypoint,
    JobRequest,
    ResourceConfig,
    create_environment,
)

# iris PriorityBand enum (iris/rpc/job.proto: PRIORITY_BAND_BATCH = 3).
IRIS_PRIORITY_BAND_BATCH = 3

assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch needs the "
    "0.2.x.dev line. Submit from /home/bizon/git/marin-freshiris."
)

IMAGE = os.environ.get(
    "EXP174_IMAGE", "pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime"
)
# marinfold ships to the pod as a **wheel on object storage**, not a git URL:
# the PyTorch runtime image has no `git`, and a wheel pins the exact code that
# was tested here rather than whatever a branch points at when the pod starts.
# Rebuild + re-upload with:
#   cd marinfold && uv build --wheel -o /tmp/mf_wheel
#   … then put_file to {S3_PREFIX}/wheels/
MARINFOLD_WHEEL = os.environ.get(
    "EXP174_MARINFOLD_WHEEL", "marinfold-0.1.0-py3-none-any.whl"
)

S3_PREFIX = os.environ.get("EXP174_S3", "s3://marin-us-east-02a/MarinFold/exp174")
GT_TAR = f"{S3_PREFIX}/gt/gt_bundle.tar.gz"

WORK_DIR = "/tmp/exp174"
# Modules the worker needs, inlined into the bootstrap so the pod needs no
# workspace bundle. Order is irrelevant — they are written, not executed.
MODULES = [
    "canonical_pdb.py",
    "document_codec.py",
    "sampler.py",
    "plans.py",
    "run_predictions.py",
    "probe_refinement.py",
]

FSSPEC_VIRTUAL_ADDRESSING = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def _entrypoint(plan, model, shard_i, num_shards, out, limit_arg, extra_args):
    """The command each job actually runs — prediction, or the E3 probe."""
    if plan == "E3":
        return f"""
"$PY" - <<'PYGT'
import fsspec, tarfile, pathlib
pathlib.Path("{WORK_DIR}/gt").mkdir(parents=True, exist_ok=True)
with fsspec.open("{GT_TAR}", "rb") as src:
    open("{WORK_DIR}/gt/bundle.tar.gz", "wb").write(src.read())
with tarfile.open("{WORK_DIR}/gt/bundle.tar.gz") as tar:
    tar.extractall("{WORK_DIR}/gt")
PYGT
"$PY" {WORK_DIR}/probe_refinement.py \\
    --model {WORK_DIR}/model_dir \\
    --gt-dir {WORK_DIR}/gt \\
    --out {WORK_DIR}/pred/probe_refinement.csv {extra_args}
"$PY" - <<'PYPUT'
import fsspec
data = open("{WORK_DIR}/pred/probe_refinement.csv", "rb").read()
with fsspec.open("{out}/probe_refinement.csv", "wb") as dst:
    dst.write(data)
PYPUT"""
    return f"""exec "$PY" {WORK_DIR}/run_predictions.py \\
    --model {S3_PREFIX}/models/{model} \\
    --model-name {model} \\
    --gt-tar {GT_TAR} \\
    --plan {plan} \\
    --out {out} \\
    --out-dir {WORK_DIR}/pred \\
    --shard {shard_i}/{num_shards}{limit_arg} {extra_args}"""


def build_bootstrap(*, plan: str, model: str, shard_i: int, num_shards: int,
                    out: str, limit: int | None, extra_args: str) -> str:
    here = Path(__file__).parent
    writes = "\n".join(
        f"echo {base64.b64encode((here / name).read_bytes()).decode()} "
        f"| base64 -d > {WORK_DIR}/{name}"
        for name in MODULES
    )
    limit_arg = f" --limit {limit}" if limit else ""
    entrypoint = _entrypoint(plan, model, shard_i, num_shards, out, limit_arg, extra_args)
    # E3 needs the model on local disk before it starts (it has no S3 fetch of
    # its own), so mirror it in the bootstrap.
    fetch_model = "" if plan != "E3" else f"""
"$PY" - <<'PYMODEL'
import fsspec, pathlib
fs = fsspec.filesystem("s3", config_kwargs={{"s3": {{"addressing_style": "virtual"}}}})
out = pathlib.Path("{WORK_DIR}/model_dir"); out.mkdir(parents=True, exist_ok=True)
for remote in fs.ls("{S3_PREFIX}/models/{model}", detail=False):
    fs.get_file(remote, str(out / remote.rsplit("/", 1)[-1]))
PYMODEL
"""
    return f"""
set -euo pipefail
echo "[exp174] host=$(hostname) plan={plan} model={model} shard={shard_i}/{num_shards}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL_ADDRESSING}
export EXP174_RUNNER_TAG=iris-cw-rno2a

mkdir -p {WORK_DIR}
{writes}

PY=$(command -v python3 || command -v python)
"$PY" -m pip install --quiet --no-input \
    transformers==4.57.6 "biotite>=1.2" fsspec s3fs boto3 pandas

# Pull the marinfold wheel off object storage with the credentials iris injected
# (fsspec reads the FSSPEC_S3 blob; pip cannot).
"$PY" - <<'PYFETCH'
import fsspec
with fsspec.open("{S3_PREFIX}/wheels/{MARINFOLD_WHEEL}", "rb") as src, \
     open("{WORK_DIR}/{MARINFOLD_WHEEL}", "wb") as dst:
    dst.write(src.read())
PYFETCH
"$PY" -m pip install --quiet --no-input "{WORK_DIR}/{MARINFOLD_WHEEL}"
"$PY" -c "import torch, transformers, biotite; from marinfold.document_structures.contacts_and_crops_v1 import build_document; print('[exp174] deps OK', torch.__version__, transformers.__version__)"

export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
{fetch_model}
# The GT bundle is a tarball on object storage; both entrypoints want it
# unpacked, and run_predictions.py does that itself. E3 takes a directory.
{entrypoint}
""".strip()


def build_request(*, plan: str, model: str, shard_i: int, num_shards: int,
                  out: str, limit: int | None, extra_args: str,
                  name_suffix: str) -> JobRequest:
    return JobRequest(
        name=(
            f"exp174-{plan.lower()}-{model.replace('_', '-').replace('.', '')}"
            f"-s{shard_i}of{num_shards}{name_suffix}"
        ),
        entrypoint=Entrypoint.from_binary(
            "bash",
            [
                "-lc",
                build_bootstrap(
                    plan=plan, model=model, shard_i=shard_i, num_shards=num_shards,
                    out=out, limit=limit, extra_args=extra_args,
                ),
            ],
        ),
        resources=ResourceConfig.with_gpu(
            "H100", count=1, image=IMAGE, cpu=8, ram="64g", disk="64g"
        ),
        environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=2,
        max_retries_preemption=100,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--plan", required=True, choices=["A", "C", "F", "E2", "E3"])
    ap.add_argument("--model", default="cc1mix5-step50000")
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--shards", default=None, help="comma-separated subset to (re)submit")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N per shard")
    ap.add_argument("--tag", default=None, help="output subdirectory (default: plan-model)")
    ap.add_argument("--name-suffix", default="", help="iris names must be unique on retry")
    ap.add_argument("--extra-args", default="", help="passed through to run_predictions.py")
    args = ap.parse_args(argv)

    tag = args.tag or f"{args.plan.lower()}-{args.model}"
    out = f"{S3_PREFIX}/pred/{tag}"
    which = (
        [int(x) for x in args.shards.split(",")]
        if args.shards
        else list(range(args.num_shards))
    )
    requests = [
        build_request(
            plan=args.plan, model=args.model, shard_i=i, num_shards=args.num_shards,
            out=out, limit=args.limit, extra_args=args.extra_args,
            name_suffix=args.name_suffix,
        )
        for i in which
    ]

    print(
        f"[exp174] {len(requests)} job(s) | plan={args.plan} model={args.model} "
        f"1xH100 batch band | image={IMAGE}\n"
        f"         gt={GT_TAR}\n         out={out}"
    )
    if os.environ.get("EXP174_DRY_RUN"):
        print("[exp174] DRY RUN — not submitting.")
        print(requests[0].entrypoint.binary_entrypoint.args[1][:2000])
        return 0

    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    # Explicit iris-backed client: current_client() off-cluster resolves to
    # LocalClient and would try to run every H100 job on the workstation.
    # open_iris_client is a context manager — it owns the controller tunnel, so
    # submission has to happen inside it.
    with open_iris_client(cluster_name="cw-rno2a", workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for request in requests:
            job = client.submit(request)
            print(f"  submitted {request.name} -> {job.job_id}")
    print(f"[exp174] {len(requests)} jobs submitted at batch priority")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
