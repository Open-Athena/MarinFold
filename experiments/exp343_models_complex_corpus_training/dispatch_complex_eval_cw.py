# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Score checkpoints on the held-out complex shard, one 1xH100 job each.

The training run logs its own complex validation loss, so exp343's number is
free. What is not free is the **control**: exp277's loss on the same documents.
Without it "the complex loss is 1.4" means nothing, because nobody knows what a
complex-blind model scores.

That also makes this its own correctness gate. exp343's loss here must reproduce
the run's W&B complex validation loss; if it does, the same pipeline's exp277
number is trustworthy, and if it does not, the discrepancy is the finding and no
comparison gets reported.

The worker needs torch + transformers, which the vLLM image already has, and no
marinfold at all -- the section labeller travels base64'd inside the request. So
there is nothing to install beyond storage libraries, and no reason for the
image's transformers to be repinned.

Jobs are submitted from the workstation as **root** jobs, so they survive this
process exiting and the "a driver must outlive its children" rule does not apply.

    uv run --no-project --with 'marin-fray' python dispatch_complex_eval_cw.py
    EVAL_CW_DRY_RUN=1 python dispatch_complex_eval_cw.py --limit 32
"""

import argparse
import base64
import dataclasses
import os
from pathlib import Path

from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

#: iris PriorityBand PRIORITY_BAND_BATCH. CoreWeave GPU work is always batch
#: band, and the band does not propagate from a CLI `--priority` to children, so
#: it is set on every request.
IRIS_PRIORITY_BAND_BATCH = 3

assert "priority" in {field.name for field in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev line. Submit from /home/bizon/git/marin-freshiris."
)

GPU_IMAGE = os.environ.get("EVAL_CW_IMAGE", "vllm/vllm-openai:v0.9.2")
PREFIX = "s3://marin-us-east-02a/MarinFold/exp343_models_complex_corpus_training"
HELD_OUT_SHARD = os.environ.get(
    "EVAL_CW_SHARD", f"{PREFIX}/documents/validation/shard-00170-of-00171.parquet"
)
OUT_S3 = os.environ.get("EVAL_CW_OUT", f"{PREFIX}/evals/complex-loss")
JOB_PREFIX = os.environ.get("EVAL_CW_JOB_PREFIX", "exp343-complex-loss")
WORK_DIR = "/tmp/exp343_complex_eval"

#: label -> HF export directory in CoreWeave S3.
#:
#: `exp277-step266344` is the control: the current default model, which has never
#: seen a multi-chain document.
#:
#: Only real, reported checkpoints live here. The export-format gate that ran
#: against a training smoke is **not** a named arm: `launch.py` tags every smoke
#: run (`-smoke-n8-rno2a`, ...), so a hardcoded smoke path silently rots into
#: either a missing object or -- worse -- a stale checkpoint from an earlier
#: smoke, which would pass the gate while testing nothing. Score an arbitrary
#: export with `--model-uri` instead; see `--help`.
#: Keys are the AGENTS.md `<wandb-run-name>-step-<N>` identifier, because the
#: label becomes the output prefix, the `label` column of every aggregate row and
#: `model_nickname` in the committed timing CSV. An abbreviation there detaches
#: those artifacts from the run that produced them.
ARMS = {
    "contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344": (
        "s3://marin-us-east-02a/MarinFold/exp277_models_single_mpnn_pilot/"
        "runs/contacts-v1-exp277-m2-p06-full-epoch-1.5B/hf/step-266344"
    ),
    "contacts-v1-exp343-m2-p06-complex-1.5B-step-280154": (
        f"{PREFIX}/runs/contacts-v1-exp343-m2-p06-complex-1.5B/hf/step-280154"
    ),
}

#: CoreWeave object storage rejects path-style S3. Literal braces on purpose.
FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)

#: Pinned to the versions that **this** pipeline resolved and ran successfully
#: with (`/bizon/exp343-complex-eval-a04`), read out of that job's install log --
#: not copied from another experiment. exp277's rollout pins
#: (`fsspec==2026.1.0`, `s3fs==2026.1.0`) are a year older here and parse
#: `FSSPEC_S3_CONFIG_KWARGS` as a raw string, which crashes every worker with
#: `AttributeError: 'str' object has no attribute 'copy'`. A pin is only
#: reproducible if it reproduces a run that actually happened.
STORAGE_PINS = (
    "'fsspec==2026.2.0' 's3fs==2026.2.0' 'aiobotocore==3.9.0' "
    "'botocore==1.43.56' 'pyarrow==25.0.1'"
)

HERE = Path(__file__).resolve().parent
WORKER_SCRIPT = HERE / "score_complex_worker.py"
SECTIONS_SCRIPT = HERE / "complex_sections.py"


def output_label(label: str, limit: int | None) -> str:
    """The label a run writes under.

    A limited run is a smoke and must not share the production `parts/` prefix:
    the resume path reads back any part already present and checks its row count,
    so a 32-row smoke part would make the next full run abort deterministically.
    """
    return f"{label}-smoke{limit}" if limit else label


def build_bootstrap(*, label: str, model: str, limit: int | None) -> str:
    """The pod-side script: pick the image's python, add storage, run the worker."""
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    sections_b64 = base64.b64encode(SECTIONS_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    return f"""
set -euo pipefail
echo "[complex-loss] host=$(hostname) label={label} image={GPU_IMAGE}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}
echo "[complex-loss] iris_FSSPEC_S3=${{FSSPEC_S3:+present}}"

mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORK_DIR}/score_complex_worker.py

# torch and transformers are baked into the image and must stay exactly as the
# vendor shipped them. These storage pins are the set exp277's rollout eval
# validated against this same image; leaving them unpinned lets a newly released
# transitive dependency replace part of a vendor-tested environment between two
# otherwise identical runs.
GPU_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import torch, transformers" >/dev/null 2>&1; then GPU_PY="$_py"; break; fi
done
echo "[complex-loss] python: ${{GPU_PY:-NONE}}"
if [ -z "$GPU_PY" ]; then echo "[complex-loss] FATAL: no python imports torch+transformers"; exit 3; fi
uv pip install --python "$GPU_PY" --quiet {STORAGE_PINS} \
  || "$GPU_PY" -m pip install --quiet {STORAGE_PINS}
"$GPU_PY" -c "import torch, transformers; print('[complex-loss] torch', torch.__version__, 'transformers', transformers.__version__)"

exec "$GPU_PY" {WORK_DIR}/score_complex_worker.py \\
    --model {model} \\
    --label {output_label(label, limit)} \\
    --shard {HELD_OUT_SHARD} \\
    --out {OUT_S3} \\
    --sections-b64 {sections_b64}{limit_arg}
""".strip()


def build_request(*, label: str, model: str, limit: int | None, suffix: str) -> JobRequest:
    return JobRequest(
        name=f"{JOB_PREFIX}-{label}{suffix}",
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(label=label, model=model, limit=limit)]
        ),
        resources=ResourceConfig.with_gpu(
            "H100", count=1, image=GPU_IMAGE, cpu=8, ram="64g", disk="128g"
        ),
        # setup_scripts=[] disables iris's default `uv sync`: with no workspace
        # bundle there is no pyproject to sync and the step fails outright.
        environment=create_environment(
            docker_image=GPU_IMAGE, env_vars={}, setup_scripts=[]
        ),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=3,
        max_retries_preemption=100,
    )


def submit(client, requests: list[JobRequest], *, must_wait: bool) -> None:
    """Submit, and when waiting, fail the driver if any child failed.

    `wait()` raises on a failed job, so waiting in a plain loop abandons every
    remaining wait and reports only the first failure. Each is caught so the rest
    still run -- and then re-raised together, because a driver that exits 0 while
    a child died reports a result that does not exist.
    """
    jobs = [client.submit(request) for request in requests]
    print(f"[complex-loss] submitted {len(jobs)} job(s)", flush=True)
    for request in requests:
        print(f"    {request.name}")
    if not must_wait:
        return
    failures = []
    for request, job in zip(requests, jobs, strict=True):
        try:
            job.wait()
        except Exception as error:  # noqa: BLE001 - collected, then re-raised
            print(f"[complex-loss] FAILED {request.name}: {error}", flush=True)
            failures.append(request.name)
    if failures:
        raise RuntimeError(f"{len(failures)} of {len(jobs)} child job(s) failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=",".join(ARMS))
    parser.add_argument(
        "--model-uri",
        help="score an export that is not a named arm (e.g. a training smoke's "
        "tagged export). Requires exactly one --labels value, used as its label.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="smoke: first N documents"
    )
    parser.add_argument(
        "--name-suffix",
        default="",
        help="appended to job names -- iris names are unique, so a retry needs one",
    )
    parser.add_argument("--cluster", default=os.environ.get("EVAL_CW_CLUSTER", "cw-us-east-02a"))
    arguments = parser.parse_args()
    labels = [label for label in arguments.labels.split(",") if label]
    if arguments.model_uri:
        if len(labels) != 1:
            parser.error("--model-uri takes exactly one --labels value")
        arms = {labels[0]: arguments.model_uri}
    else:
        arms = ARMS
        unknown = [label for label in labels if label not in arms]
        if unknown:
            parser.error(f"unknown label(s) {unknown}; known: {sorted(arms)}")
    requests = [
        build_request(
            label=label,
            model=arms[label],
            limit=arguments.limit,
            suffix=arguments.name_suffix,
        )
        for label in labels
    ]
    print(
        f"[complex-loss] {len(requests)} job(s), 1xH100 batch band, image={GPU_IMAGE}\n"
        f"              shard={HELD_OUT_SHARD}\n              out={OUT_S3}"
    )
    if os.environ.get("EVAL_CW_DRY_RUN"):
        print("[complex-loss] DRY RUN -- JobRequests built, not submitting.")
        print(requests[0].entrypoint.binary_entrypoint.args[1])
        return

    from iris.client.client import get_iris_ctx

    if get_iris_ctx() is not None:
        # In-cluster driver: these are its children and iris kills children when
        # a parent exits, so it must outlive them.
        submit(current_client(), requests, must_wait=True)
        return

    # Workstation submission. `current_client()` would fall back to LocalClient
    # and try to run every H100 job on this box, so build the iris-backed client
    # explicitly over the CLI's controller tunnel. These become ROOT jobs.
    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    print(f"[complex-loss] submitting via the {arguments.cluster} controller tunnel")
    with open_iris_client(cluster_name=arguments.cluster, workspace=None) as iris_client:
        submit(FrayIrisClient.from_iris_client(iris_client), requests, must_wait=False)


if __name__ == "__main__":
    main()
