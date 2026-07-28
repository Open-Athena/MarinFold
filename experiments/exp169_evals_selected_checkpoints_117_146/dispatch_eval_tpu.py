# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fan the rollout contact eval out over marin v5p-8 TPU slices.

The GPU twin of this is exp82's ``dispatch_rollout_eval_cw.py``. This exists
because on 2026-07-28 every amd64 CoreWeave cluster was fully committed —
rno-2a 512/512 with 229 GPUs of pending demand, us-east-02a 256/256 — while the
marin v5p pool had 83 idle ``v5p-8`` slices in ``us-central1-a`` and zero
demand. Same measurement, whichever accelerator has room.

**It runs exp82's ``score_rollout_worker.py`` unmodified.** That worker does all
of its I/O through fsspec, so pointing ``--model``/``--targets``/``--out`` at
``gs://`` instead of ``s3://`` is the entire porting story (one hard-coded
``s3://`` in its resume path was the single fix, made in exp82). Running the same
bytes on both backends is what lets these numbers be compared to the published
CoreWeave ones.

TPU specifics, all in the bootstrap rather than the worker:

* the pod syncs marin's ``vllm`` + ``tpu`` extras — marin builds vLLM from its
  own TPU fork, and that env is the supported way to run vLLM on a v5p;
* ``marinfold`` is installed ``--no-deps`` into that synced venv (it needs only
  fsspec + numpy on top) so nothing repins the fork's pinned stack;
* weights must already be **bf16** on GCS. TPU parameters are bf16, and vLLM
  shards the checkpoint as it loads: handing it fp32 weights is a known failure
  rather than a silent cast. ``prepare_hf_export.py`` + the staging jobs do this.

Submitted from the marin checkout, because that is the workspace ``--extra vllm
--extra tpu`` refer to; the dispatcher sets the CWD itself::

    uv run python dispatch_eval_tpu.py --num-shards 4
    uv run python dispatch_eval_tpu.py --num-shards 4 --shards 0 --limit 2   # smoke
"""

import argparse
import base64
import os
import subprocess
from pathlib import Path

MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/home/bizon/git/marin-freshiris"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN / ".venv/bin/iris"))
# `iris job run` bundles the CWD as the job's workspace and then runs `uv sync`
# in it on the pod. `--extra vllm --extra tpu` are marin's extras, so the
# workspace has to BE the marin checkout — submitting from an empty scratch dir
# gets "No `pyproject.toml` found" on the pod, and submitting from a large
# non-repo tree (/tmp) hangs uploading it. It must also be the *fresh* checkout:
# iris rejects a client more than 14 days old.
SUBMIT_WORKSPACE = Path(os.environ.get("EVAL_TPU_WORKSPACE", str(MARIN)))

GCS_PREFIX = os.environ.get(
    "EVAL_TPU_PREFIX", "gs://marin-us-central1/protein-structure/MarinFold/exp169")
TARGETS = os.environ.get("EVAL_TPU_TARGETS", f"{GCS_PREFIX}/eval_targets.parquet")
OUT = os.environ.get("EVAL_TPU_OUT", f"{GCS_PREFIX}/scores")
MODELS_PREFIX = f"{GCS_PREFIX}/models"

MARINFOLD_GIT = os.environ.get(
    "EVAL_TPU_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold",
)

# The three checkpoints issue #169 selected.
LABELS = (
    "exp117_e16_final_step35679",
    "exp117_e16_early_step33450",
    "exp146_3b_e8_step17839",
)

# exp82's settled recipe, unchanged.
N_ROLLOUTS = int(os.environ.get("EVAL_TPU_N_ROLLOUTS", "100"))
TOP_K = int(os.environ.get("EVAL_TPU_TOP_K", "-1"))
TOP_P = float(os.environ.get("EVAL_TPU_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EVAL_TPU_TEMPERATURE", "1.0"))

WORKER_SCRIPT = (Path(__file__).resolve().parent.parent
                 / "exp82_evals_contacts_v1_contact_prediction" / "score_rollout_worker.py")
WORKER_LOCAL = "/tmp/exp169/score_rollout_worker.py"


def build_bootstrap(*, label: str, shard_i: int, num_shards: int, limit: int | None) -> str:
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    return f"""
set -euo pipefail
echo "[eval-tpu] host=$(hostname) label={label} shard={shard_i}/{num_shards}"

mkdir -p /tmp/exp169
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

# marin's synced venv already has vLLM (its TPU fork), torch, transformers,
# pyarrow, fsspec and gcsfs. marinfold goes in --no-deps so it cannot repin any
# of that; the contacts_v1 generator needs only fsspec + numpy on top.
uv pip install --quiet --no-deps "{MARINFOLD_GIT}"
uv run --no-sync python -c \\
  "from marinfold.document_structures.contacts_v1 import build_document; print('[eval-tpu] marinfold OK')"

exec uv run --no-sync python {WORKER_LOCAL} \\
    --model {MODELS_PREFIX}/{label} \\
    --targets {TARGETS} \\
    --out {OUT} \\
    --label {label} \\
    --shard {shard_i}/{num_shards} \\
    --n-rollouts {N_ROLLOUTS} \\
    --temperature {TEMPERATURE} \\
    --top-p {TOP_P} \\
    --top-k {TOP_K}{limit_arg}
""".strip()


def submit(*, label: str, shard_i: int, num_shards: int, limit: int | None,
           tpu: str, zone: str, priority: str, dry_run: bool) -> str:
    name = f"exp169-tpu-{label.replace('_', '-')}-s{shard_i}of{num_shards}"
    command = [
        IRIS, "--cluster=marin", "job", "run",
        "--job-name", name, "--no-wait", "--enable-extra-resources",
        # `interactive` (the CLI default), not `batch`. The v5p pool is fully
        # subscribed by other people's interactive jobs, so a batch-band job
        # yields to them indefinitely — and this is exactly the bounded,
        # minutes-long shape interactive is for, not bulk work that should wait.
        # (The CoreWeave-GPU always-batch rule is about long training jobs on
        # that cluster and does not carry over here.)
        "--priority", priority, "--zone", zone, "--tpu", tpu,
        "--extra", "vllm", "--extra", "tpu",
        # A v5p-8 VM offers 100 GiB of ephemeral disk; asking for more is
        # rejected outright as unschedulable rather than queued. 64 GB holds the
        # staged checkpoint (≤ 5.6 GiB) with room to spare.
        "--cpu", "8", "--memory", "64GB", "--disk", "64GB",
        "--max-retries", "3",
        "--", "bash", "-lc",
        build_bootstrap(label=label, shard_i=shard_i, num_shards=num_shards, limit=limit),
    ]
    if dry_run:
        print(f"[eval-tpu] DRY RUN {name}\n{command[-1][:1200]}")
        return name
    SUBMIT_WORKSPACE.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=SUBMIT_WORKSPACE, check=True)
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--labels", default=",".join(LABELS),
                    help="comma-separated subset of the three checkpoints")
    ap.add_argument("--shards", default=None, help="comma-separated subset, e.g. '0'")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    ap.add_argument("--tpu", default="v5p-8")
    ap.add_argument("--priority", default="interactive",
                    choices=["production", "interactive", "batch"])
    ap.add_argument("--zone", default="us-central1-a")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    labels = [x for x in a.labels.split(",") if x]
    which = ([int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards)))
    print(f"[eval-tpu] {len(labels)} model(s) x {len(which)} shard(s) on {a.tpu} in {a.zone} "
          f"| n_rollouts={N_ROLLOUTS} top_k={TOP_K} top_p={TOP_P} T={TEMPERATURE} "
          f"limit={a.limit}\n           targets={TARGETS}\n           out={OUT}")

    submitted = [submit(label=label, shard_i=i, num_shards=a.num_shards, limit=a.limit,
                        tpu=a.tpu, zone=a.zone, priority=a.priority, dry_run=a.dry_run)
                 for label in labels for i in which]
    print(f"[eval-tpu] submitted {len(submitted)} job(s)")
    for name in submitted:
        print(f"    /bizon/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
