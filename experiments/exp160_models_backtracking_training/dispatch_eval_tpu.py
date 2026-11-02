# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fan the retraction-aware rollout eval out over marin v5p-8 TPU slices (#160).

Runs ``score_backtracking_worker.py`` on both arms of the comparison:

* ``exp160-bt50`` — the model trained on the 50:50 backtracking mix, and
* ``exp120-base`` — the checkpoint it was fine-tuned from, the control.

Both arms run the *same* worker with the *same* recipe, which is the point: the
published #82/#169 numbers were produced by a slightly different budget
(``contact_mult=6``) and by a regex readout that predates ``<retract>``, so a
comparison against them would confound the training effect with the harness.
The control is re-measured here rather than quoted.

Structured after exp169's ``dispatch_eval_tpu.py``, with the differences:

* **zone ``us-central1-a``.** The training run went to ``us-east5-a`` because
  that was where v5p-**32** slices were being provisioned; the v5p-**8** picture
  is the opposite. Live autoscaler counts on 2026-07-28:

  | scale group | ready | demand |
  |---|---|---|
  | ``tpu_v5p-preemptible_8-us-east5-a`` | 7 | 0 |
  | ``tpu_v5p-preemptible_8-us-central1-a`` | 84 | 62 |

  A v5p-8 submitted to ``us-east5-a`` sat pending for 13 minutes against those
  7 fully-occupied slices while registering no autoscaler demand at all. Read
  the per-*size* group, not the zone's reputation.

  Eval assets are therefore mirrored into ``gs://marin-us-central1`` so reads
  stay slice-local. That mirror is a **server-side bucket-to-bucket copy** —
  2.7 GiB in 48 s, and it never touches the workstation uplink, which moves the
  same bytes in ~16 minutes.
* **``marinfold`` comes from this branch, pinned to a commit.** The
  ``<retract>`` fold (``read.py``, #158) is not on ``main``, so a worker
  installed from ``main`` would parse retractions with a contact-only regex and
  silently score the backtracking arm as if it had never taken anything back.
  The commit is resolved from the local checkout and asserted to be on the
  remote, so the pod runs code that is actually published.
* **interactive priority.** A bounded, minutes-long eval; the v5p pool is
  usually subscribed by other people's interactive jobs, so a batch-band job
  yields to them indefinitely. (The always-batch rule is for long CoreWeave GPU
  training runs, not this.)

    uv run python dispatch_eval_tpu.py --num-shards 4
    uv run python dispatch_eval_tpu.py --labels exp160-bt50 --shards 0 --limit 2   # smoke
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
from pathlib import Path

MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/home/bizon/git/marin-freshiris"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN / ".venv/bin/iris"))
# `iris job run` bundles the CWD as the job's workspace and runs `uv sync` in it
# on the pod, and `--extra vllm --extra tpu` are marin's extras — so the
# workspace has to BE the marin checkout. It must also be the *fresh* one: iris
# rejects a client more than 14 days old.
SUBMIT_WORKSPACE = Path(os.environ.get("EVAL_TPU_WORKSPACE", str(MARIN)))

PREFIX = os.environ.get(
    "EXP160_EVAL_PREFIX",
    "gs://marin-us-central1/protein-structure/MarinFold/exp160_backtracking_training/eval")
TARGETS = os.environ.get("EXP160_EVAL_TARGETS", f"{PREFIX}/eval_targets.parquet")
OUT = os.environ.get("EXP160_EVAL_OUT", f"{PREFIX}/scores")
MODELS_PREFIX = f"{PREFIX}/models"

REPO_URL = "https://github.com/Open-Athena/MarinFold.git"

# label -> model directory under MODELS_PREFIX.
ARMS = {
    "exp160-bt50": "exp160-bt50-step2059",
    "exp120-base": "exp120-base",
}

# exp82's settled recipe. `contact_mult` is 8, not 6: retraction lengthens
# documents, and a budget that truncates one arm and not the other would confound
# the comparison. Both arms get the same number.
N_ROLLOUTS = int(os.environ.get("EXP160_N_ROLLOUTS", "100"))
TOP_K = int(os.environ.get("EXP160_TOP_K", "-1"))
TOP_P = float(os.environ.get("EXP160_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EXP160_TEMPERATURE", "1.0"))
CONTACT_MULT = int(os.environ.get("EXP160_CONTACT_MULT", "8"))

WORKER_SCRIPT = Path(__file__).resolve().parent / "score_backtracking_worker.py"
WORKER_LOCAL = "/tmp/exp160/score_backtracking_worker.py"


def resolve_commit(explicit: str | None) -> str:
    """The commit the pod installs ``marinfold`` from, asserted to be on origin.

    A local-only commit would install fine from a cached clone here and fail (or,
    worse, silently resolve to something else) on the pod, so the check is
    against the remote rather than the working tree.
    """
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
    print(f"[eval] marinfold @ {sha[:12]} ({remote.stdout.split()[0]})")
    return sha


def build_bootstrap(*, label: str, model_dir: str, shard_i: int, num_shards: int,
                    limit: int | None, commit: str) -> str:
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    return f"""
set -euo pipefail
echo "[eval] host=$(hostname) label={label} shard={shard_i}/{num_shards}"

mkdir -p /tmp/exp160
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

# marin's synced venv already has vLLM (its TPU fork), transformers, pyarrow,
# fsspec and gcsfs. marinfold goes in --no-deps so it cannot repin any of that;
# the contacts-v1 generator + read fold need only fsspec + numpy on top.
uv pip install --quiet --no-deps \\
  "marinfold @ git+{REPO_URL}@{commit}#subdirectory=marinfold"
uv run --no-sync python -c \\
  "from marinfold.document_structures.contacts_v1.read import fold_statements; \\
   from marinfold.document_structures.contacts_v1 import build_document; \\
   print('[eval] marinfold OK (retract fold present)')"

exec uv run --no-sync python {WORKER_LOCAL} \\
    --model {MODELS_PREFIX}/{model_dir} \\
    --targets {TARGETS} \\
    --out {OUT} \\
    --label {label} \\
    --shard {shard_i}/{num_shards} \\
    --n-rollouts {N_ROLLOUTS} \\
    --temperature {TEMPERATURE} \\
    --top-p {TOP_P} \\
    --top-k {TOP_K} \\
    --contact-mult {CONTACT_MULT} \\
    --no-per-request-seed{limit_arg}
""".strip()


def submit(*, label: str, model_dir: str, shard_i: int, num_shards: int, limit: int | None,
           commit: str, tpu: str, zone: str, priority: str, dry_run: bool) -> str:
    name = f"exp160-eval-{label}-s{shard_i}of{num_shards}"
    command = [
        IRIS, "--cluster=marin", "job", "run",
        "--job-name", name, "--no-wait", "--enable-extra-resources",
        "--priority", priority, "--zone", zone, "--tpu", tpu,
        "--extra", "vllm", "--extra", "tpu",
        # A v5p-8 VM offers 100 GiB of ephemeral disk; asking for more is
        # rejected outright as unschedulable rather than queued.
        "--cpu", "8", "--memory", "64GB", "--disk", "64GB",
        "--max-retries", "3",
        "--", "bash", "-lc",
        build_bootstrap(label=label, model_dir=model_dir, shard_i=shard_i,
                        num_shards=num_shards, limit=limit, commit=commit),
    ]
    if dry_run:
        print(f"[eval] DRY RUN {name}\n{command[-1][:1400]}")
        return name
    subprocess.run(command, cwd=SUBMIT_WORKSPACE, check=True)
    return name


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--labels", default=",".join(ARMS),
                    help="comma-separated subset of the arms to run")
    ap.add_argument("--shards", default=None, help="comma-separated subset, e.g. '0'")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    ap.add_argument("--commit", default=None, help="marinfold commit (default: local HEAD)")
    ap.add_argument("--tpu", default="v5p-8")
    ap.add_argument("--priority", default="interactive",
                    choices=["production", "interactive", "batch"])
    ap.add_argument("--zone", default="us-central1-a")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    labels = [x for x in a.labels.split(",") if x]
    unknown = [x for x in labels if x not in ARMS]
    if unknown:
        ap.error(f"unknown label(s) {unknown}; known: {sorted(ARMS)}")
    which = ([int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards)))
    commit = resolve_commit(a.commit)

    print(f"[eval] {len(labels)} arm(s) x {len(which)} shard(s) on {a.tpu} in {a.zone} "
          f"| n_rollouts={N_ROLLOUTS} top_k={TOP_K} top_p={TOP_P} T={TEMPERATURE} "
          f"contact_mult={CONTACT_MULT} limit={a.limit}\n"
          f"       targets={TARGETS}\n       out={OUT}")

    submitted = [submit(label=label, model_dir=ARMS[label], shard_i=i, num_shards=a.num_shards,
                        limit=a.limit, commit=commit, tpu=a.tpu, zone=a.zone,
                        priority=a.priority, dry_run=a.dry_run)
                 for label in labels for i in which]
    print(f"[eval] submitted {len(submitted)} job(s)")
    for name in submitted:
        print(f"    /bizon/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
