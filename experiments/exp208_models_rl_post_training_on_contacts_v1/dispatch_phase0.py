# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 — generate the exp199 baseline rollouts, with per-rollout dumps — issue #208.

Runs exp82's ``score_rollout_worker.py`` **unmodified except for its opt-in
``--dump-rollouts`` flag**, which is the point: the vote matrices are directly
comparable to every published MarinFold R-precision, and the per-rollout dump is
strictly additional output.

Two jobs in one run:

* the **vote parquets** re-measure the exp199 baseline through exp208's own eval
  invocation — the Phase 1 parity gate. It should land within 0.0023 (#180's
  four-repeat span) of the committed 0.587348 / 0.542181; if it does not,
  something about this experiment's eval path differs from the published one and
  every later comparison is suspect.
* the **rollout dumps** feed ``phase0_marginal_analysis.py``, which decides
  whether #208's consensus-marginal document term is worth building.

SUBMITTED FROM THIS DIRECTORY, NOT FROM A MARIN CHECKOUT. exp169's equivalent
dispatcher submits from the marin source tree because ``--extra vllm --extra tpu``
are marin's own extras. That stopped working on 2026-08-07, when marin main
deleted the ``vllm`` extra (e7ef104402) — a checkout new enough to clear iris's
14-day client freshness gate no longer defines the extra at all, and the pod
fails at build with ``Extra `vllm` is not defined in any project's
optional-dependencies table``. Measured here on 2026-08-10 against a 2026-08-08
checkout: all four shards failed identically. exp200 already solved this by making
the experiment directory the complete iris workspace with marin pinned at
0.2.76.dev31155643335 and the vLLM TPU fork reproduced, so exp208 submits through
:mod:`_submit` like every other exp208 job.

The worker travels base64-embedded rather than as a workspace file, so exp82
stays the single source of truth for it.

    ./stage_model_gcs.sh                          # once: exp199 -> GCS, bf16
    uv run python dispatch_phase0.py --num-shards 4 --dump-rollouts 25
    uv run python dispatch_phase0.py --shards 0 --limit 2 --dry-run
"""

import argparse
import base64
import os
from pathlib import Path

from _submit import check_clean, submit

GCS_PREFIX = os.environ.get(
    "EXP208_PREFIX", "gs://marin-us-central1/protein-structure/MarinFold/exp208")
# exp169's eval targets: the same 554 proteins the committed baseline rows were
# scored on, already co-located with the v5p pool.
TARGETS = os.environ.get(
    "EXP208_TARGETS",
    "gs://marin-us-central1/protein-structure/MarinFold/exp169/eval_targets.parquet")
OUT = os.environ.get("EXP208_PHASE0_OUT", f"{GCS_PREFIX}/phase0/scores")
MODEL = os.environ.get("EXP208_MODEL", f"{GCS_PREFIX}/models/exp199")
LABEL = os.environ.get("EXP208_LABEL", "exp199_cw_p06_aug_step145199")

MARINFOLD_GIT = os.environ.get(
    "EXP208_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold",
)

# exp82's settled recipe, unchanged. top_k OFF: #142 traced under-generation to a
# finite top_k, and #82 found T=1.0/p=0.95 near-optimal — sharpening past it
# collapses the vote, which is the very effect #208 must not cause.
N_ROLLOUTS = int(os.environ.get("EXP208_N_ROLLOUTS", "100"))
TOP_K = int(os.environ.get("EXP208_TOP_K", "-1"))
TOP_P = float(os.environ.get("EXP208_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EXP208_TEMPERATURE", "1.0"))
# Engine-level seed. With --no-per-request-seed (required on TPU) this is the ONLY
# source of sampling randomness, so a replicate must change it: re-running with
# the same seed on the same stack risks reproducing the same 100 rollouts and
# measuring nothing. exp208's first parity run used 0.
SEED = int(os.environ.get("EXP208_SEED", "0"))

WORKER_SCRIPT = (Path(__file__).resolve().parent.parent
                 / "exp82_evals_contacts_v1_contact_prediction" / "score_rollout_worker.py")
WORKER_LOCAL = "/tmp/exp208/score_rollout_worker.py"


def build_bootstrap(*, shard_i: int, num_shards: int, limit: int | None, dump: int) -> str:
    """Pod bootstrap: drop the worker on disk, add marinfold, run it.

    ``--no-deps`` so marinfold cannot repin anything in the pinned vLLM-TPU
    stack; the contacts-v1 generator needs only fsspec + numpy on top.
    """
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit else ""
    dump_arg = f" --dump-rollouts {dump}" if dump else ""
    return f"""
set -euo pipefail
echo "[phase0] host=$(hostname) shard={shard_i}/{num_shards} dump={dump}"

mkdir -p /tmp/exp208
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

uv pip install --quiet --no-deps "{MARINFOLD_GIT}"
uv run --no-sync python -c \\
  "from marinfold.document_structures.contacts_v1 import build_document; print('[phase0] marinfold OK')"

exec uv run --no-sync python {WORKER_LOCAL} \\
    --model {MODEL} \\
    --targets {TARGETS} \\
    --out {OUT} \\
    --label {LABEL} \\
    --shard {shard_i}/{num_shards} \\
    --n-rollouts {N_ROLLOUTS} \\
    --temperature {TEMPERATURE} \\
    --top-p {TOP_P} \\
    --top-k {TOP_K} \\
    --seed {SEED} \\
    --no-per-request-seed{limit_arg}{dump_arg}
""".strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-shards", type=int, default=4)
    ap.add_argument("--shards", default=None, help="comma-separated subset, e.g. '0'")
    ap.add_argument("--dump-rollouts", type=int, default=25,
                    help="per-rollout pair sets for the first N proteins OF EACH SHARD, so "
                         "4 shards x 25 gives ~100 proteins with a spread of lengths")
    ap.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    ap.add_argument("--tpu", default="v5p-8")
    ap.add_argument("--priority", default="interactive",
                    choices=["production", "interactive", "batch"])
    ap.add_argument("--zone", default="us-central1-a")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not WORKER_SCRIPT.exists():
        raise SystemExit(f"worker not found: {WORKER_SCRIPT}")
    if "--dump-rollouts" not in WORKER_SCRIPT.read_text():
        raise SystemExit(
            f"{WORKER_SCRIPT} has no --dump-rollouts flag. Phase 0 needs the per-rollout "
            "dump; the flag is opt-in and added by exp208, so a checkout without it would "
            "run an ordinary eval and silently produce no marginal data."
        )
    if not a.dry_run:
        check_clean()

    which = [int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards))
    print(f"[phase0] {len(which)} shard(s) on {a.tpu} in {a.zone} | n_rollouts={N_ROLLOUTS} "
          f"dump={a.dump_rollouts} limit={a.limit}\n         model={MODEL}\n"
          f"         targets={TARGETS}\n         out={OUT}/{LABEL}")

    names = []
    for shard_i in which:
        names.append(submit(
            job_name=f"exp208-phase0-s{shard_i}of{a.num_shards}",
            command=["bash", "-lc", build_bootstrap(
                shard_i=shard_i, num_shards=a.num_shards, limit=a.limit, dump=a.dump_rollouts)],
            raw=True,
            extras=("tpu", "vllm"),
            tpu=a.tpu,
            zone=a.zone,
            priority=a.priority,
            cpu=8, memory="64GB", disk="64GB",
            max_retries=3,
            dry_run=a.dry_run,
        ))
    print(f"[phase0] submitted {len(names)} job(s)")
    for name in names:
        print(f"    /bizon/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
