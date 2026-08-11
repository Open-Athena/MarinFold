# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 0 — generate the exp199 baseline rollouts, with per-rollout dumps — issue #208.

This is exp169's ``dispatch_eval_tpu.py`` pointed at exp199 and asking exp82's
``score_rollout_worker.py`` for ``--dump-rollouts``. It runs the worker
**unmodified in every other respect**, which is the whole point: the vote
matrices it produces are directly comparable to every published MarinFold
R-precision, and the per-rollout dump is strictly additional output.

Two jobs in one run, and it is worth being explicit about which is which:

* the **vote parquets** re-measure the exp199 baseline through exp208's own eval
  invocation — the Phase 1 parity gate. It should land within 0.0023 (#180's
  four-repeat span) of the committed 0.587348 / 0.542181, and if it does not,
  something about this experiment's eval path differs from the published one and
  every later comparison is suspect.
* the **rollout dumps** feed ``phase0_marginal_analysis.py``, which decides
  whether #208's consensus-marginal document term is worth building.

Targets are exp169's ``eval_targets.parquet``, already in us-central1 next to the
v5p capacity — no new staging, and identical protein set to the committed
baseline rows.

    ./stage_model_gcs.sh                          # once: exp199 -> GCS, bf16
    uv run python dispatch_phase0.py --num-shards 4 --dump-rollouts 25
    uv run python dispatch_phase0.py --num-shards 4 --shards 0 --limit 2 --dry-run
"""

import argparse
import base64
import os
import subprocess
from pathlib import Path

MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/home/bizon/git/marin-freshiris"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN / ".venv/bin/iris"))
# `iris job run` bundles the CWD and runs `uv sync` on the pod; `--extra vllm
# --extra tpu` are marin's own extras, so the workspace has to BE the marin
# checkout. It must also be the FRESH one -- iris rejects a client over 14 days
# old, and the frozen marin-*-latest wheels are always older than that.
SUBMIT_WORKSPACE = Path(os.environ.get("EXP208_WORKSPACE", str(MARIN)))

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
# finite top_k, and #82 found T=1.0/p=0.95 near-optimal -- sharpening past it
# collapses the vote, which is the very effect #208 is trying not to cause.
N_ROLLOUTS = int(os.environ.get("EXP208_N_ROLLOUTS", "100"))
TOP_K = int(os.environ.get("EXP208_TOP_K", "-1"))
TOP_P = float(os.environ.get("EXP208_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EXP208_TEMPERATURE", "1.0"))

WORKER_SCRIPT = (Path(__file__).resolve().parent.parent
                 / "exp82_evals_contacts_v1_contact_prediction" / "score_rollout_worker.py")
WORKER_LOCAL = "/tmp/exp208/score_rollout_worker.py"


def build_bootstrap(*, shard_i: int, num_shards: int, limit: int | None, dump: int) -> str:
    """Pod bootstrap: install marinfold into marin's synced venv, run the worker.

    ``--no-deps`` so marinfold cannot repin anything in marin's vLLM-TPU fork
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
    --no-per-request-seed{limit_arg}{dump_arg}
""".strip()


def submit(*, shard_i: int, num_shards: int, limit: int | None, dump: int,
           tpu: str, zone: str, priority: str, dry_run: bool) -> str:
    name = f"exp208-phase0-s{shard_i}of{num_shards}"
    command = [
        IRIS, "--cluster=marin", "job", "run",
        "--job-name", name, "--no-wait", "--enable-extra-resources",
        # `interactive`, not `batch`: the v5p pool is fully subscribed by other
        # people's interactive jobs, so a batch-band job yields to them
        # indefinitely, and this is the bounded minutes-long shape interactive
        # exists for. The always-batch rule is a CoreWeave-GPU rule and does not
        # carry over to the marin TPU pool.
        "--priority", priority, "--zone", zone, "--tpu", tpu,
        "--extra", "vllm", "--extra", "tpu",
        # A v5p-8 VM offers 100 GiB ephemeral disk; asking for more is rejected
        # as unschedulable rather than queued.
        "--cpu", "8", "--memory", "64GB", "--disk", "64GB",
        "--max-retries", "3",
        "--", "bash", "-lc",
        build_bootstrap(shard_i=shard_i, num_shards=num_shards, limit=limit, dump=dump),
    ]
    if dry_run:
        print(f"[phase0] DRY RUN {name}\n{command[-1]}\n")
        return name
    SUBMIT_WORKSPACE.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=SUBMIT_WORKSPACE, check=True)
    return name


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
            "dump; the flag is added by exp208 and is opt-in, so a checkout without it "
            "would run a normal eval and silently produce no marginal data."
        )

    which = [int(x) for x in a.shards.split(",")] if a.shards else list(range(a.num_shards))
    print(f"[phase0] {len(which)} shard(s) on {a.tpu} in {a.zone} | n_rollouts={N_ROLLOUTS} "
          f"dump={a.dump_rollouts} limit={a.limit}\n         model={MODEL}\n"
          f"         targets={TARGETS}\n         out={OUT}/{LABEL}")

    submitted = [submit(shard_i=i, num_shards=a.num_shards, limit=a.limit,
                        dump=a.dump_rollouts, tpu=a.tpu, zone=a.zone,
                        priority=a.priority, dry_run=a.dry_run) for i in which]
    print(f"[phase0] submitted {len(submitted)} job(s)")
    for name in submitted:
        print(f"    /bizon/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
