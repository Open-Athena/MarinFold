# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch exp124 raw rollout logging to Marin v5p TPU slices.

This is the GCS/TPU twin of ``dispatch_log_rollouts_cw.py``. It keeps all
large I/O in the same region as the exp124 checkpoint: model, targets, and
outputs are under ``gs://marin-us-east5/protein-structure/MarinFold/...``.
"""

import argparse
import base64
import os
import subprocess
from pathlib import Path


MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/Users/zack/projects/agent_workspaces/repos/marin-alpha"))
IRIS = os.environ.get("IRIS_BIN", str(MARIN / ".venv/bin/iris"))
SUBMIT_WORKSPACE = Path(os.environ.get("EXP124_ROLLOUT_TPU_WORKSPACE", str(MARIN)))

MARINFOLD_GIT = os.environ.get(
    "EXP124_ROLLOUT_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@exp124/think-loss-masked#subdirectory=marinfold",
)

GCS_PREFIX = os.environ.get(
    "EXP124_ROLLOUT_GCS_PREFIX",
    "gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/raw_rollouts",
)
TARGETS = os.environ.get("EXP124_ROLLOUT_TARGETS", f"{GCS_PREFIX}/eval_targets.parquet")
OUT = os.environ.get("EXP124_ROLLOUT_OUT", f"{GCS_PREFIX}/raw")
DEFAULT_MODEL = os.environ.get(
    "EXP124_ROLLOUT_MODEL",
    "gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/"
    "checkpoints/exp124-cv1-think-masked-e16-lr3p162e-3-wd0p2-bs256-next_token-exp177recipe-v5p128-r3/"
    "2026.07.30.4/hf/step-35680",
)
DEFAULT_LABEL = os.environ.get("EXP124_ROLLOUT_LABEL", "exp124_step35680")
JOB_PREFIX = os.environ.get("EXP124_ROLLOUT_JOB_PREFIX", "exp124-rawrollout-tpu")

N_ROLLOUTS = int(os.environ.get("EXP124_ROLLOUT_N", "100"))
TOP_K = int(os.environ.get("EXP124_ROLLOUT_TOP_K", "-1"))
TOP_P = float(os.environ.get("EXP124_ROLLOUT_TOP_P", "0.95"))
TEMPERATURE = float(os.environ.get("EXP124_ROLLOUT_TEMPERATURE", "1.0"))

WORKER_SCRIPT = Path(__file__).with_name("log_rollout_worker.py")
WORK_DIR = "/tmp/exp124_raw_rollouts"
WORKER_LOCAL = f"{WORK_DIR}/log_rollout_worker.py"


def build_bootstrap(*, label: str, model: str, shard_i: int, num_shards: int, limit: int | None) -> str:
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit is not None else ""
    return f"""
set -euo pipefail
echo "[exp124-rollout-tpu] host=$(hostname) label={label} shard={shard_i}/{num_shards}"

mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

# marin's synced env has the supported vLLM TPU fork. Install only marinfold's
# package code so it cannot perturb that pinned TPU stack.
uv pip install --quiet --no-deps "{MARINFOLD_GIT}"
uv run --no-sync python -c \
  "from marinfold.document_structures.contacts_v1 import build_document; print('[exp124-rollout-tpu] marinfold OK')"

exec uv run --no-sync python {WORKER_LOCAL} \
  --model {model} \
  --targets {TARGETS} \
  --out {OUT} \
  --label {label} \
  --shard {shard_i}/{num_shards} \
  --n-rollouts {N_ROLLOUTS} \
  --temperature {TEMPERATURE} \
  --top-p {TOP_P} \
  --top-k {TOP_K} \
  --no-per-request-seed{limit_arg}
""".strip()


def submit(
    *,
    label: str,
    model: str,
    shard_i: int,
    num_shards: int,
    limit: int | None,
    name_suffix: str,
    tpu: str,
    zone: str,
    priority: str,
    dry_run: bool,
) -> str:
    name = f"{JOB_PREFIX}-{label.replace('_', '-')}-s{shard_i}of{num_shards}{name_suffix}"
    command = [
        IRIS,
        "--cluster=marin",
        "job",
        "run",
        "--job-name",
        name,
        "--no-wait",
        "--enable-extra-resources",
        "--priority",
        priority,
        "--zone",
        zone,
        "--tpu",
        tpu,
        "--extra",
        "vllm",
        "--extra",
        "tpu",
        "--cpu",
        "8",
        "--memory",
        "64GB",
        "--disk",
        "64GB",
        "--max-retries",
        "3",
        "--",
        "bash",
        "-lc",
        build_bootstrap(label=label, model=model, shard_i=shard_i, num_shards=num_shards, limit=limit),
    ]
    if dry_run:
        print(f"[exp124-rollout-tpu] DRY RUN {name}\n{command[-1][:1600]}")
        return name
    SUBMIT_WORKSPACE.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=SUBMIT_WORKSPACE, check=True)
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("EXP124_ROLLOUT_SHARDS", "12")))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--label", default=DEFAULT_LABEL)
    parser.add_argument("--limit", type=int, default=None, help="smoke: first N targets per shard")
    parser.add_argument("--shards", default=None, help="comma-separated subset; default all")
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--tpu", default="v5p-8")
    parser.add_argument("--zone", default="us-east5-a")
    parser.add_argument("--priority", default="interactive", choices=["production", "interactive", "batch"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    which = [int(x) for x in args.shards.split(",")] if args.shards else list(range(args.num_shards))
    print(
        f"[exp124-rollout-tpu] {len(which)} job(s) on {args.tpu} in {args.zone}\n"
        f"  model={args.model}\n  targets={TARGETS}\n  out={OUT}/{args.label}\n"
        f"  n_rollouts={N_ROLLOUTS} top_k={TOP_K} top_p={TOP_P} T={TEMPERATURE} limit={args.limit}",
        flush=True,
    )

    submitted = [
        submit(
            label=args.label,
            model=args.model,
            shard_i=shard,
            num_shards=args.num_shards,
            limit=args.limit,
            name_suffix=args.name_suffix,
            tpu=args.tpu,
            zone=args.zone,
            priority=args.priority,
            dry_run=args.dry_run,
        )
        for shard in which
    ]
    print(f"[exp124-rollout-tpu] submitted {len(submitted)} job(s)")
    for name in submitted:
        print(f"    /zack/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
