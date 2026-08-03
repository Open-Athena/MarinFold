# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch exp188 per-document eval-loss scoring on marin TPU vLLM pods."""

import argparse
import base64
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKER_SCRIPT = HERE / "score_eval_loss_vllm_worker.py"
WORKER_LOCAL = "/tmp/exp188/score_eval_loss_vllm_worker.py"

MARIN = Path(os.environ.get("MARIN_CHECKOUT", "/Users/zack/projects/agent_workspaces/repos/marin-beta"))
IRIS = os.environ.get("IRIS_BIN", "/Users/zack/projects/agent_workspaces/beta/.venv-iris/bin/iris")
SUBMIT_WORKSPACE = Path(os.environ.get("EVAL_TPU_WORKSPACE", str(MARIN)))

GCS_PREFIX = os.environ.get("EXP188_PREFIX", "gs://marin-us-east5/protein-structure/MarinFold/exp188")
MODEL = os.environ.get(
    "EXP188_MODEL",
    "gs://marin-us-central1/protein-structure/MarinFold/exp169/models/exp117_e16_final_step35679",
)
INPUT_GLOB = os.environ.get(
    "EXP188_INPUT_GLOB",
    "gs://marin-us-east5/protein-structure/MarinFold/exp53_contacts_v1_5x/documents/val/*.parquet",
)
OUT = os.environ.get("EXP188_OUT", f"{GCS_PREFIX}/per_doc_loss/exp117_e16_final_step35679")


def build_bootstrap(*, shard_i: int, num_shards: int, limit: int | None, chunk: int) -> str:
    worker_b64 = base64.b64encode(WORKER_SCRIPT.read_bytes()).decode()
    limit_arg = f" --limit {limit}" if limit is not None else ""
    return f"""
set -euo pipefail
echo "[exp188] host=$(hostname) shard={shard_i}/{num_shards}"
mkdir -p /tmp/exp188
echo {worker_b64} | base64 -d > {WORKER_LOCAL}
exec uv run --no-sync python {WORKER_LOCAL} \
  --model {MODEL} \
  --input-glob '{INPUT_GLOB}' \
  --out {OUT} \
  --shard {shard_i}/{num_shards} \
  --chunk {chunk}{limit_arg}
""".strip()


def submit(*, shard_i: int, num_shards: int, limit: int | None, chunk: int, tpu: str, zone: str, priority: str, dry_run: bool) -> str:
    suffix = "smoke" if limit is not None else "full"
    name = f"exp188-loss-exp117-{suffix}-s{shard_i}of{num_shards}"
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
        build_bootstrap(shard_i=shard_i, num_shards=num_shards, limit=limit, chunk=chunk),
    ]
    if dry_run:
        print(f"[exp188] DRY RUN {name}\n{command[-1][:1600]}")
        return name
    subprocess.run(command, cwd=SUBMIT_WORKSPACE, check=True)
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shards", default=None, help="Comma-separated shard ids; default all.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--tpu", default="v5p-8")
    parser.add_argument("--zone", default="us-east5-a")
    parser.add_argument("--priority", default="interactive", choices=["production", "interactive", "batch"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    shards = [int(x) for x in args.shards.split(",")] if args.shards else list(range(args.num_shards))
    print(
        f"[exp188] scoring {len(shards)} shard(s) of {args.num_shards} on {args.tpu} in {args.zone}\n"
        f"         model={MODEL}\n         input={INPUT_GLOB}\n         out={OUT}\n         limit={args.limit} chunk={args.chunk}"
    )
    submitted = [
        submit(
            shard_i=shard_i,
            num_shards=args.num_shards,
            limit=args.limit,
            chunk=args.chunk,
            tpu=args.tpu,
            zone=args.zone,
            priority=args.priority,
            dry_run=args.dry_run,
        )
        for shard_i in shards
    ]
    print(f"[exp188] submitted {len(submitted)} job(s)")
    for name in submitted:
        print(f"    /zack/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
