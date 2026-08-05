# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch exp124 think-masked validation-loss recomputes on Iris."""

import argparse
import base64
import os
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "recompute_think_masked_val.py"
IRIS = os.environ.get("IRIS_BIN", "/Users/zack/projects/agent_workspaces/repos/marin-alpha/.venv/bin/iris")
OUT_PREFIX = os.environ.get(
    "EXP124_EVAL_PREFIX",
    "gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/eval_loss",
)
CACHE_DIR = os.environ.get(
    "EXP124_THINK_CACHE_ROOT",
    "gs://marin-us-east5/protein-structure/MarinFold/exp124_contacts_v1_think_loss_masked/cache/think-masked/2026.07.29.2",
)
PRIORITY = os.environ.get("EXP124_PRIORITY", "batch")

MODELS = {
    "exp117_e16_final_step35679": (
        "gs://marin-us-central1/protein-structure/MarinFold/exp169/models/exp117_e16_final_step35679"
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", choices=sorted(MODELS))
    parser.add_argument("--model", default=None, help="Override model path for the selected label.")
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--job-suffix", default=None)
    parser.add_argument("--zone", default="us-east5-a")
    parser.add_argument("--tpu", default="v5p-8")
    parser.add_argument("--memory", default="32GB")
    parser.add_argument("--cpu", default="8")
    args = parser.parse_args()

    model = args.model or MODELS[args.label]
    suffix = args.job_suffix or (
        f"{args.label}-full" if args.max_eval_batches is None else f"{args.label}-{args.max_eval_batches}b"
    )
    output = f"{OUT_PREFIX}/{suffix}.json"
    script_b64 = base64.b64encode(SCRIPT.read_bytes()).decode()
    limit = f" --max-eval-batches {args.max_eval_batches}" if args.max_eval_batches is not None else ""
    command_text = f"""
set -euo pipefail
mkdir -p /tmp/exp124
echo {script_b64} | base64 -d > /tmp/exp124/recompute_think_masked_val.py
exec uv run --no-sync python /tmp/exp124/recompute_think_masked_val.py \
  --label {args.label} \
  --model {model} \
  --cache-dir {CACHE_DIR} \
  --output {output}{limit}
""".strip()
    job_name = f"exp124-think-val-{suffix}"
    command = [
        IRIS,
        "--cluster=marin",
        "job",
        "run",
        "--job-name",
        job_name,
        "--no-wait",
        "--enable-extra-resources",
        "--priority",
        PRIORITY,
        "--zone",
        args.zone,
        "--tpu",
        args.tpu,
        "--extra",
        "tpu",
        "--cpu",
        args.cpu,
        "--memory",
        args.memory,
        "--disk",
        "64GB",
        "--max-retries",
        "1",
        "--",
        "bash",
        "-lc",
        command_text,
    ]
    subprocess.run(command, cwd=HERE, check=True)
    print(f"/zack/{job_name}\noutput={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
