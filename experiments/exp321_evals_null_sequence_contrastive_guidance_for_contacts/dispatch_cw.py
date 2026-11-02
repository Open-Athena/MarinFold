#!/usr/bin/env python
"""Dispatch independent single-H100 sequence-guidance shards on CoreWeave."""

import argparse
import base64
import dataclasses
import shlex
from pathlib import Path

from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

HERE = Path(__file__).resolve().parent
IMAGE = "vllm/vllm-openai:v0.9.2"
MODEL = "s3://marin-us-east-02a/MarinFold/exp304/model/exp277-step-266344"
OUT = "s3://marin-us-east-02a/MarinFold/exp321/null-sequence-guidance-v1"
MARINFOLD_REVISION = "359c3b72"
BATCH_PRIORITY = 3

assert "priority" in {item.name for item in dataclasses.fields(JobRequest)}


def encoded(path: Path) -> str:
    """Base64-encode one small source/input file for the pod bootstrap."""
    return base64.b64encode(path.read_bytes()).decode()


def bootstrap(shard: int, n_shards: int, args: argparse.Namespace) -> str:
    """Build a foreign-container command with no workspace sync."""
    files = {
        "guidance_worker_cw.py": HERE / "guidance_worker_cw.py",
        "guidance_policy.py": HERE / "guidance_policy.py",
        "stage_model.py": HERE / "stage_model.py",
        "targets.csv": HERE / "data" / "targets.csv",
    }
    transfers = "\n".join(
        f"echo {encoded(path)} | base64 -d > /tmp/exp321/{name}"
        for name, path in files.items()
    )
    flags = " --pure-ratio" if args.pure_ratio else ""
    if args.single_stream:
        flags += " --single-stream"
    if args.limit is not None:
        flags += f" --limit {args.limit}"
    return f"""
set -euo pipefail
echo "[exp321] shard {shard}/{n_shards} host=$(hostname)"
export FSSPEC_S3_CONFIG_KWARGS='{{"s3": {{"addressing_style": "virtual"}}}}'
mkdir -p /tmp/exp321
{transfers}
VLLM_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import torch, transformers" >/dev/null 2>&1; then VLLM_PY="$_py"; break; fi
done
if [ -z "$VLLM_PY" ]; then echo "FATAL: no python imports torch+transformers"; exit 3; fi
uv pip install --python "$VLLM_PY" --quiet fsspec s3fs boto3 pyarrow pandas "huggingface_hub>=0.30,<1.0"
"$VLLM_PY" /tmp/exp321/stage_model.py --source {shlex.quote(args.model)} --out /tmp/exp321-model
uv pip install --python "$VLLM_PY" --quiet --no-deps \
  "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@{MARINFOLD_REVISION}#subdirectory=marinfold"
export PYTHONPATH=/tmp/exp321:${{PYTHONPATH:-}}
exec "$VLLM_PY" /tmp/exp321/guidance_worker_cw.py \
  --model /tmp/exp321-model --targets /tmp/exp321/targets.csv \
  --out {shlex.quote(args.out)} --mode {shlex.quote(args.mode)} \
  --shard {shard}/{n_shards} --selection {shlex.quote(args.selection)} \
  --null {shlex.quote(args.null)} --scope {shlex.quote(args.scope)} \
  --gamma {args.gamma} --temperature {args.temperature} --top-p {args.top_p} \
  --n-rollouts {args.n_rollouts} --batch-size {args.batch_size}{flags}
""".strip()


def request(shard: int, n_shards: int, args: argparse.Namespace) -> JobRequest:
    """Create a root batch job that survives the local dispatcher exiting."""
    return JobRequest(
        name=f"exp321-{args.label}-s{shard}of{n_shards}",
        entrypoint=Entrypoint.from_binary("bash", ["-lc", bootstrap(shard, n_shards, args)]),
        resources=ResourceConfig.with_gpu(
            "H100", count=1, image=IMAGE, cpu=8, ram="64g", disk="128g"
        ),
        environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
        replicas=1,
        priority=BATCH_PRIORITY,
        processes_per_task=1,
        max_retries_failure=0 if args.limit is not None else 2,
        max_retries_preemption=100,
    )


def main() -> None:
    """Submit selected root shards through the Iris controller tunnel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shards", help="comma-separated shard IDs; default all")
    parser.add_argument("--label", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--selection",
        choices=["dev", "eval-holdout", "eval-all", "foldswitch-test"],
        required=True,
    )
    parser.add_argument("--null", choices=["polyala", "polylys", "shuffle"], required=True)
    parser.add_argument("--scope", choices=["positions", "all"], default="positions")
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--pure-ratio", action="store_true")
    parser.add_argument("--single-stream", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = (
        [int(item) for item in args.shards.split(",")]
        if args.shards else list(range(args.num_shards))
    )
    jobs = [request(shard, args.num_shards, args) for shard in selected]
    print(f"[exp321] {len(jobs)} H100 batch jobs; output={args.out}/{args.mode}")
    if args.dry_run:
        for job in jobs:
            print(f"{job.name} priority={job.priority} image={job.resources.image}")
        return
    with open_iris_client(cluster_name="cw-rno2a", workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for job in jobs:
            submitted = client.submit(job)
            print(
                f"[exp321] submitted {job.name} -> {getattr(submitted, 'id', submitted)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
