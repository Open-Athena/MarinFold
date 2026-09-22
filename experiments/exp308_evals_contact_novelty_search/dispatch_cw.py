#!/usr/bin/env python
"""Dispatch independent single-H100 contact novelty search shards."""

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
OUT = "s3://marin-us-east-02a/MarinFold/exp308/contact-novelty-v1"
MARINFOLD_REVISION = "b00a6ec0"
BATCH_PRIORITY = 3

assert "priority" in {item.name for item in dataclasses.fields(JobRequest)}


def encoded(path: Path) -> str:
    """Encode a small immutable source file for the pod bootstrap."""
    return base64.b64encode(path.read_bytes()).decode()


def bootstrap(shard: int, n_shards: int, args: argparse.Namespace) -> str:
    """Build the pod command without a workspace checkout or default uv sync."""
    files = {
        "novelty_worker_cw.py": HERE / "novelty_worker_cw.py",
        "novelty_policy.py": HERE / "novelty_policy.py",
        "stage_model.py": HERE / "stage_model.py",
        "targets.csv": HERE / "data" / "targets.csv",
    }
    transfers = "\n".join(
        f"echo {encoded(path)} | base64 -d > /tmp/exp308/{name}"
        for name, path in files.items()
    )
    extra = ""
    if args.limit is not None:
        extra += f" --limit {args.limit}"
    if args.max_statements is not None:
        extra += f" --max-statements {args.max_statements}"
    return f"""
set -euo pipefail
echo "[exp308] shard {shard}/{n_shards} host=$(hostname)"
export FSSPEC_S3_CONFIG_KWARGS='{{"s3": {{"addressing_style": "virtual"}}}}'
mkdir -p /tmp/exp308
{transfers}
VLLM_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import vllm" >/dev/null 2>&1; then VLLM_PY="$_py"; break; fi
done
if [ -z "$VLLM_PY" ]; then echo "FATAL: no python imports vllm"; exit 3; fi
uv pip install --python "$VLLM_PY" --quiet fsspec s3fs boto3 pyarrow pandas "huggingface_hub>=0.30,<1.0"
"$VLLM_PY" /tmp/exp308/stage_model.py --source {shlex.quote(args.model)} --out /tmp/exp308-model
uv pip install --python "$VLLM_PY" --quiet --no-deps \\
  "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@{MARINFOLD_REVISION}#subdirectory=marinfold"
export VLLM_PORT=$("$VLLM_PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
export PYTHONPATH=/tmp/exp308:${{PYTHONPATH:-}}
exec "$VLLM_PY" /tmp/exp308/novelty_worker_cw.py \\
    --model /tmp/exp308-model \\
    --targets /tmp/exp308/targets.csv \\
    --out {shlex.quote(args.out)} \\
    --shard {shard}/{n_shards} \\
    --selection {shlex.quote(args.selection)} \\
    --beam-width {args.beam_width} --epsilon {args.epsilon} \\
    --decay-contacts {args.decay_contacts} --wave-size {args.wave_size} \\
    --n-rollouts {args.n_rollouts}{extra}
""".strip()


def request(shard: int, n_shards: int, args: argparse.Namespace) -> JobRequest:
    """Create one root job that survives the workstation dispatcher exiting."""
    return JobRequest(
        name=f"exp308-{args.label}-s{shard}of{n_shards}",
        entrypoint=Entrypoint.from_binary("bash", ["-lc", bootstrap(shard, n_shards, args)]),
        resources=ResourceConfig.with_gpu("H100", count=1, image=IMAGE,
                                          cpu=8, ram="64g", disk="128g"),
        environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
        replicas=1, priority=BATCH_PRIORITY, processes_per_task=1,
        max_retries_failure=0 if args.limit is not None else 2,
        max_retries_preemption=100,
    )


def main() -> None:
    """Submit the requested shard IDs through the Iris controller tunnel."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shards", help="comma-separated shard IDs; default all")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--selection", choices=["pilot", "test", "all"], required=True)
    parser.add_argument("--beam-width", type=int, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--decay-contacts", type=int, required=True)
    parser.add_argument("--wave-size", type=int, default=10)
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-statements", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = ([int(item) for item in args.shards.split(",")]
                if args.shards else list(range(args.num_shards)))
    jobs = [request(shard, args.num_shards, args) for shard in selected]
    print(f"[exp308] {len(jobs)} H100 batch jobs; output={args.out}")
    if args.dry_run:
        for job in jobs:
            print(f"{job.name} priority={job.priority} image={job.resources.image}")
        return
    with open_iris_client(cluster_name="cw-rno2a", workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for job in jobs:
            submitted = client.submit(job)
            print(f"[exp308] submitted {job.name} -> {getattr(submitted, 'id', submitted)}",
                  flush=True)


if __name__ == "__main__":
    main()
