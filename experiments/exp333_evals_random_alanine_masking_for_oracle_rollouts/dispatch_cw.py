"""Dispatch independent alanine-masking shards on CoreWeave H100s."""

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
OUT = "s3://marin-us-east-02a/MarinFold/exp333/alanine-masking-v1/main"
MARINFOLD_REVISION = "4c985f4f"
BATCH_PRIORITY = 3
TARGETS = (
    HERE.parent
    / "exp321_evals_null_sequence_contrastive_guidance_for_contacts"
    / "data"
    / "targets.csv"
)
STAGE_MODEL = (
    HERE.parent
    / "exp321_evals_null_sequence_contrastive_guidance_for_contacts"
    / "stage_model.py"
)

assert "priority" in {item.name for item in dataclasses.fields(JobRequest)}


def encoded(path: Path) -> str:
    """Base64-encode one small source/input file for the pod bootstrap."""
    return base64.b64encode(path.read_bytes()).decode()


def bootstrap(shard: int, n_shards: int, args: argparse.Namespace) -> str:
    """Build a foreign-container command with no workspace sync."""
    files = {
        "run_rollouts.py": HERE / "run_rollouts.py",
        "masking_policy.py": HERE / "masking_policy.py",
        "stage_model.py": STAGE_MODEL,
        "targets.csv": TARGETS,
    }
    transfers = "\n".join(
        f"echo {encoded(path)} | base64 -d > /tmp/exp333/{name}"
        for name, path in files.items()
    )
    limit = f" --limit {args.limit}" if args.limit is not None else ""
    return f"""
set -euo pipefail
echo "[exp333] shard {shard}/{n_shards} host=$(hostname)"
export FSSPEC_S3_CONFIG_KWARGS='{{"s3": {{"addressing_style": "virtual"}}}}'
mkdir -p /tmp/exp333
{transfers}
VLLM_PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do
  if "$_py" -c "import torch, vllm" >/dev/null 2>&1; then VLLM_PY="$_py"; break; fi
done
if [ -z "$VLLM_PY" ]; then echo "FATAL: no python imports torch+vllm"; exit 3; fi
uv pip install --python "$VLLM_PY" --quiet fsspec s3fs boto3 pyarrow pandas "huggingface_hub>=0.30,<1.0"
"$VLLM_PY" /tmp/exp333/stage_model.py --source {shlex.quote(args.model)} --out /tmp/exp333-model
uv pip install --python "$VLLM_PY" --quiet --no-deps \
  "marinfold @ git+https://github.com/Open-Athena/MarinFold.git@{MARINFOLD_REVISION}#subdirectory=marinfold"
export PYTHONPATH=/tmp/exp333:${{PYTHONPATH:-}}
exec "$VLLM_PY" /tmp/exp333/run_rollouts.py \
  --model /tmp/exp333-model --targets /tmp/exp333/targets.csv \
  --out {shlex.quote(args.out)} --shard {shard}/{n_shards} \
  --selection {shlex.quote(args.selection)} --fractions {shlex.quote(args.fractions)} \
  --n-rollouts {args.n_rollouts} --max-num-seqs {args.max_num_seqs}{limit}
""".strip()


def request(shard: int, n_shards: int, args: argparse.Namespace) -> JobRequest:
    """Create a root batch job that survives the local dispatcher exiting."""
    return JobRequest(
        name=f"exp333-{args.label}-s{shard}of{n_shards}",
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
    parser.add_argument("--selection", choices=["dev", "heldout", "all"], required=True)
    parser.add_argument("--fractions", default="0,0.05,0.10,0.20,0.40,1.0")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = (
        [int(item) for item in args.shards.split(",")]
        if args.shards
        else list(range(args.num_shards))
    )
    jobs = [request(shard, args.num_shards, args) for shard in selected]
    print(f"[exp333] {len(jobs)} H100 batch jobs; output={args.out}")
    if args.dry_run:
        for job in jobs:
            print(f"{job.name} priority={job.priority} image={job.resources.image}")
        return
    with open_iris_client(cluster_name="cw-rno2a", workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for job in jobs:
            submitted = client.submit(job)
            print(
                f"[exp333] submitted {job.name} -> {submitted.job_id}",
                flush=True,
            )


if __name__ == "__main__":
    main()
