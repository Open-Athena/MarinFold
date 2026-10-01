"""Dispatch checkpoint-local root jobs using Marin's installed Iris runtime.

Run with ``uv run --project /path/to/marin --no-sync python /path/to/this.py``.
A smoke run must finish before a production submission is allowed.
"""

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import iris
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

HERE = Path(__file__).resolve().parent
REFERENCE = HERE.parent / "exp277_models_single_mpnn_pilot/evals/2026-09-13_rollout_v2/score_rollout_worker.py"
IMAGE = "vllm/vllm-openai:v0.9.2"


def command(shard: int, args: argparse.Namespace, plan: dict) -> str:
    """Bootstrap only pinned code, the model tokenizer, and eval-val sequences."""
    files = {"worker.py": HERE / "whole_map_worker.py", "reference_worker.py": REFERENCE,
             "validation.py": HERE / "fetch_whole_map.py",
             "targets.json": HERE / "data/whole_map_targets.json", "plan.json": args.model_plan}
    lines = ["set -euo pipefail", "mkdir -p /tmp/exp345"]
    for name, path in files.items():
        encoded = base64.b64encode(path.read_bytes()).decode()
        lines.append(f"printf %s {shlex.quote(encoded)} | base64 -d > /tmp/exp345/{name}")
    lines.extend([
        "VLLM_PY=''",
        "for candidate in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 /opt/venv/bin/python python3 python; do if \"$candidate\" -c 'import vllm' >/dev/null 2>&1; then VLLM_PY=\"$candidate\"; break; fi; done",
        'test -n "$VLLM_PY"',
        "export VLLM_PORT=$(\"$VLLM_PY\" -c 'import socket; s=socket.socket(); s.bind((\"\",0)); print(s.getsockname()[1]); s.close()')",
        'uv pip install --python "$VLLM_PY" --quiet "fsspec==2026.1.0" "s3fs==2026.1.0" "aiobotocore==2.26.0" "pyarrow>=23,<24"',
        'uv pip install --python "$VLLM_PY" --quiet --no-deps "pandas==2.2.3" "python-dateutil==2.9.0.post0" "pytz==2025.2" "tzdata==2025.2" "six==1.17.0"',
        'uv pip install --python "$VLLM_PY" --quiet --no-deps ' + shlex.quote(
            f"marinfold @ git+https://github.com/Open-Athena/MarinFold.git@{plan['marinfold_revision']}#subdirectory=marinfold"),
        'exec "$VLLM_PY" /tmp/exp345/worker.py ' + f"--shard {shard} --num-shards {args.num_shards}" + (" --smoke" if args.smoke else "") + (" --smoke-first" if args.smoke_first else ""),
    ])
    return "\n".join(lines)


def main() -> None:
    """Submit a smoke or production shard set, saving all job IDs immediately."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-first", action="store_true", help="Validate the first protein in-process before continuing; one model load")
    parser.add_argument("--target-cluster", default="cw-us-east-02a")
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shards", help="Comma-separated shard IDs for a targeted recovery")
    parser.add_argument("--attempt", default="a01")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-plan", type=Path, default=HERE / "data/whole_map_plan.json")
    args = parser.parse_args()
    plan = json.loads(args.model_plan.read_text())
    if args.smoke_first and (args.num_shards != 1 or args.smoke):
        raise ValueError("--smoke-first requires one production shard")
    if not args.smoke and not args.smoke_first and not (HERE / "data/whole_map_smoke_validation.json").exists():
        raise ValueError("Validate the smoke raw maps before production")
    shards = [0] if args.smoke else ([int(x) for x in args.shards.split(",")] if args.shards else list(range(args.num_shards)))
    requests = [JobRequest(
        name=f"exp345-whole-{'smoke' if args.smoke else 'full'}-{args.attempt}-s{shard}",
        entrypoint=Entrypoint.from_binary("bash", ["-lc", command(shard, args, plan)]),
        resources=ResourceConfig.with_gpu("H100", count=1, cpu=8, ram="64g", disk="128g", image=IMAGE, target_cluster=args.target_cluster),
        environment=create_environment(docker_image=IMAGE, setup_scripts=[], env_vars={}),
        replicas=1, priority=3, max_retries_failure=0, max_retries_preemption=100,
    ) for shard in shards]
    if args.dry_run:
        print(f"Validated {len(requests)} independent H100 batch requests in {args.target_cluster}")
        return
    log_path = HERE / "data/whole_map_jobs.json"
    history = json.loads(log_path.read_text()) if log_path.exists() else []
    with open_iris_client(cluster_name="marin", workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for request in requests:
            job = client.submit(request, adopt_existing=False)
            record = {"job_id": job.job_id, "name": request.name,
                      "timestamp_utc": datetime.now(UTC).isoformat(),
                      "worker_sha256": hashlib.sha256((HERE / "whole_map_worker.py").read_bytes()).hexdigest(),
                      "validation_sha256": hashlib.sha256((HERE / "fetch_whole_map.py").read_bytes()).hexdigest(),
                      "dispatch_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      "model_plan": args.model_plan.name,
                      "model_plan_sha256": hashlib.sha256(args.model_plan.read_bytes()).hexdigest(),
                      "target_cluster": args.target_cluster, "smoke": args.smoke, "smoke_first": args.smoke_first, "attempt": args.attempt,
                      "marin_runtime_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(iris.__file__).resolve().parent, text=True).strip()}
            history.append(record)
            log_path.write_text(json.dumps(history, indent=2) + "\n")
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
