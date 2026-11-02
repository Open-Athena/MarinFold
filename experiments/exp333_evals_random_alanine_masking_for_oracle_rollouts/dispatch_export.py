"""Submit the S3-to-HF export as a small CoreWeave CPU root job."""

import argparse
import base64
import dataclasses
import shlex
from pathlib import Path

from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from huggingface_hub import get_token
from iris.cli.connect import open_iris_client

HERE = Path(__file__).resolve().parent
IMAGE = "vllm/vllm-openai:v0.9.2"
BATCH_PRIORITY = 3

assert "priority" in {item.name for item in dataclasses.fields(JobRequest)}


def request(args: argparse.Namespace, token: str) -> JobRequest:
    """Build one root export job with injected S3 and HF credentials."""
    source = base64.b64encode((HERE / "export_cw_results.py").read_bytes()).decode()
    command = f"""
set -euo pipefail
echo {source} | base64 -d > /tmp/export_cw_results.py
PY=""
for _py in /app/.venv/bin/python /usr/local/bin/python /usr/bin/python3 python3 python; do
  if "$_py" -c "import sys" >/dev/null 2>&1; then PY="$_py"; break; fi
done
uv pip install --python "$PY" --quiet "huggingface_hub>=1.5" "fsspec==2026.1.0" \
  "s3fs==2026.1.0" "aiobotocore==2.26.0"
exec "$PY" /tmp/export_cw_results.py --source {shlex.quote(args.source)} \
  --destination {shlex.quote(args.destination)}
""".strip()
    return JobRequest(
        name=f"exp333-export-{args.label}",
        entrypoint=Entrypoint.from_binary("bash", ["-lc", command]),
        resources=ResourceConfig.with_cpu(image=IMAGE, cpu=4, ram="16g", disk="32g"),
        environment=create_environment(
            docker_image=IMAGE,
            env_vars={"HF_TOKEN": token},
            setup_scripts=[],
        ),
        replicas=1,
        priority=BATCH_PRIORITY,
        processes_per_task=1,
        max_retries_failure=0,
        max_retries_preemption=100,
    )


def main() -> None:
    """Submit a result export without exposing the stored HF token."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    token = get_token()
    if not token:
        raise ValueError("no Hugging Face token is configured")
    job = request(args, token)
    if args.dry_run:
        print(f"[exp333] {job.name} priority={job.priority} image={job.resources.image}")
        return
    with open_iris_client(cluster_name="cw-rno2a", workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        submitted = client.submit(job)
        print(f"[exp333] submitted {job.name} -> {submitted.job_id}", flush=True)


if __name__ == "__main__":
    main()
