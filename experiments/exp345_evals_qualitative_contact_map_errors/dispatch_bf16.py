"""Submit the small in-region CPU conversion, with no GPU reservation."""

import base64
import json
import shlex
from datetime import UTC, datetime
from pathlib import Path

from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

HERE = Path(__file__).resolve().parent


def main() -> None:
    """Preserve the CPU job identity alongside the inference attempts."""
    lines = ["set -euo pipefail", "mkdir -p /tmp/exp345-cast"]
    for name, path in {"stage_bf16.py": HERE / "stage_bf16.py", "plan.json": HERE / "data/whole_map_plan.json"}.items():
        lines.append(f"printf %s {shlex.quote(base64.b64encode(path.read_bytes()).decode())} | base64 -d > /tmp/exp345-cast/{name}")
    lines.extend(['uv pip install --system --quiet "fsspec==2026.1.0" "s3fs==2026.1.0" "aiobotocore==2.26.0"',
                  'python3 /tmp/exp345-cast/stage_bf16.py'])
    image = "vllm/vllm-openai:v0.9.2"
    request = JobRequest(name="exp345-bf16-inregion-a02", entrypoint=Entrypoint.from_binary("bash",["-lc","\n".join(lines)]),
                         resources=ResourceConfig.with_cpu(cpu=4, ram="32g", disk="32g", image=image, target_cluster="cw-us-east-02a"),
                         environment=create_environment(docker_image=image,setup_scripts=[],env_vars={}), replicas=1,
                         priority=3,max_retries_failure=0,max_retries_preemption=0)
    with open_iris_client(cluster_name="marin",workspace=None) as iris:
        job = FrayIrisClient.from_iris_client(iris).submit(request,adopt_existing=False)
        record = {"job_id":job.job_id,"timestamp_utc":datetime.now(UTC).isoformat(),"cluster":"cw-us-east-02a","gpu_count":0}
        path = HERE / "data/whole_map_conversion_jobs.json"
        records = json.loads(path.read_text()) if path.exists() else []
        records.append(record)
        path.write_text(json.dumps(records,indent=2)+"\n")
        print(record,flush=True)


if __name__ == "__main__":
    main()
