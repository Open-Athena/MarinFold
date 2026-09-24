#!/usr/bin/env python
"""Start a short-lived Iris CPU pod to retrieve exp304 S3 results."""

import argparse

from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

IMAGE = "python:3.12-slim"
DEFAULT_SOURCE = "s3://marin-us-east-02a/MarinFold/exp304/blind-search-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--expected", type=int, default=134)
    parser.add_argument("--label", default="full-v1")
    args = parser.parse_args()
    script = f"""
set -euo pipefail
python -m pip install --quiet fsspec s3fs
export FSSPEC_S3_CONFIG_KWARGS='{{"s3": {{"addressing_style": "virtual"}}}}'
python - <<'PY'
import fsspec
import os
import shutil
import time
from pathlib import Path

source = {args.source!r}
expected = {args.expected}
fs, path = fsspec.core.url_to_fs(source)
out = Path('/tmp/exp304-readback')
out.mkdir(exist_ok=True)
deadline = time.monotonic() + 3600
while True:
    names = [name for name in fs.find(path) if name.endswith('.parquet')]
    print(f'[exp304] found {{len(names)}}/{{expected}} parquet files', flush=True)
    if len(names) >= expected:
        break
    if time.monotonic() >= deadline:
        raise TimeoutError(f'found {{len(names)}} of {{expected}} files')
    time.sleep(30)
for name in names:
    fs.get_file(name, str(out / os.path.basename(name)))
shutil.make_archive('/tmp/exp304-readback', 'gztar', out)
print(f'[exp304] ready /tmp/exp304-readback.tar.gz with {{len(names)}} files', flush=True)
time.sleep(3600)
PY
""".strip()
    job = JobRequest(
        name=f"exp304-readback-{args.label}",
        entrypoint=Entrypoint.from_binary("bash", ["-lc", script]),
        resources=ResourceConfig.with_cpu(cpu=2, ram="8g", disk="16g", image=IMAGE),
        environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
        replicas=1,
        priority=3,
        processes_per_task=1,
        max_retries_failure=0,
        max_retries_preemption=10,
    )
    with open_iris_client(cluster_name="cw-rno2a", workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        handle = client.submit(job)
        print(f"submitted {job.name}: {handle}")


if __name__ == "__main__":
    main()
