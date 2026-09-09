"""Submit a bounded, independent H100 worker through Iris's controller."""

import argparse
import base64
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import fsspec
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

IMAGE = "pytorch/pytorch@sha256:0a3b9fedefe1f61ac4d5a9de9015c0863db27ca0fde2d4e37e6268147980b726"
HERE = Path(__file__).resolve().parent


def bundle() -> bytes:
    """Bundle the experiment and current document library as a small archive."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(HERE.glob("*.py")) + [HERE / "bootstrap.sh"]:
            archive.add(path, arcname=path.name)
        library = HERE.parents[1] / "marinfold" / "marinfold"
        for path in sorted(library.rglob("*.py")):
            archive.add(
                path,
                arcname=str(Path("marinfold/marinfold") / path.relative_to(library)),
            )
    return buffer.getvalue()


def storage_filesystem(cluster: str):
    """Build a workstation S3 client without writing credentials to disk or logs."""
    contexts = {
        "cw-rno2a": "marin-rn02a_RNO2A",
        "cw-us-east-02a": "marin-gpu_US-EAST-02A",
    }
    result = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(Path.home() / ".kube/coreweave-iris"),
            "--context",
            contexts[cluster],
            "-n",
            "iris",
            "get",
            "secret",
            "iris-task-env",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    secret = json.loads(result.stdout)["data"]
    config = json.loads(base64.b64decode(secret["FSSPEC_S3"]))
    config["endpoint_url"] = "https://cwobject.com"
    config["key"] = base64.b64decode(secret["AWS_ACCESS_KEY_ID"]).decode()
    config["secret"] = base64.b64decode(secret["AWS_SECRET_ACCESS_KEY"]).decode()
    return fsspec.filesystem("s3", **config)


def stage_bundle(content: bytes, cluster: str) -> str:
    """Upload a content-addressed bundle using the cluster's storage credentials."""
    fs = storage_filesystem(cluster)
    digest = hashlib.sha256(content).hexdigest()
    path = f"marin-us-east-02a/MarinFold/exp278-proteina/code/{digest}.tar.gz"
    fs.pipe_file(path, content)
    return "s3://" + path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster", default="cw-rno2a")
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--checkpoints", default="short")
    parser.add_argument("--no-wait", action="store_true", required=True)
    parser.add_argument("worker_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    content = bundle()
    bundle_uri = stage_bundle(content, args.cluster)
    unpack = (
        "import fsspec,hashlib,io,tarfile; "
        f"data=fsspec.open({bundle_uri!r},'rb').open().read(); "
        f"assert hashlib.sha256(data).hexdigest()=={hashlib.sha256(content).hexdigest()!r}; "
        "tarfile.open(fileobj=io.BytesIO(data)).extractall('/tmp/exp278',filter='data')"
    )
    worker_args = args.worker_args
    if worker_args and worker_args[0] == "--":
        worker_args = worker_args[1:]
    if not worker_args:
        parser.error("Provide a worker script after --")
    command = (
        "set -euo pipefail\nmkdir -p /tmp/exp278\n"
        "/opt/conda/bin/python -m pip install --quiet uv==0.8.22\n"
        "uv pip install --python /opt/conda/bin/python fsspec==2025.3.0 s3fs==2025.3.0\n"
        f"uv run --no-project /opt/conda/bin/python -c {shlex.quote(unpack)}\n"
        f"exec timeout {args.timeout}s bash /tmp/exp278/bootstrap.sh "
        + shlex.join(["/tmp/exp278/" + worker_args[0], *worker_args[1:]])
    )
    request = JobRequest(
        name=args.name,
        entrypoint=Entrypoint.from_binary("bash", ["-lc", command]),
        resources=ResourceConfig.with_gpu(
            "H100", count=1, image=IMAGE, cpu=8, ram="64g", disk="64g"
        ),
        environment=create_environment(
            docker_image=IMAGE,
            env_vars={"PROTEINA_CHECKPOINTS": args.checkpoints},
            setup_scripts=[],
        ),
        replicas=1,
        processes_per_task=1,
        priority=3,
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    with open_iris_client(cluster_name=args.cluster, workspace=None) as iris_client:
        job = FrayIrisClient.from_iris_client(iris_client).submit(request)
        print(f"Job submitted: {job.job_id}", flush=True)
    state = {
        "job_id": job.job_id,
        "name": args.name,
        "cluster": args.cluster,
        "image": IMAGE,
        "bundle_uri": bundle_uri,
        "timeout_seconds": args.timeout,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "worker_args": worker_args,
    }
    destination = HERE / "data" / f"{args.name}-submission.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
