"""Dispatch the full Helico decoy-ranking run to CoreWeave H100s.

Jobs are submitted directly from the workstation as independent Iris root jobs,
so they survive this launcher exiting. Each job owns a balanced set of
fingerprinted result parts and resumes already-valid S3 outputs.
"""

import argparse
import base64
import dataclasses
import shlex
from pathlib import Path

from fray.types import (
    Entrypoint,
    JobRequest,
    JobStatus,
    ResourceConfig,
    create_environment,
)

from full_worker_cw import (
    CUEQUIVARIANCE_VERSION,
    PYTHON_PACKAGE_VERSIONS,
    RUN_FINGERPRINT,
    TORCH_VERSION,
)

IRIS_PRIORITY_BAND_BATCH = 3
assert "priority" in {field.name for field in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; use the current marin-freshiris venv"
)

IMAGE = "pytorch/pytorch:2.13.0-cuda13.0-cudnn9-runtime"
JOB_PREFIX = "exp335-helico-full"
WORKER = Path(__file__).with_name("full_worker_cw.py")
WORK_DIR = "/tmp/exp335_dispatch"
WORKER_LOCAL = f"{WORK_DIR}/full_worker_cw.py"

# Complete Python closure resolved for the successful CPython 3.13 foreign-image
# bootstrap. Every package is named explicitly because pip dependency resolution
# in a vendor image can otherwise repin libraries underneath its bundled torch.
WORKER_BOOTSTRAP_PACKAGE_VERSIONS = {
    "aiobotocore": "3.9.2",
    "aiohappyeyeballs": "2.7.1",
    "aiohttp": "3.14.4",
    "aioitertools": "0.13.0",
    "aiosignal": "1.4.0",
    "annotated-doc": "0.0.5",
    "anyio": "4.15.1",
    "attrs": "26.1.0",
    "biopython": "1.86",
    "botocore": "1.43.106",
    "certifi": "2026.7.22",
    "charset-normalizer": "3.5.2",
    "cuequivariance": "0.12.0",
    "cuequivariance-ops-cu12": "0.8.1",
    "cuequivariance-ops-torch-cu12": "0.8.1",
    "cuequivariance-torch": "0.8.1",
    "filelock": "4.0.12",
    "frozenlist": "1.8.0",
    "fsspec": "2026.9.0",
    "h11": "0.16.0",
    "hf-xet": "1.7.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface-hub": "1.5.0",
    "idna": "3.20",
    "jmespath": "1.1.0",
    "markdown-it-py": "4.2.0",
    "mdurl": "0.1.2",
    "mpmath": "1.3.0",
    "multidict": "6.9.1",
    "networkx": "3.7",
    "numpy": "2.4.4",
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cuda-nvrtc-cu12": "12.9.86",
    "nvidia-ml-py": "13.615.71",
    "opt-einsum": "3.4.0",
    "packaging": "26.3",
    "platformdirs": "4.12.4",
    "propcache": "0.5.4",
    "pygments": "2.21.0",
    "python-dateutil": "2.9.0.post0",
    "pyyaml": "6.0.3",
    "requests": "2.33.1",
    "rich": "15.0.0",
    "s3fs": "2026.9.0",
    "scipy": "1.17.0",
    "shellingham": "1.5.4",
    "six": "1.17.0",
    "sympy": "1.14.0",
    "tmtools": "0.3.0",
    "tqdm": "4.67.3",
    "typer": "0.27.3",
    "typing-extensions": "4.16.0",
    "urllib3": "2.8.0",
    "wrapt": "2.5.0",
    "yarl": "1.25.1",
}
assert all(
    WORKER_BOOTSTRAP_PACKAGE_VERSIONS[name] == version
    for name, version in PYTHON_PACKAGE_VERSIONS.items()
)
assert (
    WORKER_BOOTSTRAP_PACKAGE_VERSIONS["cuequivariance-torch"] == CUEQUIVARIANCE_VERSION
)


def pinned_python_requirements() -> str:
    """Return the complete pinned worker dependency closure."""
    return " ".join(
        shlex.quote(f"{name}=={version}")
        for name, version in WORKER_BOOTSTRAP_PACKAGE_VERSIONS.items()
    )


def build_bootstrap(
    *,
    shard: int,
    num_shards: int,
    limit_parts: int,
    target: str | None,
    worker_bytes: bytes,
) -> str:
    """Build a self-contained worker bootstrap for a foreign CUDA image."""
    worker_b64 = base64.b64encode(worker_bytes).decode()
    requirements = pinned_python_requirements()
    limit_arg = f" --limit-parts {limit_parts}" if limit_parts else ""
    target_arg = f" --target {shlex.quote(target)}" if target else ""
    return f"""
set -euo pipefail
export EXP335_BOOTSTRAP_STARTED=$(date +%s.%N)
echo "[exp335] host=$(hostname) shard={shard}/{num_shards} image={IMAGE}"
nvidia-smi -L
mkdir -p {WORK_DIR}
echo {worker_b64} | base64 -d > {WORKER_LOCAL}

PY=""
for _py in /opt/conda/bin/python /usr/local/bin/python /usr/bin/python3 python3 python; do
  if "$_py" -c "import torch" >/dev/null 2>&1; then PY="$_py"; break; fi
done
if [ -z "$PY" ]; then echo "[exp335] FATAL: no Python imports torch"; exit 3; fi
echo "[exp335] python=$PY torch=$($PY -c 'import torch; print(torch.__version__)')"

"$PY" -m pip install --quiet --no-cache-dir --no-deps --break-system-packages \
  {requirements}

exec "$PY" {WORKER_LOCAL} --shard {shard} --num-shards {num_shards}{target_arg}{limit_arg}
""".strip()


def build_request(
    *,
    shard: int,
    num_shards: int,
    limit_parts: int,
    target: str | None,
    name_suffix: str,
    worker_bytes: bytes,
) -> JobRequest:
    """Build one preemptible batch-priority single-H100 request."""
    resources = ResourceConfig.with_gpu(
        "H100", count=1, image=IMAGE, cpu=8, ram="64g", disk="64g"
    )
    return JobRequest(
        name=f"{JOB_PREFIX}-s{shard:03d}-of-{num_shards:03d}{name_suffix}",
        entrypoint=Entrypoint.from_binary(
            "bash",
            [
                "-lc",
                build_bootstrap(
                    shard=shard,
                    num_shards=num_shards,
                    limit_parts=limit_parts,
                    target=target,
                    worker_bytes=worker_bytes,
                ),
            ],
        ),
        resources=resources,
        environment=create_environment(
            docker_image=IMAGE, env_vars={}, setup_scripts=[]
        ),
        replicas=1,
        priority=IRIS_PRIORITY_BAND_BATCH,
        processes_per_task=1,
        max_retries_failure=2,
        max_retries_preemption=100,
        max_task_failures=2,
    )


def parse_shards(spec: str | None, num_shards: int) -> list[int]:
    """Parse and validate a comma-separated shard subset."""
    shards = (
        list(range(num_shards))
        if spec is None
        else [int(item) for item in spec.split(",")]
    )
    if not shards or len(shards) != len(set(shards)):
        raise ValueError("shard selection must be non-empty and unique")
    if any(shard < 0 or shard >= num_shards for shard in shards):
        raise ValueError(f"shards must lie in [0, {num_shards})")
    return shards


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=96)
    parser.add_argument("--shards", default=None)
    parser.add_argument("--limit-parts", type=int, default=0)
    parser.add_argument("--target", default=None)
    parser.add_argument("--name-suffix", default="")
    parser.add_argument("--cluster", default="cw-rno2a")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="return after submitting root jobs; they continue independently",
    )
    return parser.parse_args()


def main() -> None:
    """Submit selected shards and optionally wait for every terminal state."""
    args = parse_args()
    shards = parse_shards(args.shards, args.num_shards)
    worker_bytes = WORKER.read_bytes()
    requests = [
        build_request(
            shard=shard,
            num_shards=args.num_shards,
            limit_parts=args.limit_parts,
            target=args.target,
            name_suffix=args.name_suffix,
            worker_bytes=worker_bytes,
        )
        for shard in shards
    ]
    print(
        f"[exp335] {len(requests)} root job(s), 1xH100 batch band; "
        f"shards={shards[0]}..{shards[-1]} of {args.num_shards}; "
        f"target={args.target}; limit_parts={args.limit_parts}; "
        f"fingerprint={RUN_FINGERPRINT}; "
        f"torch={TORCH_VERSION}; image={IMAGE}"
    )
    if args.dry_run:
        for request in requests[:3]:
            print(
                f"  {request.name}: priority={request.priority} "
                f"gpu={request.resources.device.variant}x{request.resources.device.count}"
            )
        return

    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    with open_iris_client(cluster_name=args.cluster, workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        jobs = []
        for request in requests:
            job = client.submit(request)
            jobs.append((request, job))
            print(f"[exp335] submitted {request.name}: {job.job_id}", flush=True)
        if args.no_wait:
            print("[exp335] --no-wait: root jobs remain active after launcher exit")
            return
        outcomes = []
        for request, job in jobs:
            try:
                status = job.wait()
            except Exception as error:  # noqa: BLE001 - report every shard
                status = f"{type(error).__name__}: {error}"
            outcomes.append((request.name, status))
            print(f"[exp335] {request.name}: {status}", flush=True)
        failed = [name for name, status in outcomes if status != JobStatus.SUCCEEDED]
        if failed:
            raise RuntimeError(f"{len(failed)}/{len(outcomes)} shards failed: {failed}")


if __name__ == "__main__":
    main()
