# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch the ESMFold2 self-consistency check to CoreWeave rno-2a.

One 1xH100 task (or a few) running `refold_worker_cw.py`. Same batch-band,
no-gang shape as `dispatch_redesign_cw.py`; the differences are the image
(ESMFold2 needs Python 3.12 and Biohub's `esm` package) and the fact that this
is a *sample*, not a pass over the corpus.

Sizing: exp78 measured ESMFold2 at 42.9 s/protein at L 250-300 with
`n_samples=5`. At 1 sample that is ~8.6 s, so 500 backbones x 8 designs =
4,000 refolds is ~9.5 GPU-hours — a couple of hours across a handful of the
idle prepaid GPUs. n=4,000 puts the 95% CI on the per-sequence rate inside
+/-1.6 %, which is far tighter than the decision needs.

    uv run python dispatch_refold_cw.py --shards 4 --backbones 500 --dry-run
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import os
from pathlib import Path

from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment

IRIS_PRIORITY_BAND_BATCH = 3

assert "priority" in {f.name for f in dataclasses.fields(JobRequest)}, (
    "This fray build lacks JobRequest.priority; batch-band dispatch requires the "
    "0.2.x.dev fray line. Submit from a fresh marin checkout."
)

# Biohub's `esm` package requires Python >=3.12,<3.13 (exp78's finding), which
# the pytorch 2.4.1 images do not ship — hence a CUDA base plus a python3.12.
IMAGE = os.environ.get("EXP266_REFOLD_IMAGE",
                       "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04")
MARINFOLD_GIT = os.environ.get(
    "EXP266_CW_MARINFOLD",
    "marinfold @ git+https://github.com/Open-Athena/MarinFold.git#subdirectory=marinfold",
)

S3_PREFIX = os.environ.get("EXP266_CW_S3_PREFIX", "s3://marin-us-east-02a/MarinFold/exp266")
DOCUMENTS_GLOB = os.environ.get("EXP266_REFOLD_DOCS", f"{S3_PREFIX}/documents/*.parquet")
BACKBONES_GLOB = os.environ.get("EXP266_REFOLD_BACKBONES", f"{S3_PREFIX}/backbones/*.parquet")
OUT_PREFIX = os.environ.get("EXP266_REFOLD_OUT", f"{S3_PREFIX}/refold")
JOB_PREFIX = os.environ.get("EXP266_REFOLD_JOB_PREFIX", "exp266-refold")

WORK_DIR = "/tmp/exp266"
WORKER_FILES = ("backbone.py", "selfconsistency.py", "refold_worker_cw.py")

FSSPEC_VIRTUAL_ADDRESSING_EXPORT = (
    """export FSSPEC_S3_CONFIG_KWARGS='{"s3": {"addressing_style": "virtual"}}'"""
)


def _encoded_sources() -> str:
    here = Path(__file__).resolve().parent
    return "\n".join(
        f'echo {base64.b64encode((here / n).read_bytes()).decode()} | base64 -d > {WORK_DIR}/{n}'
        for n in WORKER_FILES
    )


def build_bootstrap(*, shard_i: int, num_shards: int, backbones: int, seed: int) -> str:
    return f"""
set -euo pipefail
echo "[exp266-refold] host=$(hostname) shard={shard_i}/{num_shards} image={IMAGE}"
nvidia-smi -L || true

{FSSPEC_VIRTUAL_ADDRESSING_EXPORT}

apt-get update -qq
apt-get install -y -qq --no-install-recommends git python3.12 python3.12-venv curl
PY=/opt/venv/bin/python
python3.12 -m venv /opt/venv
$PY -m pip install --quiet --upgrade pip

mkdir -p {WORK_DIR}
{_encoded_sources()}

$PY -m pip install --quiet torch
# Biohub's `esm` (NOT the unrelated PyPI `esm`) registers ESMFold2 with
# transformers; no PyPI release, so install from git. exp78's recipe.
$PY -m pip install --quiet "esm @ git+https://github.com/Biohub/esm.git@main" \\
    "transformers>=4.40" accelerate "huggingface_hub[hf_transfer]" \\
    gemmi numpy fsspec s3fs boto3 pyarrow
$PY -m pip install --quiet --no-deps "{MARINFOLD_GIT}"

export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH={WORK_DIR}:${{PYTHONPATH:-}}
exec $PY {WORK_DIR}/refold_worker_cw.py \\
    --documents-glob "{DOCUMENTS_GLOB}" \\
    --backbones-glob "{BACKBONES_GLOB}" \\
    --out "{OUT_PREFIX}/refold-{shard_i:03d}-of-{num_shards:03d}.parquet" \\
    --backbones {backbones} \\
    --seed {seed + shard_i}
""".strip()


def build_request(*, shard_i: int, num_shards: int, backbones: int, seed: int,
                  cpu: int, ram: str, disk: str, priority: int) -> JobRequest:
    return JobRequest(
        name=f"{JOB_PREFIX}-s{shard_i}of{num_shards}",
        entrypoint=Entrypoint.from_binary(
            "bash", ["-lc", build_bootstrap(shard_i=shard_i, num_shards=num_shards,
                                            backbones=backbones, seed=seed)]),
        resources=ResourceConfig.with_gpu("H100", count=1, image=IMAGE,
                                          cpu=cpu, ram=ram, disk=disk),
        environment=create_environment(docker_image=IMAGE, env_vars={}, setup_scripts=[]),
        replicas=1,
        priority=priority,
        processes_per_task=1,
        max_retries_failure=3,
        max_retries_preemption=100,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=4,
                    help="Tasks; each draws its own disjoint-ish backbone sample "
                         "via --seed offset.")
    ap.add_argument("--backbones", type=int, default=125,
                    help="Backbones sampled PER TASK (x8 designs each).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", type=int, default=8)
    ap.add_argument("--ram", default="64g")
    ap.add_argument("--disk", default="128g")
    ap.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    ap.add_argument("--cluster", default="cw-rno2a")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    priority = IRIS_PRIORITY_BAND_BATCH if args.priority == "batch" else 0
    reqs = [build_request(shard_i=i, num_shards=args.shards, backbones=args.backbones,
                          seed=args.seed, cpu=args.cpu, ram=args.ram, disk=args.disk,
                          priority=priority)
            for i in range(args.shards)]

    if args.dry_run:
        print(f"[exp266-refold] DRY RUN — {len(reqs)} JobRequests "
              f"({args.shards * args.backbones} backbones, "
              f"{args.shards * args.backbones * 8} refolds)")
        print(reqs[0].entrypoint.binary_entrypoint.args[1])
        return

    from fray.iris_backend import FrayIrisClient
    from iris.cli.connect import open_iris_client

    with open_iris_client(cluster_name=args.cluster, workspace=None) as iris_client:
        client = FrayIrisClient.from_iris_client(iris_client)
        for req in reqs:
            client.submit(req)
            print(f"  submitted {req.name}")


if __name__ == "__main__":
    raise SystemExit(main())
