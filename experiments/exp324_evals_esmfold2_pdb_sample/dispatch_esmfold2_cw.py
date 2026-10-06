# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch ESMFold2 pilot shards to CoreWeave through Iris.

This uses `iris job run` root submissions rather than Fray children, so each
shard is an independent root job that survives the launcher process.

Example:
    uv run python experiments/exp324_evals_esmfold2_pdb_sample/dispatch_esmfold2_cw.py \
      --manifest experiments/exp324_evals_esmfold2_pdb_sample/data/sample_100_manifest.csv \
      --shards 4 --n-samples 1 --dry-run
"""

import argparse
import shutil
import shlex
import subprocess
from pathlib import Path

IMAGE = "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04"
DEFAULT_OUT_PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/pilot_100_n1"
IRIS = [
    "uv",
    "run",
    "--project",
    "/tmp/marin-iris-origin-main-fresh/lib/iris",
    "iris",
    "--cluster=marin",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--num-loops", type=int, default=20)
    parser.add_argument("--num-sampling-steps", type=int, default=100)
    parser.add_argument("--limit-per-shard", type=int)
    parser.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--job-prefix", default="exp324-esmfold2")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bundle-dir", type=Path, default=Path("/tmp/exp324_esmfold2_bundle"))
    return parser.parse_args()


def stage_bundle(bundle_dir: Path, manifest: Path) -> Path:
    """Create a tiny Iris workspace bundle with just the worker and manifest."""
    exp_rel = Path("experiments/exp324_evals_esmfold2_pdb_sample")
    target_dir = bundle_dir / exp_rel
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    target_dir.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    shutil.copy2(here / "esmfold2_worker_cw.py", target_dir / "esmfold2_worker_cw.py")
    data_dir = target_dir / "data"
    data_dir.mkdir()
    staged_manifest = data_dir / manifest.name
    shutil.copy2(manifest, staged_manifest)
    return exp_rel / "data" / manifest.name


def bootstrap(*, manifest: Path, out_prefix: str, shard_index: int, num_shards: int, n_samples: int,
              num_loops: int, num_sampling_steps: int, limit_per_shard: int | None) -> str:
    worker = "experiments/exp324_evals_esmfold2_pdb_sample/esmfold2_worker_cw.py"
    limit = f" --limit {limit_per_shard}" if limit_per_shard is not None else ""
    return f"""
set -euo pipefail
echo "[exp324] host=$(hostname) shard={shard_index}/{num_shards} image={IMAGE}"
nvidia-smi -L || true
export FSSPEC_S3_CONFIG_KWARGS='{{"s3": {{"addressing_style": "virtual"}}}}'
for attempt in 1 2 3; do
  apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates && break
  echo "[exp324] apt attempt $attempt failed; retrying" >&2
  sleep $((attempt * 10))
done
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv --python 3.12 /opt/venv
PY=/opt/venv/bin/python
$PY -m ensurepip --upgrade >/dev/null 2>&1 || true
$PY -m pip install --quiet --upgrade pip
$PY -m pip install --quiet torch
$PY -m pip install --quiet "esm @ git+https://github.com/Biohub/esm.git@main"
$PY -m pip install --quiet accelerate "huggingface_hub[hf_transfer]" numpy fsspec s3fs boto3
$PY -c "from esm.models.esmfold2 import EsmFold2Model, ESMFold2InputBuilder, ESMFOLD2_HF_REPO; import transformers; print('[exp324] transformers', transformers.__version__, 'repo', ESMFOLD2_HF_REPO)"
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
exec $PY {shlex.quote(worker)} \\
  --manifest {shlex.quote(str(manifest))} \\
  --out-prefix {shlex.quote(out_prefix)} \\
  --shard-index {shard_index} \\
  --num-shards {num_shards} \\
  --n-samples {n_samples} \\
  --num-loops {num_loops} \\
  --num-sampling-steps {num_sampling_steps}{limit}
""".strip()


def command_for_shard(args: argparse.Namespace, shard_index: int) -> list[str]:
    cmd = [
        *IRIS,
        "job",
        "run",
        "--target-cluster",
        "cw-rno2a",
        "--no-wait",
        "--enable-extra-resources",
        "--gpu",
        "H100x1",
        "--cpu",
        "8",
        "--memory",
        "64GB",
        "--disk",
        "128GB",
        "--priority",
        args.priority,
        "--max-retries",
        str(args.max_retries),
        "--task-image",
        IMAGE,
        "--no-sync",
        "--job-name",
        f"{args.job_prefix}-s{shard_index:03d}-of-{args.shards:03d}",
        "--",
        "bash",
        "-lc",
        bootstrap(
            manifest=args.manifest,
            out_prefix=args.out_prefix,
            shard_index=shard_index,
            num_shards=args.shards,
            n_samples=args.n_samples,
            num_loops=args.num_loops,
            num_sampling_steps=args.num_sampling_steps,
            limit_per_shard=args.limit_per_shard,
        ),
    ]
    return cmd


def main() -> None:
    args = parse_args()
    staged_manifest = stage_bundle(args.bundle_dir, args.manifest)
    dispatch_args = argparse.Namespace(**{**vars(args), "manifest": staged_manifest})
    for shard_index in range(args.shards):
        cmd = command_for_shard(dispatch_args, shard_index)
        print(" ".join(shlex.quote(part) for part in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=args.bundle_dir)


if __name__ == "__main__":
    main()
