# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Dispatch MarinFold rollout scoring shards for the exp324 10k PDB-deduped manifest."""

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path

IMAGE = "vllm/vllm-openai:v0.10.2"
DEFAULT_OUT_PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/marinfold_10k_rollout_scores"
IRIS = ["uv", "run", "--project", "/tmp/marin-iris-origin-main-fresh/lib/iris", "iris", "--cluster=marin"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", choices=["contacts-v1", "delta-v2"], required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--shards", type=int, default=256)
    parser.add_argument("--start-shard", type=int, default=0)
    parser.add_argument("--end-shard", type=int, help="Exclusive shard index; defaults to --shards")
    parser.add_argument("--n-rollouts", type=int, default=100)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument("--limit-per-shard", type=int)
    parser.add_argument("--fixed-residue-position-embeddings", choices=["auto", "force"], default="auto")
    parser.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--job-prefix", default="exp324-marinfold-10k")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bundle-dir", type=Path, default=Path("/tmp/exp324_marinfold_rollout_bundle"))
    return parser.parse_args()


def stage_bundle(bundle_dir: Path, targets: Path, kind: str) -> Path:
    exp_rel = Path("experiments/exp324_evals_esmfold2_pdb_sample")
    target_dir = bundle_dir / exp_rel
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    target_dir.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    worker = "score_contacts_v1_rollout_10k_worker.py" if kind == "contacts-v1" else "score_delta_rollout_10k_worker.py"
    shutil.copy2(here / worker, target_dir / worker)
    if kind == "delta-v2":
        shutil.copy2(here / "delta_stream_rollout.py", target_dir / "delta_stream_rollout.py")
    data_dir = target_dir / "data"
    data_dir.mkdir()
    staged_targets = data_dir / targets.name
    shutil.copy2(targets, staged_targets)
    return exp_rel / "data" / targets.name


def bootstrap(args: argparse.Namespace, *, staged_targets: Path, shard_index: int) -> str:
    worker = "score_contacts_v1_rollout_10k_worker.py" if args.kind == "contacts-v1" else "score_delta_rollout_10k_worker.py"
    worker_path = f"experiments/exp324_evals_esmfold2_pdb_sample/{worker}"
    limit = f" --limit {args.limit_per_shard}" if args.limit_per_shard is not None else ""
    fixed_positions = (
        f" --fixed-residue-position-embeddings {args.fixed_residue_position_embeddings}"
        if args.kind == "contacts-v1"
        else ""
    )
    install_marinfold = "\nuv pip install --system --quiet 'marinfold @ git+https://github.com/Open-Athena/MarinFold.git@main#subdirectory=marinfold' --no-deps" if args.kind == "contacts-v1" else ""
    return f"""
set -euo pipefail
echo "[exp324-marinfold-10k] host=$(hostname) label={args.label} shard={shard_index}/{args.shards} kind={args.kind}"
nvidia-smi -L || true
export FSSPEC_S3_CONFIG_KWARGS='{{"s3": {{"addressing_style": "virtual"}}}}'
PY=python3
uv pip install --system --quiet numpy pandas pyarrow fsspec s3fs boto3 transformers tokenizers hf_transfer{install_marinfold}
export HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export VLLM_PORT=$($PY -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
exec $PY {shlex.quote(worker_path)} \
  --model {shlex.quote(args.model)} \
  --targets {shlex.quote(str(staged_targets))} \
  --out {shlex.quote(args.out_prefix)} \
  --label {shlex.quote(args.label)} \
  --shard {shard_index}/{args.shards} \
  --n-rollouts {args.n_rollouts} \
  --chunk {args.chunk}{limit}{fixed_positions}
""".strip()


def command_for_shard(args: argparse.Namespace, staged_targets: Path, shard_index: int) -> list[str]:
    return [
        *IRIS,
        "job", "run",
        "--target-cluster", "cw-rno2a",
        "--no-wait",
        "--enable-extra-resources",
        "--gpu", "H100x1",
        "--cpu", "8",
        "--memory", "64GB",
        "--disk", "192GB",
        "--priority", args.priority,
        "--max-retries", str(args.max_retries),
        "--task-image", IMAGE,
        "--no-sync",
        "--job-name", f"{args.job_prefix}-{args.label}-s{shard_index:03d}-of-{args.shards:03d}",
        "--",
        "bash", "-lc", bootstrap(args, staged_targets=staged_targets, shard_index=shard_index),
    ]


def main() -> None:
    args = parse_args()
    staged_targets = stage_bundle(args.bundle_dir, args.targets, args.kind)
    end_shard = args.shards if args.end_shard is None else args.end_shard
    for shard_index in range(args.start_shard, end_shard):
        cmd = command_for_shard(args, staged_targets, shard_index)
        print(" ".join(shlex.quote(part) for part in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=args.bundle_dir)


if __name__ == "__main__":
    main()
