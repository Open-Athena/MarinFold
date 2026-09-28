"""Dispatch pyconfind contact scoring shards for exp324 ESMFold2 outputs."""

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path

IMAGE = "python:3.12-slim"
DEFAULT_PREDICTION_PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/pdb_deduped_10k_n1"
DEFAULT_OUT_PREFIX = "s3://marin-us-east-02a/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/pdb_deduped_10k_n1_scored"
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
    parser.add_argument("--gt-parquet", type=Path, required=True)
    parser.add_argument("--prediction-prefix", default=DEFAULT_PREDICTION_PREFIX)
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--shards", type=int, default=64)
    parser.add_argument("--limit-per-shard", type=int)
    parser.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--job-prefix", default="exp324-score-esmfold2-10k")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bundle-dir", type=Path, default=Path("/tmp/exp324_score_esmfold2_bundle"))
    return parser.parse_args()


def stage_bundle(bundle_dir: Path, manifest: Path, gt_parquet: Path) -> tuple[Path, Path]:
    exp_rel = Path("experiments/exp324_evals_esmfold2_pdb_sample")
    target_dir = bundle_dir / exp_rel
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    target_dir.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    shutil.copy2(here / "score_esmfold2_contacts_cw.py", target_dir / "score_esmfold2_contacts_cw.py")
    data_dir = target_dir / "data"
    data_dir.mkdir()
    staged_manifest = data_dir / manifest.name
    staged_gt = data_dir / gt_parquet.name
    shutil.copy2(manifest, staged_manifest)
    shutil.copy2(gt_parquet, staged_gt)
    return exp_rel / "data" / manifest.name, exp_rel / "data" / gt_parquet.name


def bootstrap(*, manifest: Path, gt_parquet: Path, prediction_prefix: str, out_prefix: str,
              shard_index: int, num_shards: int, limit_per_shard: int | None) -> str:
    worker = "experiments/exp324_evals_esmfold2_pdb_sample/score_esmfold2_contacts_cw.py"
    limit = f" --limit {limit_per_shard}" if limit_per_shard is not None else ""
    return f"""
set -euo pipefail
echo "[exp324-score] host=$(hostname) shard={shard_index}/{num_shards} image={IMAGE}"
export FSSPEC_S3_CONFIG_KWARGS='{{"s3": {{"addressing_style": "virtual"}}}}'
for attempt in 1 2 3; do
  apt-get update -qq && apt-get install -y -qq --no-install-recommends git curl ca-certificates build-essential && break
  echo "[exp324-score] apt attempt $attempt failed; retrying" >&2
  sleep $((attempt * 10))
done
python -m pip install --quiet --upgrade pip
python -m pip install --quiet numpy pandas pyarrow fsspec s3fs boto3 gemmi
python -m pip install --quiet 'marinfold[contacts-v1] @ git+https://github.com/Open-Athena/MarinFold.git@main#subdirectory=marinfold'
exec python {shlex.quote(worker)} \
  --manifest {shlex.quote(str(manifest))} \
  --gt-parquet {shlex.quote(str(gt_parquet))} \
  --prediction-prefix {shlex.quote(prediction_prefix)} \
  --out-prefix {shlex.quote(out_prefix)} \
  --shard-index {shard_index} \
  --num-shards {num_shards}{limit}
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
        "--cpu",
        "8",
        "--memory",
        "32GB",
        "--disk",
        "32GB",
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
            gt_parquet=args.gt_parquet,
            prediction_prefix=args.prediction_prefix,
            out_prefix=args.out_prefix,
            shard_index=shard_index,
            num_shards=args.shards,
            limit_per_shard=args.limit_per_shard,
        ),
    ]
    return cmd


def main() -> None:
    args = parse_args()
    staged_manifest, staged_gt = stage_bundle(args.bundle_dir, args.manifest, args.gt_parquet)
    dispatch_args = argparse.Namespace(**{**vars(args), "manifest": staged_manifest, "gt_parquet": staged_gt})
    for shard_index in range(args.shards):
        cmd = command_for_shard(dispatch_args, shard_index)
        print(" ".join(shlex.quote(part) for part in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=args.bundle_dir)


if __name__ == "__main__":
    main()
