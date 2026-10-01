"""Dispatch exp324 ColabFold MSA-depth shards to Iris CPU workers."""

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path

IMAGE = "python:3.12-slim"
DEFAULT_OUT_PREFIX = "gs://marin-us-central1/protein-structure/MarinFold/exp324_esmfold2_pdb_sample/msa_depth_colabfold"
IRIS = ["uv", "run", "--project", "/tmp/marin-iris-origin-main-fresh/lib/iris", "iris", "--controller-url", "http://10.128.0.3:10000"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=Path("data/sample_10000_targets.parquet"))
    parser.add_argument("--out-prefix", default=DEFAULT_OUT_PREFIX)
    parser.add_argument("--shards", type=int, default=32)
    parser.add_argument("--limit-per-shard", type=int)
    parser.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--job-prefix", default="exp324-msa-depth-colabfold")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bundle-dir", type=Path, default=Path("/tmp/exp324_msa_depth_bundle"))
    return parser.parse_args()


def stage_bundle(bundle_dir: Path, targets: Path) -> Path:
    exp_rel = Path("experiments/exp324_evals_esmfold2_pdb_sample")
    target_dir = bundle_dir / exp_rel
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    target_dir.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    shutil.copy2(here / "compute_msa_depth_colabfold.py", target_dir / "compute_msa_depth_colabfold.py")
    msa_src = here.parent / "exp12_data_protenix_foldbench_monomers" / "msa_depth.py"
    msa_dst = bundle_dir / "experiments" / "exp12_data_protenix_foldbench_monomers"
    msa_dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(msa_src, msa_dst / "msa_depth.py")
    data_dir = target_dir / "data"
    data_dir.mkdir()
    staged_targets = data_dir / targets.name
    shutil.copy2(here / targets if not targets.is_absolute() else targets, staged_targets)
    return exp_rel / "data" / targets.name


def bootstrap(targets: Path, out_prefix: str, shard_index: int, shards: int, limit_per_shard: int | None) -> str:
    worker = "experiments/exp324_evals_esmfold2_pdb_sample/compute_msa_depth_colabfold.py"
    limit = f" --limit {limit_per_shard}" if limit_per_shard is not None else ""
    return f"""
set -euo pipefail
echo "[exp324-msa-depth] host=$(hostname) shard={shard_index}/{shards}"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet pandas pyarrow numpy requests fsspec gcsfs
exec python {shlex.quote(worker)} \
  --targets {shlex.quote(str(targets))} \
  --out-prefix {shlex.quote(out_prefix)} \
  --shard-index {shard_index} \
  --num-shards {shards}{limit}
""".strip()


def command_for_shard(args: argparse.Namespace, targets: Path, shard_index: int) -> list[str]:
    return [
        *IRIS,
        "job", "run",
        "--no-wait",
        "--enable-extra-resources",
        "--cpu", "2",
        "--memory", "8GB",
        "--disk", "20GB",
        "--zone", args.zone,
        "--priority", args.priority,
        "--max-retries", str(args.max_retries),
        "--task-image", IMAGE,
        "--no-sync",
        "--job-name", f"{args.job_prefix}-s{shard_index:03d}-of-{args.shards:03d}",
        "--",
        "bash", "-lc",
        bootstrap(targets, args.out_prefix, shard_index, args.shards, args.limit_per_shard),
    ]


def main() -> None:
    args = parse_args()
    staged_targets = stage_bundle(args.bundle_dir, args.targets)
    for shard_index in range(args.shards):
        cmd = command_for_shard(args, staged_targets, shard_index)
        print(" ".join(shlex.quote(part) for part in cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True, cwd=args.bundle_dir)


if __name__ == "__main__":
    main()
