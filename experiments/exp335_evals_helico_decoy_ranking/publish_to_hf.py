"""Publish consolidated exp335 results to the public MarinFold HF bucket."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from stage_full_cw import (
    DEFAULT_KUBECONFIG,
    INPUT_PREFIX,
    REFERENCE_CSV_REMOTE,
    REFERENCE_CSV_SHA256,
    coreweave_s3,
)

DESTINATION = (
    "hf://buckets/open-athena/MarinFold/data/evals/exp335_helico_decoy_ranking/full-v1"
)
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PLOTS_DIR = HERE / "plots"


def task_path(path: Path) -> Path:
    """Resolve a CLI path relative to the experiment directory."""
    return path if path.is_absolute() else HERE / path


def download_verified(fs, remote: str, destination: Path, expected: str) -> None:
    """Download one rebuild input and verify its SHA-256."""
    payload = fs.cat_file(remote)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ValueError(f"s3://{remote}: digest {observed} != {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(destination)


def rebuild_public_artifacts(args: argparse.Namespace) -> None:
    """Rebuild the consolidated tables and figures from durable S3 results."""
    fs = coreweave_s3(args.kubeconfig)
    manifest_remote = f"{INPUT_PREFIX}/manifest.json"
    manifest_digest = fs.cat_file(f"{manifest_remote}.sha256").decode().strip()
    download_verified(
        fs,
        manifest_remote,
        args.input_dir / "manifest.json",
        manifest_digest,
    )
    download_verified(
        fs,
        REFERENCE_CSV_REMOTE,
        args.af2rank_csv,
        REFERENCE_CSV_SHA256,
    )
    commands = [
        [
            sys.executable,
            str(HERE / "fetch_full_cw.py"),
            "--input-dir",
            str(args.input_dir),
            "--output-dir",
            str(args.results_dir),
            "--num-shards",
            str(args.num_shards),
            "--kubeconfig",
            str(args.kubeconfig),
            "--allow-remote-only-inputs",
        ],
        [
            sys.executable,
            str(HERE / "analyze_full.py"),
            "--results-dir",
            str(args.results_dir),
            "--af2rank-csv",
            str(args.af2rank_csv),
            "--output-dir",
            str(DATA_DIR),
        ],
        [sys.executable, str(HERE / "plot_full.py")],
        [sys.executable, str(HERE / "build_summary.py")],
    ]
    for command in commands:
        print(" ".join(command), flush=True)
        subprocess.run(command, cwd=HERE, check=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("scratch/full_results")
    )
    parser.add_argument("--input-dir", type=Path, default=Path("scratch/full_inputs"))
    parser.add_argument(
        "--af2rank-csv",
        type=Path,
        default=Path("scratch/reference/rosetta_gapseq.csv"),
    )
    parser.add_argument("--kubeconfig", type=Path, default=DEFAULT_KUBECONFIG)
    parser.add_argument("--num-shards", type=int, default=96)
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="upload already-rebuilt artifacts without reading the durable S3 run",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Validate auth and copy the reproducible public result bundle."""
    args = parse_args()
    whoami = json.loads(
        subprocess.check_output(["hf", "auth", "whoami", "--format", "json"], text=True)
    )
    orgs = whoami.get("orgs", "")
    if "open-athena" not in orgs:
        raise PermissionError("active Hugging Face token lacks open-athena access")
    args.results_dir = task_path(args.results_dir)
    args.input_dir = task_path(args.input_dir)
    args.af2rank_csv = task_path(args.af2rank_csv)
    args.kubeconfig = task_path(args.kubeconfig)
    if not args.skip_rebuild:
        rebuild_public_artifacts(args)
    paths = [
        args.results_dir / "candidate_metrics.csv",
        args.results_dir / "sample_metrics.csv",
        args.results_dir / "timings.csv",
        args.results_dir / "manifest.json",
        DATA_DIR / "full_metric_summary.csv",
        DATA_DIR / "full_native_summary.csv",
        DATA_DIR / "full_paired_comparisons.csv",
        DATA_DIR / "full_per_target_metrics.csv",
        DATA_DIR / "full_native_ranking.csv",
        DATA_DIR / "full_run_manifest.json",
        PLOTS_DIR / "full_metric_comparison.png",
        PLOTS_DIR / "full_metric_comparison.png.meta.json",
        PLOTS_DIR / "full_native_selection.png",
        PLOTS_DIR / "full_native_selection.png.meta.json",
        PLOTS_DIR / "full_native_top5.png",
        PLOTS_DIR / "full_native_top5.png.meta.json",
        PLOTS_DIR / "summary.pdf",
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing publish inputs: {missing}")
    for path in paths:
        destination = f"{DESTINATION}/{path.name}"
        command = ["hf", "buckets", "cp", str(path), destination, "--format", "agent"]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
