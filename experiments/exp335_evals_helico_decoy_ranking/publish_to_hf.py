"""Publish consolidated exp335 results to the public MarinFold HF bucket."""

import argparse
import json
import subprocess
from pathlib import Path

DESTINATION = (
    "hf://buckets/open-athena/MarinFold/data/evals/exp335_helico_decoy_ranking/full-v1"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("scratch/full_results")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots"))
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
    paths = [
        args.results_dir / "candidate_metrics.csv",
        args.results_dir / "sample_metrics.csv",
        args.results_dir / "timings.csv",
        args.results_dir / "manifest.json",
        args.data_dir / "full_metric_summary.csv",
        args.data_dir / "full_native_summary.csv",
        args.data_dir / "full_per_target_metrics.csv",
        args.data_dir / "full_native_ranking.csv",
        args.data_dir / "full_run_manifest.json",
        args.plots_dir / "full_metric_comparison.png",
        args.plots_dir / "full_native_selection.png",
        args.plots_dir / "summary.pdf",
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
