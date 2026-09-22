"""Publish exp311's pinned inputs and complete result tables to the public bucket."""

import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEST = "hf://buckets/open-athena/MarinFold/data/exp311-helico-exp277-contact-count-sweep"


def sync(source: Path, suffix: str) -> None:
    """Upload a directory without deleting any prior public artifact."""
    if not source.exists():
        raise FileNotFoundError(source)
    subprocess.run([
        "hf", "buckets", "sync", str(source), f"{DEST}/{suffix}/",
        "--no-delete", "--exclude", "smoke/**",
    ], check=True)


def main() -> None:
    required = (
        HERE / "scratch" / "results" / "samples.csv",
        HERE / "scratch" / "results" / "timings.csv",
        HERE / "scratch" / "results" / "run_manifest.json",
        HERE / "data" / "metric_summary.csv",
        HERE / "data" / "per_target_selection.csv",
        HERE / "data" / "input_manifest.json",
        HERE / "plots" / "summary.pdf",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    sync(HERE / "scratch" / "targets", "inputs")
    sync(HERE / "scratch" / "results", "results")
    sync(HERE / "data", "analysis")
    sync(HERE / "plots", "plots")
    print(DEST)


if __name__ == "__main__":
    main()
