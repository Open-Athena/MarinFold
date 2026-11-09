"""Publish the small analysis tables and plots beside the raw GPU artifacts."""

import argparse
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="gb-v1-20261009")
    args = parser.parse_args()
    root = f"hf://buckets/open-athena/MarinFold/data/exp354-short-rollout-consensus/{args.run_id}"
    for source, destination in (("data", "analysis"), ("plots", "plots")):
        subprocess.run(["hf", "buckets", "sync", str(HERE / source), f"{root}/{destination}"], check=True)
    subprocess.run(["hf", "buckets", "cp", str(HERE / "README.md"), f"{root}/README.md"], check=True)


if __name__ == "__main__":
    main()
