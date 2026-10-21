"""Report durable S3 completion for the exp335 CoreWeave run."""

import argparse
import json
import re
from pathlib import Path

from full_worker_cw import OUTPUT_ROOT, RUN_FINGERPRINT, assign_tasks
from stage_full_cw import DEFAULT_KUBECONFIG, coreweave_s3

PART_PATTERN = re.compile(r"/([^/]+)/part-(\d+)-(\d+)\.timing\.json$")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("scratch/full_inputs"))
    parser.add_argument("--num-shards", type=int, default=96)
    parser.add_argument("--kubeconfig", type=Path, default=DEFAULT_KUBECONFIG)
    return parser.parse_args()


def main() -> None:
    """Compare expected tasks with timing companions in object storage."""
    args = parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    shards = assign_tasks(manifest, args.num_shards)
    expected = {
        (task["target"], task["start"], task["end"]): shard_index
        for shard_index, tasks in enumerate(shards)
        for task in tasks
    }
    fs = coreweave_s3(args.kubeconfig)
    prefix = f"{OUTPUT_ROOT}/{RUN_FINGERPRINT}".removeprefix("s3://")
    try:
        paths = fs.find(prefix)
    except FileNotFoundError:
        paths = []
    completed = set()
    for path in paths:
        match = PART_PATTERN.search(path)
        if match:
            completed.add((match.group(1), int(match.group(2)), int(match.group(3))))
    unexpected = completed - set(expected)
    missing = set(expected) - completed
    if unexpected:
        raise ValueError(f"unexpected timing objects: {sorted(unexpected)[:10]}")
    incomplete_shards = sorted({expected[task] for task in missing})
    summary = {
        "run_fingerprint": RUN_FINGERPRINT,
        "expected_parts": len(expected),
        "completed_parts": len(completed),
        "completed_candidates": sum(end - start for _, start, end in completed),
        "expected_candidates": manifest["n_candidates"],
        "completion_fraction": (
            sum(end - start for _, start, end in completed) / manifest["n_candidates"]
        ),
        "incomplete_shards": incomplete_shards,
        "n_incomplete_shards": len(incomplete_shards),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
