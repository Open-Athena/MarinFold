"""Validate and summarize the strict CoreWeave contact rollout."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
TARGETS = HERE / "data/foldbench_complex_contact_eval_targets.parquet"
DEFAULT_TIMINGS = HERE / "data/timings.csv"
DEFAULT_MANIFEST = HERE / "data/contact_rollout_manifest.json"
MODEL_LABEL = "contacts-v1-exp343-m2-p06-complex-1.5B-step-280154"
S3_PREFIX = (
    "s3://marin-us-east-02a/MarinFold/"
    "exp350_evals_complex_holdout_survival/foldbench-pair-holdout-v1/"
    "rollout-final-context-complete"
)
HF_PREFIX = (
    "hf://buckets/open-athena/MarinFold/data/evals/"
    "exp350_foldbench_pair_holdout/contact_eval_v1/rollout_v1"
)


def sha256(path: Path) -> str:
    """Return a file's SHA-256 hex digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path, root: Path) -> dict:
    """Describe one immutable raw rollout artifact."""
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Require 100 finished rollouts per target and write timings plus provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--timings", type=Path, default=DEFAULT_TIMINGS)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    score_paths = sorted((args.root / "scores").glob("*.parquet"))
    timing_paths = sorted((args.root / "timings").glob("*.parquet"))
    complete_paths = sorted((args.root / "complete").glob("*.json"))
    if not (len(score_paths) == len(timing_paths) == len(complete_paths) == 6):
        raise ValueError(
            "Expected six score, timing, and completion parts; got "
            f"{len(score_paths)}, {len(timing_paths)}, {len(complete_paths)}"
        )

    timing_table = pa.concat_tables([pq.read_table(path) for path in timing_paths])
    timing_rows = sorted(timing_table.to_pylist(), key=lambda row: row["stem"])
    targets = pq.read_table(TARGETS).to_pylist()
    expected = {row["stem"] for row in targets}
    observed = {row["stem"] for row in timing_rows}
    if observed != expected or len(timing_rows) != len(expected):
        raise ValueError(
            f"Timing/target mismatch: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    invalid = [
        row["stem"]
        for row in timing_rows
        if row["n_rollouts"] != 100
        or row["unfinished_rollouts"] != 0
        or not row["complete"]
    ]
    if invalid:
        raise ValueError(f"Incomplete strict rollout targets: {invalid}")
    write_csv(args.timings, timing_rows)

    completions = [json.loads(path.read_text()) for path in complete_paths]
    total_rollouts = sum(row["total_rollouts"] for row in completions)
    usable_rollouts = sum(row["usable_rollouts"] for row in completions)
    unfinished_rollouts = sum(row["unfinished_rollouts"] for row in completions)
    if (total_rollouts, usable_rollouts, unfinished_rollouts) != (2300, 2300, 0):
        raise ValueError(
            "Unexpected completion totals: "
            f"{total_rollouts}, {usable_rollouts}, {unfinished_rollouts}"
        )

    raw_paths = score_paths + timing_paths + complete_paths
    manifest = {
        "model_label": MODEL_LABEL,
        "targets_sha256": sha256(TARGETS),
        "s3_prefix": S3_PREFIX,
        "public_prefix": HF_PREFIX,
        "jobs": [
            f"/bizon/exp350-foldbench-complex-s{index}of6-final23"
            for index in range(6)
        ],
        "selection": {
            "targets": len(expected),
            "rollouts_per_target": 100,
            "total_rollouts": total_rollouts,
            "usable_rollouts": usable_rollouts,
            "unfinished_rollouts": unfinished_rollouts,
            "score_rows": sum(pq.read_metadata(path).num_rows for path in score_paths),
        },
        "derived": {
            "timings.csv": sha256(args.timings),
            "contact_r_precision.csv": sha256(HERE / "data/contact_r_precision.csv"),
            "contact_r_precision_summary.csv": sha256(
                HERE / "data/contact_r_precision_summary.csv"
            ),
        },
        "files": [artifact(path, args.root) for path in raw_paths],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest["selection"], indent=2))


if __name__ == "__main__":
    main()
