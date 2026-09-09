"""Read small durable markers for a live scale-run progress snapshot."""

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from launch import storage_filesystem


def case_counts(
    case: dict, root: str, paths: set[str], fs, sequence_files: dict
) -> dict:
    """Count committed artifacts; never download coordinate or PDB columns."""
    base = f"{root}/cases/{case['id']}"
    counts = {"generated": 0, "refolded": 0, "quality_pass": 0}
    for stage, mapping in (
        ("generated", {"count": "generated"}),
        ("folded", {"candidates": "refolded", "quality_pass": "quality_pass"}),
    ):
        marker = next(
            (
                f"{base}/{stage}/{name}.json"
                for name in ("complete", "progress")
                if f"{base}/{stage}/{name}.json" in paths
            ),
            None,
        )
        if marker:
            saved = json.loads(fs.cat(marker))
            counts.update({dest: saved[source] for source, dest in mapping.items()})
    # A batch archive is atomic and authoritative even if its progress PUT was interrupted.
    counts["generated"] = case["batch_size"] * sum(
        f"{base}/generated/batch-{batch:05d}.npz" in paths
        for batch in range(case["batches"])
    )
    counts["sequences_saved"] = sequence_files.get(case["id"], 0) * case["batch_size"]
    counts["complete"] = int(f"{base}/folded/complete.json" in paths)
    return {
        "case": case["id"],
        "worker": case["worker"],
        "length": case["length"],
        "condition": case["condition"],
        **counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    fs = storage_filesystem(args.cluster)
    root = plan["output"].removeprefix("s3://")
    paths = set(fs.find(root + "/cases"))
    sequence_files = Counter(
        path.split("/cases/", 1)[1].split("/", 1)[0]
        for path in paths
        if "/folded/sequences/" in path and path.endswith(".parquet")
    )
    active = [
        case
        for case in plan["cases"]
        if any(
            f"{root}/cases/{case['id']}/{stage}/{name}.json" in paths
            for stage in ("generated", "folded")
            for name in ("complete", "progress")
        )
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(
            pool.map(
                lambda case: case_counts(case, root, paths, fs, sequence_files), active
            )
        )
    statuses = [json.loads(fs.cat(path)) for path in fs.glob(root + "/workers/*.json")]
    totals = {
        key: sum(row[key] for row in rows)
        for key in (
            "generated",
            "sequences_saved",
            "refolded",
            "quality_pass",
            "complete",
        )
    }
    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": plan["run_id"],
        "planned_raw": plan["raw_candidates"],
        "target_documents": plan["target_documents"],
        **totals,
        "quality_retention": totals["quality_pass"] / totals["refolded"]
        if totals["refolded"]
        else None,
        "final_retained_documents": None,
        "final_retention_note": "Sequence/reference exclusion and global structural diversity cap still required. Quality-pass documents are provisional.",
        "worker_phases": dict(Counter(row["phase"] for row in statuses)),
        "sequence_batch_files": sum(
            "/folded/sequences/" in path and path.endswith(".parquet") for path in paths
        ),
        "snapshot_note": "Live nontransactional snapshot of committed batch boundaries; in-flight refolds are excluded.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(snapshot, indent=2) + "\n"
    (args.output / f"snapshot-{stamp}.json").write_text(payload)
    (args.output / "latest.json").write_text(payload)
    fs.pipe_file(f"{root}/reports/snapshot-{stamp}.json", payload.encode())
    if rows:
        with (args.output / f"cases-{stamp}.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(payload)


if __name__ == "__main__":
    main()
