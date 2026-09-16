"""Read small durable markers for a live scale-run progress snapshot."""

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

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


def staleness_report(
    detail: dict, plan: dict, rows: list[dict], stale_hours: float
) -> tuple[dict, list[dict]]:
    """Flag unfinished workers that have stopped writing, whatever Iris reports.

    Job state cannot separate working from hung: a preempted attempt can stay
    `running` with no pod (`PodDeleted`, `WorkloadEvictedDueToPreempted`) and
    never reschedule, so the queue silently stops while the job looks healthy.
    Recent object-store writes are the only reliable liveness signal, and this
    listing is already paid for by the progress snapshot.
    """
    worker_of = {case["id"]: case["worker"] for case in plan["cases"]}
    planned = Counter(case["worker"] for case in plan["cases"])
    done = Counter(row["worker"] for row in rows if row["complete"])
    newest: dict[int, datetime] = {}
    for path, info in detail.items():
        worker = worker_of.get(path.split("/cases/", 1)[1].split("/", 1)[0])
        mtime = info.get("LastModified") or info.get("mtime")
        if (
            worker is not None
            and mtime
            and (worker not in newest or mtime > newest[worker])
        ):
            newest[worker] = mtime
    now = datetime.now(timezone.utc)
    table = []
    for worker in sorted(planned):
        if done[worker] >= planned[worker]:
            continue
        mtime = newest.get(worker)
        table.append(
            {
                "worker": worker,
                "cases_complete": done[worker],
                "cases_planned": planned[worker],
                "last_write_utc": mtime.isoformat() if mtime else "",
                "idle_hours": round((now - mtime).total_seconds() / 3600, 2)
                if mtime
                else "",
            }
        )
    idle = [row["idle_hours"] for row in table if row["idle_hours"] != ""]
    stalled = [
        row
        for row in table
        if row["idle_hours"] != "" and row["idle_hours"] >= stale_hours
    ]
    summary = {
        "threshold_hours": stale_hours,
        "unfinished_workers": len(table),
        "never_written": sum(1 for row in table if row["idle_hours"] == ""),
        "stalled_workers": len(stalled),
        "median_idle_hours": round(median(idle), 2) if idle else None,
        "max_idle_hours": max(idle) if idle else None,
        "note": "Idle hours count time since the worker last wrote to the object store. Iris job state cannot detect a preempted attempt that stayed 'running' with no pod, so this is the liveness signal. A freshly resubmitted or just-preempted worker on a long-length queue can sit above the threshold legitimately until its first batch commits; confirm with job describe before replacing one.",
    }
    return summary, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stale-hours", type=float, default=3.0)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    fs = storage_filesystem(args.cluster)
    root = plan["output"].removeprefix("s3://")
    detail = fs.find(root + "/cases", detail=True)
    paths = set(detail)
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
    staleness, staleness_rows = staleness_report(detail, plan, rows, args.stale_hours)
    snapshot["staleness"] = staleness
    args.output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(snapshot, indent=2) + "\n"
    (args.output / f"snapshot-{stamp}.json").write_text(payload)
    (args.output / "latest.json").write_text(payload)
    fs.pipe_file(f"{root}/reports/snapshot-{stamp}.json", payload.encode())
    if rows:
        with (args.output / f"cases-{stamp}.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    if staleness_rows:
        with (args.output / f"staleness-{stamp}.csv").open("w") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(staleness_rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(staleness_rows)
    print(payload)
    if staleness["stalled_workers"]:
        print(
            f"STALL DETECTED: {staleness['stalled_workers']} of "
            f"{staleness['unfinished_workers']} unfinished workers have written "
            f"nothing for >= {args.stale_hours}h "
            f"(median {staleness['median_idle_hours']}h, max "
            f"{staleness['max_idle_hours']}h). Iris may still report them running; "
            f"check job describe: a recent attempt that is progressing is normal "
            f"churn, while an old attempt stuck on PodDeleted or "
            f"WorkloadEvictedDueToPreempted needs cancel and resubmit."
        )


if __name__ == "__main__":
    main()
