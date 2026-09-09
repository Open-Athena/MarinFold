"""Persist Iris task-attempt wall time for every exp278 single-H100 job."""

import json
from pathlib import Path

from iris.cli.connect import open_iris_client
from rigging.timing import Timestamp

from analyze_screen import write_csv

HERE = Path(__file__).resolve().parent


def main() -> None:
    rows = []
    now = Timestamp.now()
    with open_iris_client(cluster_name="cw-rno2a", workspace=None) as client:
        jobs = client.list_jobs(prefix="/bizon/exp278-", limit=200)
        for job in jobs:
            for task in client.list_tasks(job.job_id):
                for attempt in task.attempts:
                    start = attempt.started_at
                    finish = attempt.finished_at or task.finished_at or job.finished_at
                    seconds = (
                        0.0
                        if start is None
                        else max(0, (finish or now).epoch_ms() - start.epoch_ms())
                        / 1000
                    )
                    rows.append(
                        {
                            "job_id": job.job_id.to_wire(),
                            "task_id": task.task_id.to_wire(),
                            "attempt": attempt.attempt_number,
                            "state": attempt.state.value,
                            "started_epoch_ms": start.epoch_ms() if start else None,
                            "finished_epoch_ms": finish.epoch_ms() if finish else None,
                            "elapsed_seconds": seconds,
                            "gpu_count": 1,
                            "h100_hours": seconds / 3600,
                            "node_name": attempt.node_name,
                        }
                    )
    write_csv(HERE / "data/iris-resource-time.csv", rows)
    summary = {
        "snapshot_epoch_ms": now.epoch_ms(),
        "jobs": len(jobs),
        "attempts": len(rows),
        "h100_hours": sum(row["h100_hours"] for row in rows),
        "pilot_cap_h100_hours": 100,
        "interpretation": "one H100 per task; Iris attempt wall time including setup, compilation, inference and failures; not a provider invoice",
    }
    (HERE / "data/iris-resource-time.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
