"""Record W&B progress and exact Iris liveness; never dispatch or stop work.

Run using the workstation's current Marin environment, which supplies the
lightweight Iris state API. The experiment's worker dependency lock is unchanged.
"""

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import wandb
from iris.cli.connect import open_iris_client
from iris.cluster.types import JobName


def main() -> None:
    """Collect facts for the agent's next placement or recovery decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    args = parser.parse_args()
    db = sqlite3.connect(args.database)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    active = db.execute(
        "SELECT t.*,d.dispatch_id,d.iris_job_id FROM trials t "
        "JOIN dispatches d USING(trial_id) WHERE d.active=1"
    ).fetchall()
    api = wandb.Api(timeout=45)
    runs = {r.id: r for r in api.runs("open-athena/MarinFold", filters={"group": "exp362-smaller-model-scaling"})}
    observations = []
    with open_iris_client(cluster_name="marin", workspace=None) as client:
        for row in active:
            run = runs.get(row["wandb_run_id"])
            summary = dict(run.summary) if run is not None else {}
            state = client.job_state(JobName.from_wire(row["iris_job_id"])).value
            progress = summary.get("run_progress")
            observations.append((row, run, state, progress, summary))
    # Commit only after all service calls succeed, so an incomplete fleet query
    # cannot look like a fresh decision snapshot.
    now = datetime.now(UTC).isoformat()
    with db:
        for row, run, state, progress, summary in observations:
            db.execute(
                "INSERT INTO observations(trial_id,dispatch_id,observed_at,wandb_state,run_progress,iris_running) "
                "VALUES(?,?,?,?,?,?)",
                (row["trial_id"], row["dispatch_id"], now, run.state if run else None, progress, state == "running"),
            )
            if progress is not None and progress > row["high_water_progress"]:
                db.execute("UPDATE trials SET high_water_progress=? WHERE trial_id=?", (progress, row["trial_id"]))
            print(json.dumps({
                "observed_at": now, "trial": row["trial_id"], "wandb_id": row["wandb_run_id"],
                "wandb_state": run.state if run else None, "iris_state": state,
                "run_progress": progress,
                "metrics": {key: value for key, value in summary.items() if key in (
                    "train/loss", "eval/loss", "train/step", "throughput/duration",
                    "throughput/total_tokens", "num_params", "parameter_count",
                )},
            }))
    db.close()


if __name__ == "__main__":
    main()
