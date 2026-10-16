"""Read W&B and persist observations; never submit, stop, or choose a placement."""

import argparse
import json
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import wandb


def main() -> None:
    """Report all group runs and update active dispatch progress if requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path)
    parser.add_argument(
        "--iris-bin", default="/home/bizon/git/marin-freshiris/.venv/bin/iris"
    )
    args = parser.parse_args()
    observed_at = datetime.now(UTC).isoformat()
    runs = list(
        wandb.Api().runs(
            "open-athena/MarinFold", filters={"group": "exp364-mpnn-only-lr-sweep"}
        )
    )
    records = {}
    for run in runs:
        summary = dict(run.summary)
        row = {
            "observed_at": observed_at,
            "id": run.id,
            "url": run.url,
            "state": run.state,
            "progress": summary.get("run_progress"),
            "step": summary.get("_step"),
            "metrics": {
                key: value
                for key, value in summary.items()
                if key
                in (
                    "train/loss",
                    "eval/loss",
                    "throughput/total_tokens",
                    "throughput/tokens_per_second",
                    "throughput/step_duration",
                )
            },
        }
        records[run.id] = row
        print(json.dumps(row), flush=True)
    if args.database is None:
        return
    with sqlite3.connect(args.database) as db:
        active = db.execute(
            "SELECT t.trial_id, t.wandb_run_id, t.high_water_progress, d.dispatch_id, d.iris_job_id "
            "FROM trials t JOIN dispatches d USING(trial_id) WHERE d.active = 1"
        ).fetchall()
        for trial, run_id, high_water, dispatch, iris_id in active:
            # W&B was read first. The exact Iris root supplies only liveness;
            # failures propagate so stale observations cannot drive recovery.
            result = subprocess.run(
                [args.iris_bin, "--cluster", "marin", "job", "describe", iris_id],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            states = [
                line.split()[1]
                for line in result.stdout.splitlines()
                if line.startswith("State:")
            ]
            if len(states) != 1:
                raise ValueError(f"Cannot establish exact Iris state for {iris_id}")
            row = records.get(run_id, {})
            progress = row.get("progress")
            db.execute(
                "INSERT INTO observations(trial_id,dispatch_id,observed_at,wandb_state,run_progress,iris_running) VALUES (?,?,?,?,?,?)",
                (
                    trial,
                    dispatch,
                    observed_at,
                    row.get("state"),
                    progress,
                    int(states[0] == "running"),
                ),
            )
            if progress is not None and progress > high_water:
                db.execute(
                    "UPDATE trials SET high_water_progress = ?, status = 'running' WHERE trial_id = ?",
                    (progress, trial),
                )
                db.execute(
                    "INSERT INTO events(recorded_at,kind,trial_id,dispatch_id,detail) VALUES (?,?,?,?,?)",
                    (
                        observed_at,
                        "progress",
                        trial,
                        dispatch,
                        json.dumps({"previous": high_water, "current": progress}),
                    ),
                )
            print(
                json.dumps(
                    {"trial": trial, "iris_id": iris_id, "iris_state": states[0]}
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
