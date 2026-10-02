"""Record W&B and exact Iris observations without selecting recovery actions."""

import argparse
import json
import re
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import wandb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--marin-checkout", type=Path, default=Path("/home/bizon/git/marin")
    )
    args = parser.parse_args()
    api = wandb.Api()
    runs = {
        r.id: r
        for r in api.runs(
            "open-athena/MarinFold", filters={"group": "exp347-qwen-base-contacts"}
        )
    }
    repo = Path(__file__).resolve().parents[2]
    history_changed = False
    with sqlite3.connect(args.database) as db:
        db.row_factory = sqlite3.Row
        active = db.execute(
            "SELECT d.*,t.wandb_run_id,t.wandb_url FROM dispatches d JOIN trials t USING(trial_id) WHERE d.active=1"
        ).fetchall()
        for row in active:
            run = runs.get(row["wandb_run_id"])
            progress = run.summary.get("run_progress") if run else None
            command = [
                "uv",
                "run",
                "--project",
                str(args.marin_checkout),
                "--package",
                "marin-iris",
                "iris",
                "--cluster",
                "marin",
                "job",
                "describe",
                row["iris_job_id"],
            ]
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            state_match = re.search(r"^State: (\S+)", result.stdout, re.MULTILINE)
            if not state_match:
                raise ValueError("Iris describe returned no job state")
            state = state_match[1]
            terminal = state in {
                "succeeded",
                "failed",
                "killed",
                "cancelled",
                "canceled",
                "unschedulable",
            }
            now = datetime.now(UTC).isoformat()
            db.execute(
                "INSERT INTO observations(trial_id,dispatch_id,observed_at,wandb_state,run_progress,iris_running) VALUES(?,?,?,?,?,?)",
                (
                    row["trial_id"],
                    row["dispatch_id"],
                    now,
                    run.state if run else None,
                    progress,
                    int(not terminal),
                ),
            )
            if run:
                db.execute(
                    "UPDATE trials SET wandb_url=?,high_water_progress=MAX(high_water_progress,?),status=? WHERE trial_id=?",
                    (
                        run.url,
                        progress or 0,
                        "awaiting_checkpoint" if terminal else "running",
                        row["trial_id"],
                    ),
                )
                if row["wandb_url"] is None:
                    subprocess.run(
                        [
                            "uv",
                            "run",
                            "--project",
                            str(repo / "scripts"),
                            "python",
                            str(repo / "scripts/history.py"),
                            "new",
                            "--wandb-url",
                            run.url,
                            "--wandb-name",
                            run.name,
                            "--experiment",
                            Path(__file__).parent.name,
                            "--kind",
                            "models",
                            "--short",
                            "Qwen3.5 base full-weight contact fine-tuning",
                            "--iris-jobs",
                            row["iris_job_id"],
                        ],
                        check=True,
                    )
                    history_changed = True
                elif row["attempt"] > 1:
                    subprocess.run(
                        [
                            "uv",
                            "run",
                            "--project",
                            str(repo / "scripts"),
                            "python",
                            str(repo / "scripts/history.py"),
                            "add-iris-job",
                            run.name,
                            row["iris_job_id"],
                        ],
                        check=True,
                        capture_output=True,
                    )
                    history_changed = True
            if terminal:
                db.execute(
                    "UPDATE dispatches SET active=0,ended_at=?,outcome=? WHERE dispatch_id=?",
                    (now, state, row["dispatch_id"]),
                )
                db.execute(
                    "INSERT INTO events(recorded_at,kind,trial_id,dispatch_id,detail) VALUES(?,?,?,?,?)",
                    (
                        now,
                        "dispatch_terminal",
                        row["trial_id"],
                        row["dispatch_id"],
                        state,
                    ),
                )
            print(
                json.dumps(
                    {
                        "trial": row["trial_id"],
                        "iris": state,
                        "wandb": run.state if run else None,
                        "progress": progress,
                        "summary": dict(run.summary) if run else None,
                    }
                ),
                flush=True,
            )
    if history_changed:
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(repo / "scripts"),
                "python",
                str(repo / "scripts/history.py"),
                "update-index",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
