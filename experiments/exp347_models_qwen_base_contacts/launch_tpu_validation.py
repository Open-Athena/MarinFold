"""Submit one bounded, regional full-state TPU validation and persist its identity."""

import argparse
import json
import netrc
import os
import shlex
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from common import FORMATS

ROOT = "gs://marin-us-east5/protein-structure/MarinFold/exp347_qwen_base_contacts"


def main() -> None:
    """Launch exactly the selected validation, without scheduling production jobs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.attempt < 1:
        raise ValueError("Attempt numbers must be positive")
    experiment = Path(__file__).resolve().parent
    receipts = experiment / "data/tpu_forks"
    for filename in ("cpu_math_validation.json", "cpu_checkpoint_validation.json"):
        if json.loads((receipts / filename).read_text())["status"] != "passed":
            raise ValueError(f"Numerical validation has not passed: {filename}")
    snapshot = json.loads((receipts / f"{args.format}_parent.json").read_text())
    name = f"debug-{snapshot['fork_run']}-a{args.attempt:02d}"
    job = f"exp347-tpu-{args.format}-validate-a{args.attempt:02d}"
    job_id = f"/timodonnell/{job}"
    command = [
        "uv",
        "run",
        "--project",
        "/home/bizon/git/marin",
        "--package",
        "marin-iris",
        "iris",
        "--cluster",
        "marin",
        "job",
        "run",
        "--zone",
        "us-east5-a",
        "--priority",
        "batch",
        "--user",
        "timodonnell",
        "--job-name",
        job,
        "--no-wait",
        "--enable-extra-resources",
        "--tpu",
        "v5p-8",
        "--cpu",
        "32",
        "--memory",
        "256GB",
        "--disk",
        "64GB",
        "--no-sync",
        "--timeout",
        "7200",
        "--max-retries",
        "0",
        "-e",
        "DECOUPLE_GCLOUD",
        "TRUE",
        "-e",
        "OMP_NUM_THREADS",
        "8",
        "-e",
        "TOKENIZERS_PARALLELISM",
        "false",
        "-e",
        "WANDB_API_KEY",
        "<redacted>",
        "--",
        "uv",
        "run",
        "--project",
        "tpu_runtime",
        "--locked",
        "--extra",
        "tpu",
        "python",
        "validate_tpu_runtime.py",
        "--format",
        args.format,
        "--attempt",
        str(args.attempt),
    ]
    redacted = shlex.join(command)
    print(redacted, flush=True)
    if args.dry_run:
        return
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth is None:
            raise ValueError("W&B credential is missing")
        key = auth[2]
    database = args.database.resolve()
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA foreign_keys=ON")
        if db.execute("SELECT 1 FROM dispatches WHERE active=1").fetchone():
            raise ValueError(
                "The first-validation gate permits only one active TPU job"
            )
        if db.execute("SELECT 1 FROM trials WHERE trial_id=?", (name,)).fetchone():
            raise ValueError(
                "Attempt already exists; inspect its result before retrying"
            )
        db.execute(
            "INSERT INTO trials(trial_id,env_json) VALUES(?,?)",
            (
                name,
                json.dumps({"kind": "validation", "format": args.format, "steps": 2}),
            ),
        )
        db.execute(
            "INSERT INTO runs(regional_run_id,trial_id,region,wandb_run_id,checkpoint_root) VALUES(?,?,?,?,?)",
            (name, name, "us-east5", name, f"{ROOT}/checkpoints/{name}"),
        )
        submitted = datetime.now(UTC).isoformat()
        db.execute(
            "INSERT INTO events(recorded_at,kind,regional_run_id,detail) VALUES(?,?,?,?)",
            (
                submitted,
                "validation_submission_requested",
                name,
                json.dumps({"job_id": job_id, "command": redacted}),
            ),
        )
    command[command.index("<redacted>")] = key
    result = subprocess.run(
        command, cwd=experiment, capture_output=True, text=True, check=False
    )
    output = (result.stderr + result.stdout).replace(key, "<redacted>")
    print(output, flush=True)
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute(
            "INSERT INTO events(recorded_at,kind,regional_run_id,detail) VALUES(?,?,?,?)",
            (
                datetime.now(UTC).isoformat(),
                "validation_submission_result",
                name,
                json.dumps({"returncode": result.returncode, "output": output}),
            ),
        )
        if result.returncode == 0 and job_id in result.stdout:
            db.execute(
                "INSERT INTO dispatches(dispatch_id,regional_run_id,attempt,iris_job_id,tpu_slice,chips,priority_band,command_redacted,submitted_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    job,
                    name,
                    args.attempt,
                    job_id,
                    "v5p-8",
                    4,
                    "batch",
                    redacted,
                    submitted,
                ),
            )
            db.execute(
                "UPDATE runs SET status='submitted' WHERE regional_run_id=?", (name,)
            )
            db.execute("UPDATE trials SET status='submitted' WHERE trial_id=?", (name,))
    if result.returncode != 0 or job_id not in result.stdout:
        raise RuntimeError(
            "Validation submission failed or returned an unexpected identity; inspect Iris before retrying"
        )


if __name__ == "__main__":
    main()
