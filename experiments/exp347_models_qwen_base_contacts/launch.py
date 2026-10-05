"""Submit exactly one explicitly selected H100 training gang."""

import argparse
import json
import netrc
import os
import shlex
import sqlite3
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from common import FORMATS, MODELS, ROOT, TOKEN_BUDGET, run_name

IMAGE = "pytorch/pytorch@sha256:b574d4ccf6d8856a5d87dcadc667aa4f95dc18d337ef3a28d02b7b01897d7081"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=MODELS, required=True)
    parser.add_argument("--format", choices=FORMATS, required=True)
    parser.add_argument(
        "--cluster", choices=["cw-us-east-02a", "cw-rno2a"], default="cw-us-east-02a"
    )
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--cpus", type=int, default=32)
    parser.add_argument("--nodes", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--timeout-days", type=int, default=14)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume-check", action="store_true")
    parser.add_argument("--tokens", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--data")
    parser.add_argument("--initialize-from")
    parser.add_argument("--eval-every-tokens", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--marin-checkout", type=Path, default=Path("/home/bizon/git/marin")
    )
    parser.add_argument("--database", type=Path, required=True)
    args = parser.parse_args()
    if args.cpus < 32:
        raise ValueError("Eight ranks need at least four CPU threads each")
    if args.nodes != 1 and not args.run_id:
        raise ValueError("A new multi-node profile needs its own run identity")
    name = args.run_id or run_name(args.size, args.format, args.smoke)
    if (
        args.data or args.initialize_from or args.eval_every_tokens
    ) and not args.run_id:
        raise ValueError("Scale-up settings require a distinct run identity")
    job = f"{name}-a{args.attempt:02d}"
    tokens = args.tokens or (100_000 if args.smoke else TOKEN_BUDGET)
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
        "run",
        "--target-cluster",
        args.cluster,
        "--priority",
        "batch",
        "--user",
        "timodonnell",
        "--job-name",
        job,
        "--no-wait",
        "--enable-extra-resources",
        "--gpu",
        "H100x8",
        "--cpu",
        str(args.cpus),
        "--memory",
        "512GB",
        "--disk",
        "256GB",
        "--no-sync",
        "--timeout",
        str(args.timeout_days * 86400),
        "--task-image",
        IMAGE,
        "-e",
        "EXP347_CLUSTER",
        args.cluster,
        "-e",
        "WANDB_API_KEY",
        "<redacted>",
        "--",
        "bash",
        "gpu_bootstrap.sh",
        "--size",
        args.size,
        "--format",
        args.format,
        "--tokens",
        str(tokens),
    ]
    if args.smoke:
        command += ["--smoke", "--eval-documents", "8", "--accumulation", "1"]
    if args.accumulation is not None:
        command += ["--accumulation", str(args.accumulation)]
    if args.nodes != 1:
        command[command.index("--task-image") : command.index("--task-image")] = [
            "--replicas",
            str(args.nodes),
        ]
    if args.resume_check:
        command.append("--resume-check")
    if args.run_id:
        command += ["--run-id", args.run_id]
    if args.data:
        command += ["--data", args.data]
    if args.initialize_from:
        command += ["--initialize-from", args.initialize_from]
    if args.eval_every_tokens:
        command += ["--eval-every-tokens", str(args.eval_every_tokens)]
    redacted = shlex.join(command)
    print(redacted, flush=True)
    if args.dry_run:
        return
    with sqlite3.connect(args.database) as db:
        if db.execute(
            "SELECT 1 FROM dispatches WHERE trial_id=? AND active=1", (name,)
        ).fetchone():
            raise ValueError("Trial already has an active dispatch; reconcile it first")
        configuration = {
            "size": args.size,
            "format": args.format,
            "tokens": tokens,
            "smoke": args.smoke,
        }
        if args.accumulation is not None:
            configuration["accumulation"] = args.accumulation
        if args.run_id:
            configuration.update(
                {
                    "run_id": args.run_id,
                    "data": args.data,
                    "initialize_from": args.initialize_from,
                    "eval_every_tokens": args.eval_every_tokens,
                    "nodes": args.nodes,
                }
            )
        config = json.dumps(configuration)
        db.execute(
            "INSERT OR IGNORE INTO trials(trial_id,env_json,wandb_run_id,checkpoint_root) VALUES(?,?,?,?)",
            (name, config, name, f"{ROOT}/checkpoints/{name}"),
        )
        old = db.execute(
            "SELECT env_json FROM trials WHERE trial_id=?", (name,)
        ).fetchone()[0]
        if old != config and not args.smoke:
            raise ValueError("Production trial configuration changed across dispatches")
        key = os.environ.get("WANDB_API_KEY")
        if not key:
            auth = netrc.netrc().authenticators("api.wandb.ai")
            if auth is None:
                raise ValueError("W&B credential missing")
            key = auth[2]
        command[command.index("<redacted>")] = key
        submitted = datetime.now(UTC).isoformat()
        result = subprocess.run(
            command,
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=False,
        )
        print(result.stderr.replace(key, "<redacted>"), flush=True)
        print(result.stdout.replace(key, "<redacted>"), flush=True)
        if result.returncode:
            raise RuntimeError(f"Iris submission failed with exit {result.returncode}")
        job_id = f"/timodonnell/{job}"
        if job_id not in result.stdout:
            raise ValueError(
                "Submission did not return the expected job ID; inspect Iris before retrying"
            )
        db.execute(
            """INSERT INTO dispatches(dispatch_id,trial_id,attempt,iris_job_id,cluster,gpu_variant,
                      nodes,gpus,priority_band,command_redacted,submitted_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (
                job,
                name,
                args.attempt,
                job_id,
                args.cluster,
                "H100",
                args.nodes,
                8 * args.nodes,
                "batch",
                redacted,
                submitted,
            ),
        )
        db.execute(
            "INSERT INTO events(recorded_at,kind,trial_id,dispatch_id,detail) VALUES(?,?,?,?,?)",
            (
                submitted,
                "dispatch_submitted",
                name,
                job,
                "Explicit operator-selected training dispatch",
            ),
        )


if __name__ == "__main__":
    main()
