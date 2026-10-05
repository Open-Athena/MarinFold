"""Register an evaluation run and submit its checkpoint-local persistent driver."""

import argparse
import json
import netrc
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import wandb
from launch_eval import HERE, build_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-run", required=True)
    parser.add_argument("--evaluation-run", required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--timeout-days", type=int, default=14)
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--reference-e8", action="store_true")
    args = parser.parse_args()
    job = f"{args.evaluation_run}-a{args.attempt:02d}"
    repo = HERE.parents[1]
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        auth = netrc.netrc().authenticators("api.wandb.ai")
        if auth is None:
            raise ValueError("Missing W&B credentials")
        key = auth[2]
    with tempfile.TemporaryDirectory(prefix="exp347-eval-driver-") as directory:
        destination = Path(directory)
        build_bundle(destination, args.targets)
        for name in [
            "periodic_eval.py",
            "aggregate_eval.py",
            "pyproject.toml",
            "uv.lock",
        ]:
            shutil.copy2(HERE / name, destination / name)
        if args.request_file:
            shutil.copy2(args.request_file, destination / "request.json")
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
            "--target-cluster",
            "cw-us-east-02a",
            "--priority",
            "batch",
            "--user",
            "timodonnell",
            "--job-name",
            job,
            "--no-wait",
            "--enable-extra-resources",
            "--cpu",
            "2",
            "--memory",
            "8GB",
            "--disk",
            "8GB",
            "--timeout",
            str(args.timeout_days * 86400),
            "-e",
            "WANDB_API_KEY",
            "<redacted>",
            "--",
            "uv",
            "run",
            "--locked",
            "--extra",
            "driver",
            "python",
            "periodic_eval.py",
            "--training-run",
            args.training_run,
            "--evaluation-run",
            args.evaluation_run,
            "--shards",
            str(args.shards),
        ]
        if args.request_file:
            command += ["--request-file", "request.json"]
        if args.reference_e8:
            command += ["--reference-e8"]
        with wandb.init(
            entity="open-athena",
            project="MarinFold",
            id=args.evaluation_run,
            name=args.evaluation_run,
            resume="allow",
            job_type="eval",
            group="exp347-qwen-base-contacts",
            config={
                "training_run": args.training_run,
                "shards": args.shards,
                "universe": "legacy-e8-reference" if args.reference_e8 else "eval-val",
            },
        ) as run:
            history_command = [
                "uv",
                "run",
                "--project",
                str(repo / "scripts"),
                "python",
                str(repo / "scripts/history.py"),
            ]
            if args.attempt > 1:
                subprocess.run(
                    [*history_command, "add-iris-job", run.name, f"/timodonnell/{job}"],
                    check=True,
                )
            else:
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
                        HERE.name,
                        "--kind",
                        "evals",
                        "--short",
                        "Canonical 100-rollout R-precision; E8 reference"
                        if args.reference_e8
                        else "Periodic Qwen checkpoint eval-val R-precision",
                        "--iris-jobs",
                        f"/timodonnell/{job}",
                    ],
                    check=True,
                )
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
        print(shlex.join(command), flush=True)
        command[command.index("<redacted>")] = key
        result = subprocess.run(
            command, cwd=destination, capture_output=True, text=True, check=False
        )
        print(result.stdout.replace(key, "<redacted>"), flush=True)
        print(result.stderr.replace(key, "<redacted>"), flush=True)
        if result.returncode:
            raise RuntimeError(f"Iris submission failed: {result.returncode}")
        print(
            json.dumps(
                {"job": f"/timodonnell/{job}", "evaluation_run": args.evaluation_run}
            )
        )


if __name__ == "__main__":
    main()
