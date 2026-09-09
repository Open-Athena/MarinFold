# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a pinned, region-local TPU pilot or the three production phases."""

import argparse
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

import wandb
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client

from .inputs import EXPERIMENT, ROOT, inspect_cache, source_identity

PROJECT = "experiments/exp279_models_exact_soft_contact_targets"
MODULE = "experiments.exp279_models_exact_soft_contact_targets"
REGIONS = {
    "us-east1": ("marin-us-east1", "us-east1-d"),
    "us-east5": ("marin-us-east5", "us-east5-b"),
}


def worker_request(
    *,
    name: str,
    arm: str,
    run_name: str,
    output: str,
    region: str,
    tpu: str,
    stop_after: int | None,
    env: dict[str, str],
) -> JobRequest:
    """Build one full-model gang, including all hosts and batch priority."""
    command = [
        f"{PROJECT}/.venv/bin/python",
        "-m",
        f"{MODULE}.train",
        "--manifest",
        "inputs.json",
        "--arm",
        arm,
        "--run-name",
        run_name,
        "--output",
        output,
        "--resume-latest",
        "--run",
    ]
    if stop_after is not None:
        command += ["--stop-after", str(stop_after)]
    return JobRequest(
        name=name,
        entrypoint=Entrypoint.from_binary(
            "bash",
            [
                "-c",
                "set -e\n"
                f"uv sync --locked --project {PROJECT} --extra tpu --no-dev\n"
                + "exec "
                + shlex.join(command),
            ],
        ),
        resources=ResourceConfig.with_tpu(
            tpu, zone=REGIONS[region][1], regions=[region], preemptible=True
        ),
        environment=create_environment(env_vars=env, setup_scripts=[]),
        priority=3,
        max_retries_failure=0,
        max_retries_preemption=100,
    )


def stage_bundle(destination: Path, manifest: dict) -> None:
    """Copy exactly the source files hashed in the manifest, plus runtime metadata."""
    paths = list(EXPERIMENT.glob("*.py")) + [
        EXPERIMENT / "pyproject.toml",
        EXPERIMENT / "uv.lock",
        EXPERIMENT / "README.md",
        ROOT / "experiments/exp232_sweep_cv1_decontam/training_contract.py",
        ROOT / "scripts/history.py",
        ROOT / "scripts/_lib.py",
    ]
    for source in paths:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (destination / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("ce", "soft"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--region", choices=REGIONS, default="us-east1")
    parser.add_argument(
        "--tpu", choices=("v6e-8", "v6e-16", "v6e-32", "v6e-64"), default="v6e-32"
    )
    parser.add_argument("--pilot-updates", type=int)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument(
        "--resume-record",
        type=Path,
        help="Reuse a pilot's frozen manifest and placement",
    )
    args = parser.parse_args()
    if not args.run_name.startswith(f"exp279-{args.arm}-"):
        parser.error("Run name must identify the arm")
    if args.pilot_updates is not None and not 0 < args.pilot_updates <= 217801:
        parser.error("Pilot must stop within the base phase")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("Commit the launch source before freezing the worker bundle")
    bucket, _ = REGIONS[args.region]
    output = f"gs://{bucket}/protein-structure/MarinFold/exp279"
    cache_root = f"gs://{bucket}/protein-structure/MarinFold/exp232_train_trc/tokenized"
    manifest = {
        "source": source_identity(),
        "inputs": {
            name: inspect_cache(f"{cache_root}/{suffix}", name)
            for name, suffix in {
                "afdb": "contacts_v1/afdb/2026.08.14/train",
                "esm": "contacts_v1/esm/2026.08.14/train",
                "val": "contacts-v1-val/2026.07.25/validation",
            }.items()
        },
    }
    if args.resume_record is not None:
        previous = json.loads(args.resume_record.read_text())
        for key, value in {
            "run_name": args.run_name,
            "output": output,
            "region": args.region,
            "tpu": args.tpu,
        }.items():
            if previous[key] != value:
                raise ValueError(f"Pilot {key} differs from this launch")
        # A later history-only commit must not change the identity of already
        # running code. Actual runtime source, lock, packages and inputs must match.
        manifest["source"]["git_sha"] = previous["manifest"]["source"]["git_sha"]
        if manifest != previous["manifest"]:
            raise ValueError("Pilot source, dependencies or input ledgers changed")
    credentials = wandb.Api().api_key
    if not credentials:
        raise ValueError("A W&B credential is required for production training")
    env = {
        "EXP279_GIT_SHA": manifest["source"]["git_sha"],
        "WANDB_API_KEY": credentials,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "MARIN_PREFIX": output,
        "PYTHONUNBUFFERED": "1",
    }
    # The staged source snapshot survives local edits and contains no workstation
    # credentials, .git internals, unrelated experiments or model weights.
    with tempfile.TemporaryDirectory(prefix="exp279-bundle-") as directory:
        stage = Path(directory)
        stage_bundle(stage, manifest)
        if args.pilot_updates is not None:
            request = worker_request(
                name=args.job_name,
                arm=args.arm,
                run_name=args.run_name,
                output=output,
                region=args.region,
                tpu=args.tpu,
                stop_after=args.pilot_updates,
                env=env,
            )
        else:
            command = [
                f"{PROJECT}/.venv/bin/python",
                "-m",
                f"{MODULE}.production",
                "--arm",
                args.arm,
                "--run-name",
                args.run_name,
                "--output",
                output,
                "--region",
                args.region,
                "--tpu",
                args.tpu,
                "--job-prefix",
                args.job_name,
            ]
            request = JobRequest(
                name=args.job_name,
                entrypoint=Entrypoint.from_binary(
                    "bash",
                    [
                        "-c",
                        "set -e\n"
                        f"uv sync --locked --project {PROJECT} --no-dev\n"
                        + "exec "
                        + shlex.join(command),
                    ],
                ),
                resources=ResourceConfig.with_cpu(
                    cpu=2, ram="8g", regions=[args.region], preemptible=False
                ),
                environment=create_environment(env_vars=env, setup_scripts=[]),
                priority=3,
            )
        print(
            json.dumps(
                {
                    "job": args.job_name,
                    "run": args.run_name,
                    "arm": args.arm,
                    "tpu": args.tpu,
                    "region": args.region,
                    "pilot_updates": args.pilot_updates,
                    "source": manifest["source"],
                    "output": output,
                },
                indent=2,
            ),
            flush=True,
        )
        with open_iris_client(cluster_name="marin", workspace=stage) as iris:
            handle = FrayIrisClient.from_iris_client(iris).submit(
                request, adopt_existing=False
            )
            record = {
                "job_id": handle.job_id,
                "run_name": args.run_name,
                "output": output,
                "region": args.region,
                "tpu": args.tpu,
                "pilot_updates": args.pilot_updates,
                "manifest": manifest,
            }
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text(json.dumps(record, indent=2) + "\n")
            print(handle.job_id, flush=True)


if __name__ == "__main__":
    main()
