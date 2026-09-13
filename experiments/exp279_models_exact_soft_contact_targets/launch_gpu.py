# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the exact experiment on CoreWeave H100s using the existing S3 caches."""

import argparse
import base64
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

import wandb
import fsspec.config
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client
from iris.cluster.setup_scripts import cuda_toolchain_setup_script

from .inputs import ROOT, inspect_cache, source_identity
from .launch import MODULE, PROJECT, stage_bundle

CLUSTER = "cw-us-east-02a"
OUTPUT = "s3://marin-us-east-02a/MarinFold/exp279"


def cuda_setup() -> str:
    """Use Iris's CUDA precedence repair with the locked direct cuDNN wheel."""
    script = cuda_toolchain_setup_script()
    install = '    uv pip install --python "$IRIS_VENV/bin/python" \\\n'
    package = '      "$_cuda13_package==$_cuda13_version"'
    if script.count(install) != 1 or script.count(package) != 1:
        raise ValueError("Pinned Iris CUDA setup changed")
    wheel = "https://pypi.nvidia.com/nvidia-cudnn-cu13/nvidia_cudnn_cu13-9.26.0.17.dev59162438-py3-none-manylinux_2_27_x86_64.whl"
    choose = f"""    if [ "$_cuda13_package" = "nvidia-cudnn-cu13" ]; then
      _cuda13_spec={shlex.quote(wheel)}
    else
      _cuda13_spec="$_cuda13_package==$_cuda13_version"
    fi
"""
    return script.replace(install, choose + install).replace(
        package, '      "$_cuda13_spec"'
    )


def gpu_worker_request(
    *,
    name: str,
    arm: str,
    run_name: str,
    nodes: int,
    stop_after: int | None,
    env: dict[str, str],
) -> JobRequest:
    """Request complete H100 nodes at batch priority, with pinned CUDA setup."""
    if nodes not in (1, 2, 4):
        raise ValueError("Use a validated 1/2/4-node H100 gang")
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
        OUTPUT,
        "--resume-latest",
        "--run",
    ]
    if stop_after is not None:
        command += ["--stop-after", str(stop_after)]
    # Keep setup and execution in one shell. The upstream setup may exit early
    # when no CUDA binaries are found, so isolate it in a checked subshell.
    script = (
        f'set -e\nexport UV_PROJECT_ENVIRONMENT="$PWD/{PROJECT}/.venv"\n'
        f"uv sync --locked --project {PROJECT} --extra gpu --no-dev\n"
        f'export IRIS_VENV="$PWD/{PROJECT}/.venv"\n'
        'export PATH="$IRIS_VENV/bin:$PATH"\n'
        + "(\n"
        + cuda_setup()
        + ")\n"
        + "exec "
        + shlex.join(command)
    )
    return JobRequest(
        name=name,
        entrypoint=Entrypoint.from_binary("bash", ["-c", script]),
        resources=ResourceConfig.with_gpu(
            "H100", count=8, replicas=nodes, cpu=32, ram="256g", disk="80g"
        ),
        environment=create_environment(env_vars=env, setup_scripts=[]),
        priority=3,
        max_retries_failure=0,
        max_retries_preemption=100,
    )


def configure_local_s3() -> None:
    """Use this cluster's storage credentials for small submit-side ledger reads."""
    secret = json.loads(
        subprocess.check_output(
            [
                "kubectl",
                "--kubeconfig",
                str(Path.home() / ".kube/coreweave-iris"),
                "--context",
                "marin-gpu_US-EAST-02A",
                "-n",
                "iris",
                "get",
                "secret",
                "iris-task-env",
                "-o",
                "json",
            ]
        )
    )["data"]
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        os.environ[name] = base64.b64decode(secret[name]).decode()
    config = json.loads(base64.b64decode(secret["FSSPEC_S3"]))
    config["endpoint_url"] = "https://cwobject.com"
    config.setdefault("config_kwargs", {})["s3"] = {"addressing_style": "virtual"}
    os.environ["FSSPEC_S3"] = json.dumps(config)
    fsspec.config.set_conf_env(fsspec.config.conf)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("ce", "soft"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--nodes", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--pilot-updates", type=int)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--resume-record", type=Path)
    args = parser.parse_args()
    if not args.run_name.startswith(f"exp279-{args.arm}-"):
        parser.error("Run name must identify the arm")
    if args.pilot_updates is not None and not 0 < args.pilot_updates <= 217801:
        parser.error("Pilot must stop within the base phase")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("Commit the launch source before freezing the bundle")
    configure_local_s3()
    cache_root = "s3://marin-us-east-02a/MarinFold"
    manifest = {
        "source": source_identity(),
        "inputs": {
            name: inspect_cache(f"{cache_root}/{suffix}", name)
            for name, suffix in {
                "afdb": "exp232_sweep_cv1_decontam/tokenized/contacts_v1/afdb/2026.08.14/train",
                "esm": "exp232_sweep_cv1_decontam/tokenized/contacts_v1/esm/2026.08.14/train",
                "val": "exp154_qwen_contacts_v1/tokenized/contacts-v1-val/2026.07.25/validation",
            }.items()
        },
    }
    if args.resume_record:
        previous = json.loads(args.resume_record.read_text())
        for key, value in {
            "run_name": args.run_name,
            "cluster": CLUSTER,
            "nodes": args.nodes,
            "output": OUTPUT,
        }.items():
            if previous[key] != value:
                raise ValueError(f"Pilot {key} differs")
        manifest["source"]["git_sha"] = previous["manifest"]["source"]["git_sha"]
        if manifest != previous["manifest"]:
            raise ValueError("Pilot source, dependencies or input ledgers changed")
    credentials = wandb.Api().api_key
    if not credentials:
        raise ValueError("W&B credential required")
    env = {
        "EXP279_GIT_SHA": manifest["source"]["git_sha"],
        "WANDB_API_KEY": credentials,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "MARIN_PREFIX": OUTPUT,
        "PYTHONUNBUFFERED": "1",
    }
    with tempfile.TemporaryDirectory(prefix="exp279-gpu-bundle-") as directory:
        stage = Path(directory)
        stage_bundle(stage, manifest)
        if args.pilot_updates is not None:
            request = gpu_worker_request(
                name=args.job_name,
                arm=args.arm,
                run_name=args.run_name,
                nodes=args.nodes,
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
                OUTPUT,
                "--region",
                CLUSTER,
                "--tpu",
                "H100",
                "--nodes",
                str(args.nodes),
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
                        f'export UV_PROJECT_ENVIRONMENT="$PWD/{PROJECT}/.venv"\n'
                        f"uv sync --locked --project {PROJECT} --no-dev\n"
                        + "exec "
                        + shlex.join(command),
                    ],
                ),
                resources=ResourceConfig.with_cpu(cpu=2, ram="8g"),
                environment=create_environment(env_vars=env, setup_scripts=[]),
                priority=3,
            )
        print(
            json.dumps(
                {
                    "job": args.job_name,
                    "run": args.run_name,
                    "nodes": args.nodes,
                    "source": manifest["source"],
                    "output": OUTPUT,
                },
                indent=2,
            ),
            flush=True,
        )
        with open_iris_client(cluster_name=CLUSTER, workspace=stage) as iris:
            handle = FrayIrisClient.from_iris_client(iris).submit(
                request, adopt_existing=False
            )
            record = {
                "job_id": handle.job_id,
                "run_name": args.run_name,
                "cluster": CLUSTER,
                "nodes": args.nodes,
                "output": OUTPUT,
                "pilot_updates": args.pilot_updates,
                "manifest": manifest,
            }
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text(json.dumps(record, indent=2) + "\n")
            print(handle.job_id, flush=True)


if __name__ == "__main__":
    main()
