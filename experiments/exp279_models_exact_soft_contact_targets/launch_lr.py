"""Submit one matched 32-H100 LR continuation through the Iris federation."""

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import wandb
from fray.iris_backend import (
    convert_constraints,
    convert_entrypoint,
    convert_environment,
    convert_resources,
    resolve_coscheduling,
)
from fray.types import Entrypoint
from iris.cli.connect import open_iris_client
from iris.rpc import job_pb2
from rigging.filesystem.storage_path import StoragePath

from .inputs import ROOT, source_identity, verify_manifest
from .launch import MODULE, PROJECT, stage_bundle
from .launch_gpu import CLUSTER, OUTPUT, configure_local_s3, gpu_worker_request
from .lr_trial import PARENT_REVISION, PARENT_RUN, RATES, START, STOP, validate_parent


def fork_contract() -> dict:
    """Freeze the permanent parent's provenance without copying any weights."""
    root = OUTPUT + f"/checkpoints/{PARENT_RUN}"
    parent = json.loads(StoragePath(root + "/experiment.json").read_text())
    checkpoint = root + f"/step-{START - 1}"
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", PARENT_REVISION, "--", PROJECT],
        cwd=ROOT,
        text=True,
    ).splitlines()
    paths = sorted(
        name
        for name in names
        if Path(name).parent == Path(PROJECT) and name.endswith(".py")
    )
    paths += [
        "experiments/exp232_sweep_cv1_decontam/training_contract.py",
        "scripts/history.py",
        "scripts/_lib.py",
    ]
    return {
        "parent_checkpoint": checkpoint,
        "parent_identity": parent,
        "parent_source_paths": paths,
        "checkpoint_hashes": {
            name: hashlib.sha256(
                StoragePath(checkpoint + "/" + name).read_bytes()
            ).hexdigest()
            for name in ("metadata.json", "manifest.json")
        },
    }


def trial_request(
    *,
    trial: str,
    run_name: str,
    job_name: str,
    env: dict[str, str],
    stop_after: int = STOP,
    cluster: str = CLUSTER,
):
    """Reuse the validated GPU setup and enforce the matched gang geometry."""
    if cluster not in (CLUSTER, "cw-rno2a"):
        raise ValueError("The matched trial requires a supported H100 target")
    request = gpu_worker_request(
        name=job_name,
        arm="soft",
        run_name=run_name,
        nodes=4,
        stop_after=None,
        env=env,
    )
    binary = request.entrypoint.binary_entrypoint
    if binary is None:
        raise ValueError("GPU worker must have a shell entry point")
    script = binary.args[-1]
    setup, separator, _ = script.rpartition("exec ")
    if not separator:
        raise ValueError("GPU setup no longer has its expected exec boundary")
    command = [
        f"{PROJECT}/.venv/bin/python",
        "-m",
        f"{MODULE}.lr_trial",
        "--manifest",
        "inputs.json",
        "--contract",
        "fork.json",
        "--trial",
        trial,
        "--run-name",
        run_name,
        "--output",
        OUTPUT,
        "--stop-after",
        str(stop_after),
    ]
    return replace(
        request,
        resources=replace(request.resources, target_cluster=cluster),
        entrypoint=Entrypoint.from_binary(
            "bash", ["-c", setup + "exec " + shlex.join(command)]
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", choices=RATES, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--resume-record", type=Path)
    parser.add_argument("--stop-after", type=int, default=STOP)
    parser.add_argument("--cluster", choices=(CLUSTER, "cw-rno2a"), default=CLUSTER)
    args = parser.parse_args()
    if not args.run_name.startswith(f"exp279-soft-{args.trial}-"):
        parser.error("Run name must identify the trial")
    if not START < args.stop_after <= STOP:
        parser.error("Stop is outside the trial window")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise ValueError("Commit the runtime before submitting the frozen bundle")
    configure_local_s3()
    contract = fork_contract()
    manifest = {
        "source": source_identity(),
        "inputs": contract["parent_identity"]["manifest"]["inputs"],
    }
    verify_manifest(manifest)
    validate_parent(contract, manifest)
    if args.resume_record:
        previous = json.loads(args.resume_record.read_text())
        if (previous["trial"], previous["run_name"]) != (args.trial, args.run_name):
            raise ValueError("Cannot resume another trial")
        manifest["source"]["git_sha"] = previous["manifest"]["source"]["git_sha"]
        if manifest != previous["manifest"] or contract != previous["contract"]:
            raise ValueError("Trial runtime, inputs or lineage changed")
    credential = wandb.Api().api_key
    if not credential:
        raise ValueError("W&B credential required")
    env = {
        "EXP279_GIT_SHA": manifest["source"]["git_sha"],
        "WANDB_API_KEY": credential,
        "WANDB_ENTITY": "open-athena",
        "WANDB_PROJECT": "MarinFold",
        "MARIN_PREFIX": OUTPUT,
        "PYTHONUNBUFFERED": "1",
    }
    request = trial_request(
        trial=args.trial,
        run_name=args.run_name,
        job_name=args.job_name,
        env=env,
        stop_after=args.stop_after,
        cluster=args.cluster,
    )
    print(
        json.dumps(
            {
                "trial": args.trial,
                "lr": RATES[args.trial],
                "start": START,
                "stop": args.stop_after,
                "parent": contract["parent_checkpoint"],
                "source": manifest["source"],
                "target": args.cluster,
                "gpus": 32,
                "user": "bizon",
                "priority": "batch",
            },
            indent=2,
        ),
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="exp279-lr-") as directory:
        stage = Path(directory)
        stage_bundle(stage, manifest)
        (stage / "fork.json").write_text(json.dumps(contract, indent=2) + "\n")
        with open_iris_client(cluster_name="marin", workspace=stage) as iris:
            job = iris.submit(
                name=request.name,
                entrypoint=convert_entrypoint(request.entrypoint),
                resources=convert_resources(request.resources),
                environment=convert_environment(
                    request.environment, request.resources.device
                ),
                constraints=convert_constraints(request.resources),
                coscheduling=resolve_coscheduling(request.resources, 4),
                replicas=4,
                max_retries_failure=request.max_retries_failure,
                max_retries_preemption=request.max_retries_preemption,
                priority_band=job_pb2.PRIORITY_BAND_BATCH,
                user="bizon",
            )
            record = {
                "trial": args.trial,
                "run_name": args.run_name,
                "job_id": str(job.job_id),
                "cluster": args.cluster,
                "gpus": 32,
                "stop_after": args.stop_after,
                "manifest": manifest,
                "contract": contract,
            }
            args.record.parent.mkdir(parents=True, exist_ok=True)
            args.record.write_text(json.dumps(record, indent=2) + "\n")
            print(str(job.job_id), flush=True)


if __name__ == "__main__":
    main()
