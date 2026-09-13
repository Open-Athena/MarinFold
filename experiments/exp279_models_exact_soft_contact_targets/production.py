# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Keep the prescribed phase jobs alive and advance only after a complete checkpoint."""

import argparse
import json
import os

from fray.current_client import current_client
from iris.cluster.client.job_info import get_job_info
from levanter.checkpoint import discover_latest_checkpoint
from rigging.filesystem.storage_path import StoragePath

from .launch import REGIONS, worker_request
from .launch_gpu import CLUSTER, OUTPUT, gpu_worker_request
from .train import phase_for_update


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("arm", "run-name", "output", "region", "tpu", "job-prefix"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--nodes", type=int, default=4)
    args = parser.parse_args()
    if get_job_info() is None:
        raise RuntimeError("The production driver must execute inside Iris")
    gpu = args.region == CLUSTER and args.tpu == "H100"
    if gpu and args.output != OUTPUT:
        raise ValueError("GPU output must use the region-local experiment prefix")
    if not gpu and args.region not in REGIONS:
        raise ValueError("Unsupported region")
    client = current_client()
    root = args.output.rstrip("/") + f"/checkpoints/{args.run_name}"
    env = {
        name: os.environ[name]
        for name in (
            "EXP279_GIT_SHA",
            "WANDB_API_KEY",
            "WANDB_ENTITY",
            "WANDB_PROJECT",
            "MARIN_PREFIX",
            "PYTHONUNBUFFERED",
        )
    }
    previous_update = -1
    while True:
        checkpoint = discover_latest_checkpoint(root)
        update = (
            0
            if checkpoint is None
            else json.loads(StoragePath(checkpoint + "/metadata.json").read_text())[
                "step"
            ]
            + 1
        )
        phase = phase_for_update(update)
        if phase is None:
            print(f"Full recipe completed: {checkpoint}", flush=True)
            return
        if update <= previous_update:
            raise RuntimeError(
                "Training job completed without advancing its checkpoint"
            )
        previous_update = update
        common = dict(
            name=f"{args.job_prefix}-{phase}",
            arm=args.arm,
            run_name=args.run_name,
            stop_after=None,
            env=env,
        )
        if gpu:
            request = gpu_worker_request(**common, nodes=args.nodes)
        else:
            request = worker_request(
                **common, output=args.output, region=args.region, tpu=args.tpu
            )
        handle = client.submit(request)
        print(
            f"Phase {phase}, restored update {update}, job {handle.job_id}", flush=True
        )
        handle.wait()


if __name__ == "__main__":
    main()
