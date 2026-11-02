# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit the shared-weight RNO2A job with both kinds of retry disabled."""

import argparse
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from rigging.timing import Duration


def main() -> None:
    """Use the explicit API because CLI --max-retries leaves preemption retries on."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    with open_iris_client(cluster_name="marin", workspace=args.workspace) as client:
        job = client.submit(
            name=args.name,
            entrypoint=Entrypoint.from_command(
                "bash",
                "bootstrap.sh",
                "--shared-rno",
                "--out",
                args.out,
            ),
            resources=ResourceSpec(
                cpu=32, memory="256GB", disk="128GB", device=gpu_device("h100", 8)
            ),
            environment=EnvironmentSpec(setup_scripts=[]),
            constraints=[
                Constraint.create(key="cluster", op=ConstraintOp.EQ, value="cw-rno2a")
            ],
            priority_band=job_pb2.PRIORITY_BAND_BATCH,
            max_retries_failure=0,
            max_retries_preemption=0,
            max_task_failures=0,
            timeout=Duration.from_seconds(14400),
            task_image="vllm/vllm-openai:v0.19.1",
        )
        print(job.job_id, flush=True)


if __name__ == "__main__":
    main()
