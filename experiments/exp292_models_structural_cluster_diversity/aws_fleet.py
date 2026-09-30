"""Shared EC2 fan-out control for the exp292 production launchers.

The account allows a fixed number of on-demand vCPUs in the ``m7i`` bucket, so
a 256-shard stage cannot be submitted in one go. Two details make a naive loop
fail in ways that are easy to misread:

* An instance keeps its vCPU reservation until it is **fully terminated**, so
  workers that are merely shutting down still count against the limit. Sizing a
  wave against ``pending`` plus ``running`` alone overshoots.
* ``RunInstances`` refuses the whole call when the limit is reached, which in a
  bare loop aborts the launcher and strands the workers it already started with
  no record of them.

``drain`` handles both: it counts every quota-holding state, treats a refusal as
backpressure to retry on the next wave, and advances the queue by what actually
launched rather than by what it intended to launch.
"""

import time
from collections.abc import Callable, Sequence

from botocore.exceptions import ClientError

ACCOUNT_VCPU_LIMIT = 445
INSTANCE_VCPUS = {"m7i.4xlarge": 16, "m7i.8xlarge": 32}
HOLDING_STATES = ("pending", "running", "shutting-down", "stopping", "rebooting")
EXPERIMENT_TAG = "exp292"


def max_concurrent(instance_type: str) -> int:
    """Return how many workers of this type fit inside the account limit."""
    if instance_type not in INSTANCE_VCPUS:
        raise ValueError(f"Unknown vCPU count for instance type {instance_type}")
    return ACCOUNT_VCPU_LIMIT // INSTANCE_VCPUS[instance_type]


def live_instances(ec2) -> int:
    """Count experiment workers still holding vCPU capacity."""
    total = 0
    for page in ec2.get_paginator("describe_instances").paginate(
        Filters=[
            {"Name": "instance-state-name", "Values": list(HOLDING_STATES)},
            {"Name": "tag:Experiment", "Values": [EXPERIMENT_TAG]},
        ]
    ):
        total += sum(len(item["Instances"]) for item in page["Reservations"])
    return total


def drain(
    ec2,
    shards: Sequence[str],
    launch_one: Callable[[str], str],
    *,
    concurrency: int,
    poll_seconds: int = 30,
    on_launch: Callable[[str, str], None] | None = None,
) -> None:
    """Launch every shard in waves that respect the account's vCPU limit.

    ``launch_one`` starts one worker and returns its instance ID; ``on_launch``
    is called with ``(shard, instance_id)`` immediately afterwards so the caller
    can persist the record before the next request is made.
    """
    pending = list(shards)
    while pending:
        free = concurrency - live_instances(ec2)
        if free <= 0:
            time.sleep(poll_seconds)
            continue
        launched = 0
        for shard in pending[:free]:
            try:
                instance_id = launch_one(shard)
            except ClientError as error:
                if error.response["Error"]["Code"] != "VcpuLimitExceeded":
                    raise
                # Capacity the describe call had not seen yet, usually workers
                # still shutting down. Retry this shard on the next wave.
                print(
                    f"vcpu limit reached, {len(pending) - launched} shards still queued",
                    flush=True,
                )
                break
            launched += 1
            if on_launch is not None:
                on_launch(shard, instance_id)
        pending = pending[launched:]
        if pending:
            time.sleep(poll_seconds)
