"""Launch Torch ranks across an explicitly sized Iris H100 gang."""

import hashlib
import socket
import subprocess
import sys
import time

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info


def main() -> None:
    info = get_job_info()
    if info is None:
        raise RuntimeError("Training must be launched inside Iris")
    command = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=8"]
    if info.num_tasks == 1:
        subprocess.run([*command, "--standalone", *sys.argv[1:]], check=True)
        return
    ctx = get_iris_ctx()
    if ctx is None:
        raise RuntimeError("Missing Iris registry context")
    name = "exp347-torch"
    endpoint_id = None
    if info.task_index == 0:
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        address = f"{info.advertise_host}:{port}"
        endpoint_id = ctx.registry.register(name, address)
    else:
        deadline = time.monotonic() + 600
        while True:
            try:
                resolved = ctx.resolver.resolve(name)
                if not resolved.is_empty:
                    address = resolved.first().url
                    break
            except ConnectError as error:
                if error.code not in {Code.NOT_FOUND, Code.UNAVAILABLE}:
                    raise
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "No training coordinator registered within ten minutes"
                )
            time.sleep(2)
    try:
        # CoreWeave hostnames resolve to loopback in these pods. Static rendezvous
        # keeps the registered routable address as MASTER_ADDR, rather than letting
        # elastic rendezvous publish a second hostname-derived worker endpoint.
        # Zero worker restarts leave whole-gang preemption and fixed world size to Iris.
        subprocess.run(
            [
                *command,
                f"--nnodes={info.num_tasks}",
                f"--node_rank={info.task_index}",
                "--rdzv_backend=static",
                f"--rdzv_endpoint={address}",
                "--rdzv_id="
                + hashlib.sha256(str(info.job_id).encode()).hexdigest()[:16],
                "--max_restarts=0",
                *sys.argv[1:],
            ],
            check=True,
        )
    finally:
        if endpoint_id is not None:
            ctx.registry.unregister(endpoint_id)


if __name__ == "__main__":
    main()
