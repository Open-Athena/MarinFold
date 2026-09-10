# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the checkpoint once, then run independent workers in one eight-GPU pod.

This bounded locality fallback copies 5.885 GB once into RNO2A when the checkpoint
region has no free GPUs. All eight workers share that local copy. Automatic pod
retries are disabled to avoid repeating the cross-region transfer unnoticed.
"""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

from conditioning_worker import MODEL_FILES, MODEL_URI, stage_model, write_json


def main() -> None:
    """Wait for every local GPU worker and fail if any worker failed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--stem")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("Expected one to eight local GPU workers")
    started = time.monotonic()
    cache = Path("/tmp/exp254-m2-p06-step145199")
    cached = all(
        (cache / name).exists() and (cache / name).stat().st_size == size
        for name, (size, _) in MODEL_FILES.items()
    )
    model = stage_model(MODEL_URI)
    write_json(
        f"{args.out}/staging-{args.workers}.json",
        {
            "source": MODEL_URI,
            "files": MODEL_FILES,
            "bytes": sum(size for size, _ in MODEL_FILES.values()),
            "stage_seconds": time.monotonic() - started,
            "cache_reused": cached,
            "destination_cluster": "cw-rno2a",
            "workers": args.workers,
        },
    )
    processes = []
    for shard in range(args.workers):
        with socket.socket() as reservation:
            reservation.bind(("", 0))
            port = reservation.getsockname()[1]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard), "VLLM_PORT": str(port)}
        command = [
            "uv",
            "run",
            "--no-project",
            "--no-sync",
            sys.executable,
            "conditioning_worker.py",
            "--plan",
            str(args.plan),
            "--model",
            model,
            "--out",
            args.out,
            "--shard",
            f"{shard}/{args.workers}",
        ]
        if args.stem:
            if args.workers != 1:
                raise ValueError("An operational smoke uses one worker")
            command.extend(["--stem", args.stem])
        processes.append(subprocess.Popen(command, env=env))
    failures = [(i, process.wait()) for i, process in enumerate(processes)]
    failures = [(i, status) for i, status in failures if status]
    if failures:
        raise RuntimeError(f"GPU workers failed: {failures}")


if __name__ == "__main__":
    main()
