"""Run a broader AFDB audit on a region-pinned Iris CPU worker.

The launch workspace contains only this experiment's frozen source and metadata.
Original mmCIFs are parsed in memory; C-alpha arrays and measured results are
streamed to the matching us-central1 GCS bucket. Completion is published last.
"""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tarfile
from pathlib import Path
from time import perf_counter

from structure_audit import filesystem


def main() -> None:
    """Execute the source audit and preserve its outputs even on an ordinary failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if not args.output.startswith(
        "gs://marin-us-central1/protein-structure/MarinFold/exp292/"
    ):
        raise ValueError(
            "This worker must write to the matching us-central1 experiment prefix"
        )
    root = Path(__file__).resolve().parent
    results, cache = root / "results", root / "structures"
    results.mkdir(exist_ok=True)
    for path in (root / "inputs").iterdir():
        if path.is_file():
            shutil.copy2(path, results / path.name)
    fs = filesystem()
    started = perf_counter()
    returncode = None
    command = [
        sys.executable,
        str(root / "structure_audit.py"),
        "--sample",
        str(root / "inputs/sample.csv"),
        "--cache",
        str(cache),
        "--output",
        str(results),
        "--workers",
        str(args.workers),
        "--skip-raw-cache",
    ]
    print("Starting AFDB source audit", flush=True)
    with (results / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            if process.stdout is None:
                raise RuntimeError("Worker stdout pipe is unavailable")
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for path in sorted(results.iterdir()):
                if path.is_file():
                    fs.put_file(str(path), args.output + "/" + path.name)
            if returncode == 0:
                with (
                    fs.open(args.output + "/ca-structures.tar.gz", "wb") as handle,
                    tarfile.open(fileobj=handle, mode="w|gz") as archive,
                ):
                    for path in sorted(cache.glob("*.npz")):
                        archive.add(path, arcname=path.name)
            status = {
                "returncode": returncode,
                "elapsed_seconds": perf_counter() - started,
                "hostname": socket.gethostname(),
                "command": command,
                "status": "complete" if returncode == 0 else "failed",
            }
            fs.pipe_file(
                args.output + "/execution.json",
                (json.dumps(status, indent=2) + "\n").encode(),
            )
    if returncode:
        raise RuntimeError(f"AFDB audit failed with exit code {returncode}")
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
