"""Bound an EC2 curation process and preserve its outputs before termination."""

import argparse
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3


def main() -> None:
    """Run one frozen curation job with periodic logs and durable final artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = root / "results"
    output.mkdir(exist_ok=True)
    s3 = boto3.client("s3", region_name="us-west-2")
    worker_config = json.loads((root / "worker-config.json").read_text())
    started = time.monotonic()
    returncode = None
    with (output / "worker.log").open("w") as log:
        process = subprocess.Popen(
            [
                "/opt/exp292-bootstrap/bin/uv",
                "run",
                "--frozen",
                "--extra",
                "esm",
                "python",
                "sample_esm.py",
                "--plan",
                "inputs/plan.parquet",
                "--retained",
                "inputs/retained.parquet",
                "--output",
                "results",
                *worker_config["arguments"],
            ],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started > 9000:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise TimeoutError("Curation exceeded its 150 minute bound")
                s3.upload_file(
                    str(output / "worker.log"), args.bucket, args.prefix + "/worker.log"
                )
                time.sleep(30)
            returncode = process.returncode
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            (output / "execution.json").write_text(
                json.dumps(
                    {
                        "returncode": returncode,
                        "elapsed_seconds": time.monotonic() - started,
                        "status": "complete" if returncode == 0 else "failed",
                    },
                    indent=2,
                )
                + "\n"
            )
            paths = [
                p
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name != "execution.json"
            ]

            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = [
                    pool.submit(
                        s3.upload_file,
                        str(path),
                        args.bucket,
                        args.prefix + "/" + path.relative_to(output).as_posix(),
                    )
                    for path in paths
                ]
                for future in futures:
                    future.result()
            # Completion is published only after every result object is durable.
            s3.upload_file(
                str(output / "execution.json"),
                args.bucket,
                args.prefix + "/execution.json",
            )
    if returncode:
        raise RuntimeError(f"Curation worker failed with exit code {returncode}")


if __name__ == "__main__":
    main()
