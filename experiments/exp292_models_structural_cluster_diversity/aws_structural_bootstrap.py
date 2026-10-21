"""Run one ESM structural-ranking shard on EC2."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import boto3


def main() -> None:
    """Download one queue, rank it, and publish completion last."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--metadata-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=18000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    inputs = root / "inputs"
    output = root / "output"
    inputs.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    queue = inputs / "structural_queue.parquet"
    source_key = args.metadata_prefix.rstrip("/") + f"/shard={args.shard}/structural_queue.parquet"
    s3 = boto3.client("s3", region_name="us-west-2")
    s3.download_file(args.bucket, source_key, str(queue))
    log_path = output / "worker.log"
    command = [
        "/opt/exp292-bootstrap/bin/uv",
        "run",
        "--frozen",
        "--extra",
        "esm",
        "python",
        "curate_esm_structures.py",
        "--queue",
        str(queue),
        "--output",
        str(output),
        "--workers",
        "24",
    ]
    started = time.monotonic()
    returncode = None
    error = None
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started > args.timeout_seconds:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise TimeoutError("ESM structural ranking exceeded its execution bound")
                s3.upload_file(
                    str(log_path),
                    args.bucket,
                    args.output_prefix + f"/shard={args.shard}/worker.log",
                )
                time.sleep(30)
            returncode = process.returncode
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            shard_prefix = args.output_prefix + f"/shard={args.shard}"
            status = {
                "status": "complete" if returncode == 0 else "failed",
                "returncode": returncode,
                "error": error,
                "elapsed_seconds": time.monotonic() - started,
                "source": f"s3://{args.bucket}/{source_key}",
                "shard": args.shard,
                "command": command,
            }
            status_path = output / "execution.json"
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            for path in sorted(output.iterdir()):
                if path.is_file() and path.name != "execution.json":
                    s3.upload_file(str(path), args.bucket, shard_prefix + "/" + path.name)
            s3.upload_file(
                str(status_path), args.bucket, shard_prefix + "/execution.json"
            )
    if returncode:
        raise RuntimeError(f"ESM structural shard failed with exit code {returncode}")


if __name__ == "__main__":
    main()
