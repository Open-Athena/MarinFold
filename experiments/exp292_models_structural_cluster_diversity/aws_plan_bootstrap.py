"""Run the ESM planning pass on EC2 and publish completion last."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import boto3


def upload_tree(s3, root: Path, bucket: str, prefix: str) -> None:
    """Upload a directory with paths relative to its parent."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            key = prefix + "/" + path.relative_to(root.parent).as_posix()
            s3.upload_file(str(path), bucket, key)


def main() -> None:
    """Execute the bounded planner and preserve logs and results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    work = root / "work"
    output = root / "output" / "plan"
    output.mkdir(parents=True, exist_ok=True)
    log_path = root / "output" / "worker.log"
    s3 = boto3.client("s3", region_name="us-west-2")
    command = [
        "/opt/exp292-bootstrap/bin/uv",
        "run",
        "--frozen",
        "--extra",
        "esm",
        "python",
        "build_esm_plan.py",
        "--droplist",
        "inputs/droplist_final.parquet",
        "--work",
        str(work),
        "--output",
        str(output),
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
                    raise TimeoutError("ESM plan build exceeded its execution bound")
                s3.upload_file(str(log_path), args.bucket, args.prefix + "/worker.log")
                time.sleep(30)
            returncode = process.returncode
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            status = {
                "status": "complete" if returncode == 0 else "failed",
                "returncode": returncode,
                "error": error,
                "elapsed_seconds": time.monotonic() - started,
                "command": command,
            }
            status_path = root / "output" / "execution.json"
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            s3.upload_file(str(log_path), args.bucket, args.prefix + "/worker.log")
            if output.exists():
                upload_tree(s3, output, args.bucket, args.prefix)
            if returncode == 0:
                plan_record = output.parent / "plan.json"
                s3.upload_file(
                    str(plan_record), args.bucket, args.prefix + "/plan.json"
                )
            for diagnostic in output.parent.glob("*.parquet"):
                s3.upload_file(
                    str(diagnostic),
                    args.bucket,
                    args.prefix + "/diagnostics/" + diagnostic.name,
                )
            s3.upload_file(
                str(status_path), args.bucket, args.prefix + "/execution.json"
            )
    if returncode:
        raise RuntimeError(f"ESM plan builder failed with exit code {returncode}")


if __name__ == "__main__":
    main()
