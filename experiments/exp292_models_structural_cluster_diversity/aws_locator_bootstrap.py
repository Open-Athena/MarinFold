"""Run the ESM Atlas row locator on EC2 and publish completion last."""

import argparse
import json
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3


def download_prefix(s3, bucket: str, prefix: str, destination: Path) -> int:
    """Download every parquet object below a source-local S3 prefix."""
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objects.extend(
            item for item in page.get("Contents", []) if item["Key"].endswith(".parquet")
        )
    if not objects:
        raise ValueError(f"No parquet plan objects below s3://{bucket}/{prefix}")

    def download(item: dict) -> None:
        relative = item["Key"][len(prefix) :].lstrip("/")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, item["Key"], str(target))

    with ThreadPoolExecutor(max_workers=32) as pool:
        for future in [pool.submit(download, item) for item in objects]:
            future.result()
    return len(objects)


def upload_tree(s3, root: Path, bucket: str, prefix: str) -> None:
    """Upload result files using paths relative to the result root."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            s3.upload_file(
                str(path),
                bucket,
                prefix + "/" + path.relative_to(root).as_posix(),
            )


def main() -> None:
    """Download the compact plan, scan Atlas, and preserve durable outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--plan-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=21600)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = root / "inputs" / "plan"
    work = root / "work"
    result_root = root / "output"
    located = result_root / "located"
    result_root.mkdir(exist_ok=True)
    log_path = result_root / "worker.log"
    s3 = boto3.client("s3", region_name="us-west-2")
    plan_files = download_prefix(s3, args.bucket, args.plan_prefix, plan)
    command = [
        "/opt/exp292-bootstrap/bin/uv",
        "run",
        "--frozen",
        "--extra",
        "esm",
        "python",
        "locate_esm_rows.py",
        "--plan-glob",
        str(plan / "*" / "*.parquet"),
        "--work",
        str(work),
        "--output",
        str(located),
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
                    raise TimeoutError("ESM Atlas locator exceeded its execution bound")
                s3.upload_file(
                    str(log_path), args.bucket, args.output_prefix + "/worker.log"
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
            status = {
                "status": "complete" if returncode == 0 else "failed",
                "returncode": returncode,
                "error": error,
                "elapsed_seconds": time.monotonic() - started,
                "plan_files": plan_files,
                "command": command,
            }
            status_path = result_root / "execution.json"
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            s3.upload_file(
                str(log_path), args.bucket, args.output_prefix + "/worker.log"
            )
            if returncode == 0:
                upload_tree(s3, located, args.bucket, args.output_prefix + "/located")
                s3.upload_file(
                    str(result_root / "locator.json"),
                    args.bucket,
                    args.output_prefix + "/locator.json",
                )
            s3.upload_file(
                str(status_path),
                args.bucket,
                args.output_prefix + "/execution.json",
            )
    if returncode:
        raise RuntimeError(f"ESM locator failed with exit code {returncode}")


if __name__ == "__main__":
    main()
