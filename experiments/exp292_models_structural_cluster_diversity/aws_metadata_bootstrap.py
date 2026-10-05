"""Run one ESM metadata-curation shard on EC2."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

MMSEQS_KEY = "exp292/production-v1/tools/mmseqs-linux-avx2.tar.gz"
REFERENCE_KEYS = (
    "exp292/production-v1/reference/eval_queries.fasta",
    "exp292/production-v1/reference/foldbench_all_queries.fasta",
)
MMSEQS_SHA256 = "1fb6d8dfe3c83379d2d59ccda19ffc151b69b80d0a62be4691abcd8b4c19e4f2"


def download_prefix(s3, bucket: str, prefix: str, destination: Path) -> int:
    """Download all parquet parts belonging to one located shard."""
    objects = []
    for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        objects.extend(
            item for item in page.get("Contents", []) if item["Key"].endswith(".parquet")
        )
    if not objects:
        raise ValueError(f"No located parquet files below s3://{bucket}/{prefix}")

    def download(item: dict) -> None:
        target = destination / Path(item["Key"]).name
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, item["Key"], str(target))

    with ThreadPoolExecutor(max_workers=16) as pool:
        for future in [pool.submit(download, item) for item in objects]:
            future.result()
    return len(objects)


def upload_tree(s3, root: Path, bucket: str, prefix: str) -> None:
    """Upload a completed shard result tree."""
    for path in sorted(root.rglob("*")):
        if path.is_file():
            s3.upload_file(
                str(path),
                bucket,
                prefix + "/" + path.relative_to(root).as_posix(),
            )


def main() -> None:
    """Fetch shard inputs, run curation, and publish completion last."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--located-prefix", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--max-clusters", type=int)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    input_dir = root / "inputs"
    located_dir = input_dir / "located"
    tools_dir = root / "tools"
    output = root / "output"
    output.mkdir(exist_ok=True)
    log_path = output / "worker.log"
    s3 = boto3.client("s3", region_name="us-west-2")
    plan_files = download_prefix(
        s3,
        args.bucket,
        args.located_prefix.rstrip("/") + f"/shard={args.shard}/",
        located_dir,
    )
    archive = tools_dir / "mmseqs-linux-avx2.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(args.bucket, MMSEQS_KEY, str(archive))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != MMSEQS_SHA256:
        raise ValueError("Frozen MMseqs2 archive hash mismatch")
    with tarfile.open(archive) as bundle:
        bundle.extractall(tools_dir, filter="data")
    references = []
    for key in REFERENCE_KEYS:
        target = input_dir / Path(key).name
        target.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(args.bucket, key, str(target))
        references.append(target)
    command = [
        "/opt/exp292-bootstrap/bin/uv",
        "run",
        "--frozen",
        "--extra",
        "esm",
        "python",
        "curate_esm_metadata.py",
        "--located",
        str(located_dir),
        "--reference",
        str(references[0]),
        "--reference",
        str(references[1]),
        "--mmseqs",
        str(tools_dir / "mmseqs" / "bin" / "mmseqs"),
        "--mmseqs-archive",
        str(archive),
        "--work",
        str(root / "work"),
        "--output",
        str(output),
    ]
    if args.max_clusters is not None:
        command.extend(["--max-clusters", str(args.max_clusters)])
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
                    raise TimeoutError("ESM metadata curation exceeded its execution bound")
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
                "located_plan_files": plan_files,
                "shard": args.shard,
                "command": command,
            }
            status_path = output / "execution.json"
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            if returncode == 0:
                upload_tree(s3, output, args.bucket, shard_prefix)
            else:
                s3.upload_file(str(log_path), args.bucket, shard_prefix + "/worker.log")
            s3.upload_file(
                str(status_path), args.bucket, shard_prefix + "/execution.json"
            )
    if returncode:
        raise RuntimeError(f"ESM metadata shard failed with exit code {returncode}")


if __name__ == "__main__":
    main()
