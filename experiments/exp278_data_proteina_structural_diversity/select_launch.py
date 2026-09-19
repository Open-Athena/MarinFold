"""Submit the single-node whole-corpus selection job.

Selection runs on one CPU node rather than a fan-out because Foldseek and
MMseqs2 want every structure on local disk; the stage markers in the object
store, not this launcher, are what make a preempted node resumable.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fray.iris_backend import FrayIrisClient
from iris.cli.connect import open_iris_client

from launch import IMAGE, bundle, create_cpu_request, stage_bundle, storage_filesystem

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--name", required=True)
    parser.add_argument("--only", action="append")
    parser.add_argument("--cpu", type=int, default=160)
    parser.add_argument("--ram", default="1000g")
    parser.add_argument("--disk", default="4000g")
    parser.add_argument("--threads", type=int, default=150)
    parser.add_argument("--timeout", type=int, default=4 * 86400)
    args = parser.parse_args()

    plan = json.loads(args.manifest.read_text())
    fs = storage_filesystem(args.cluster)
    root = plan["output"].removeprefix("s3://")
    payload = args.manifest.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest_uri = f"s3://{root}/manifests/{digest}.json"
    fs.pipe_file(manifest_uri.removeprefix("s3://"), payload)
    content = bundle()
    bundle_uri = stage_bundle(content, args.cluster)

    worker_args = [
        "select_corpus.py",
        "--manifest",
        manifest_uri,
        "--threads",
        str(args.threads),
    ]
    for stage in args.only or []:
        worker_args += ["--only", stage]
    request = create_cpu_request(
        args.name,
        worker_args,
        bundle_uri,
        hashlib.sha256(content).hexdigest(),
        args.timeout,
        cpu=args.cpu,
        ram=args.ram,
        disk=args.disk,
    )
    with open_iris_client(cluster_name=args.cluster, workspace=None) as client:
        job = FrayIrisClient.from_iris_client(client).submit(request)
    report = HERE / "data" / plan["run_id"] / "selection"
    report.mkdir(parents=True, exist_ok=True)
    record = {
        "job_id": job.job_id,
        "cluster": args.cluster,
        "name": args.name,
        "stages": args.only or "all",
        "cpu": args.cpu,
        "ram": args.ram,
        "disk": args.disk,
        "threads": args.threads,
        "submitted_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": digest,
        "bundle_uri": bundle_uri,
        "image": IMAGE,
        "worker_args": worker_args,
    }
    (report / f"{args.name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
