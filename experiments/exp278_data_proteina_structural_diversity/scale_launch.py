"""Submit independently preemptible scale workers; Iris owns admission/fairness."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fray.iris_backend import FrayIrisClient
from iris.cli.connect import open_iris_client

from launch import (
    IMAGE,
    bundle,
    create_worker_request,
    stage_bundle,
    storage_filesystem,
)

HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--cluster", choices=["cw-us-east-02a", "cw-rno2a"], default="cw-us-east-02a"
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--spacing-seconds", type=float, default=3)
    parser.add_argument("--no-wait", action="store_true", required=True)
    args = parser.parse_args()
    plan = json.loads(args.manifest.read_text())
    fs = storage_filesystem(args.cluster)
    root = plan["output"].removeprefix("s3://")
    payload = args.manifest.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    manifest_uri = f"s3://{root}/manifests/{digest}.json"
    fs.pipe_file(manifest_uri.removeprefix("s3://"), payload)
    if not fs.exists(root + "/control.json"):
        fs.pipe_file(
            root + "/control.json",
            json.dumps(
                {
                    "pause": False,
                    "pause_at_utc": None,
                    "review_action": "continue while reviewing",
                    "retain_all_backbones_and_sequences": True,
                }
            ).encode(),
        )
    content = bundle()
    bundle_uri = stage_bundle(content, args.cluster)
    report = HERE / "data" / plan["run_id"]
    report.mkdir(parents=True, exist_ok=True)
    if not 0 <= args.start < args.end <= plan["workers"]:
        raise ValueError("Worker range is outside the frozen manifest")
    with open_iris_client(cluster_name=args.cluster, workspace=None) as client:
        fray = FrayIrisClient.from_iris_client(client)
        for worker in range(args.start, args.end):
            record_path = report / f"worker-{worker:04d}-submission.json"
            if record_path.exists():
                previous = json.loads(record_path.read_text())
                if (
                    previous["manifest_sha256"] != digest
                    or previous["cluster"] != args.cluster
                ):
                    raise ValueError(
                        "Worker already has a different manifest or cluster; reconcile its old job first"
                    )
                print(f"Already submitted: {previous['job_id']}", flush=True)
                continue
            name = f"{args.name_prefix}-w{worker:04d}"
            worker_args = [
                "scale_worker.py",
                "--manifest",
                manifest_uri,
                "--worker",
                str(worker),
            ]
            request = create_worker_request(
                name,
                worker_args,
                bundle_uri,
                hashlib.sha256(content).hexdigest(),
                "short,long",
                4 * 86400,
                preemption_retries=100,
                failure_retries=2,
            )
            job = fray.submit(request)
            record = {
                "worker": worker,
                "job_id": job.job_id,
                "cluster": args.cluster,
                "submitted_utc": datetime.now(timezone.utc).isoformat(),
                "priority_band": "batch",
                "priority": 3,
                "gpu_count": 1,
                "replicas": 1,
                "preemption_retries": 100,
                "manifest_uri": manifest_uri,
                "manifest_sha256": digest,
                "bundle_uri": bundle_uri,
                "image": IMAGE,
                "worker_args": worker_args,
            }
            record_path.write_text(json.dumps(record, indent=2) + "\n")
            fs.pipe_file(
                f"{root}/submissions/worker-{worker:04d}.json",
                json.dumps(record).encode(),
            )
            print(json.dumps(record), flush=True)
            time.sleep(args.spacing_seconds)


if __name__ == "__main__":
    main()
