"""Submit CPU workers that repack the run's artifacts and upload them to the Hub.

Each worker owns `index % workers` of the shard plan, so the set is covered
without coordination and a replacement resumes by asking the Hub what already
exists. The Hub token is passed through the task environment only; it is never
written to the submission records this script persists.
"""

import argparse
import configparser
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from fray.iris_backend import FrayIrisClient
from iris.cli.connect import open_iris_client

from launch import IMAGE, bundle, create_cpu_request, stage_bundle, storage_filesystem

HERE = Path(__file__).resolve().parent


def hub_token(name: str) -> str:
    """Read a named Hub token from the workstation's stored tokens."""
    config = configparser.ConfigParser()
    config.read(Path.home() / ".cache/huggingface/stored_tokens")
    return config[name]["hf_token"].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cluster", default="cw-us-east-02a")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--kind", action="append")
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--token-name", default="write2")
    parser.add_argument("--cpu", type=int, default=24)
    parser.add_argument("--ram", default="180g")
    parser.add_argument("--disk", default="512g")
    parser.add_argument("--timeout", type=int, default=3 * 86400)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--failure-retries", type=int, default=10)
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
    report = HERE / "data" / plan["run_id"] / "uploads"
    report.mkdir(parents=True, exist_ok=True)
    token = hub_token(args.token_name)
    end = args.workers if args.end is None else args.end

    with open_iris_client(cluster_name=args.cluster, workspace=None) as client:
        fray = FrayIrisClient.from_iris_client(client)
        for worker in range(args.start, end):
            name = f"{args.name_prefix}-u{worker:03d}"
            worker_args = [
                "hf_upload.py",
                "--manifest",
                manifest_uri,
                "--repo",
                args.repo,
                "--worker",
                str(worker),
                "--workers",
                str(args.workers),
                "--batch",
                str(args.batch),
            ]
            for kind in args.kind or []:
                worker_args += ["--kind", kind]
            request = create_cpu_request(
                name,
                worker_args,
                bundle_uri,
                hashlib.sha256(content).hexdigest(),
                args.timeout,
                cpu=args.cpu,
                ram=args.ram,
                disk=args.disk,
                env_vars={"HF_TOKEN": token, "HF_HUB_ENABLE_HF_TRANSFER": "0"},
                failure_retries=args.failure_retries,
            )
            job = fray.submit(request)
            record = {
                "worker": worker,
                "workers": args.workers,
                "job_id": job.job_id,
                "cluster": args.cluster,
                "repo": args.repo,
                "kinds": args.kind or "all",
                "submitted_utc": datetime.now(timezone.utc).isoformat(),
                "manifest_sha256": digest,
                "bundle_uri": bundle_uri,
                "image": IMAGE,
                "worker_args": worker_args,
                "commit_batch": args.batch,
                "failure_retries": args.failure_retries,
                "token_name": args.token_name,
            }
            (report / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
            print(json.dumps(record), flush=True)
            time.sleep(2)


if __name__ == "__main__":
    main()
