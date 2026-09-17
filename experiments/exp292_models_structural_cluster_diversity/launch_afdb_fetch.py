"""Submit the source-local AFDB validation job to the marin Iris cluster.

The job reads the compact production plan, full-GETs every original AFDB v4
mmCIF in ``us-central1``, and writes validated C-alpha arrays back to the
same region. Source objects, workers and output all stay in ``us-central1``;
nothing crosses a region boundary. The launch record pins the resources, the
input/output URIs and the client version so the run can be reproduced.
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import gcsfs

BUNDLE_EXCLUDES = ("^data/", "^gallery/", "^plots/", "^\\.venv/")


def resolve_plan(input_pattern: str) -> dict:
    """Count the plan files and bytes the job will read."""
    filesystem = gcsfs.GCSFileSystem()
    matches = filesystem.glob(input_pattern)
    if not matches:
        raise ValueError(f"Plan pattern matched no objects: {input_pattern}")
    sizes = [filesystem.info(path)["size"] for path in matches]
    return {"plan_files": len(matches), "plan_bytes": sum(sizes)}


def build_command(args: argparse.Namespace) -> list[str]:
    """Render the exact iris submission for this run."""
    submit = [
        "uv",
        "run",
        "iris",
        "--cluster",
        args.cluster,
        "job",
        "run",
        "--job-name",
        args.run_name,
        "--cpu",
        "1",
        "--memory",
        "2GB",
        "--extra",
        "pipeline",
        "--region",
        args.region,
        "--priority",
        args.priority,
        "--no-wait",
    ]
    for pattern in BUNDLE_EXCLUDES:
        submit.extend(["--exclude", pattern])
    worker = [
        "python",
        "afdb_fetch_cli.py",
        "--input",
        args.input,
        "--output",
        args.output,
        "--max-workers",
        str(args.max_workers),
        "--fetch-concurrency",
        str(args.fetch_concurrency),
        "--worker-cpu",
        str(args.worker_cpu),
        "--worker-memory",
        args.worker_memory,
        "--worker-disk",
        args.worker_disk,
        "--region",
        args.region,
    ]
    if args.num_docs is not None:
        worker.extend(["--num-docs", str(args.num_docs)])
    if not args.preemptible:
        worker.append("--no-preemptible")
    return submit + ["--"] + worker


def main() -> None:
    """Record the planned submission and optionally send it to the cluster."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--record-dir", type=Path, required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cluster", default="marin")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--max-workers", type=int, default=256)
    parser.add_argument("--fetch-concurrency", type=int, default=16)
    parser.add_argument("--worker-cpu", type=float, default=1)
    parser.add_argument("--worker-memory", default="8g")
    parser.add_argument("--worker-disk", default="8g")
    parser.add_argument("--num-docs", type=int)
    parser.add_argument("--priority", default="batch")
    parser.add_argument(
        "--preemptible", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not args.output.endswith(".parquet"):
        raise ValueError("Validated output must name a parquet pattern")
    if args.num_docs is None and "{shard" not in args.output:
        raise ValueError("A full run must write one output parquet per input shard")
    args.record_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(args)
    record = {
        "status": "prepared",
        "run_name": args.run_name,
        "cluster": args.cluster,
        "iris_job": f"/bizon/{args.run_name}",
        "dashboard": f"https://iris.oa.dev/#/job/%2Fbizon%2F{args.run_name}",
        "input": args.input,
        "output": args.output,
        "region": args.region,
        "priority": args.priority,
        "preemptible": args.preemptible,
        "max_workers": args.max_workers,
        "fetch_concurrency": args.fetch_concurrency,
        "worker_cpu": args.worker_cpu,
        "worker_memory": args.worker_memory,
        "worker_disk": args.worker_disk,
        "num_docs": args.num_docs,
        "command": shlex.join(command),
        **resolve_plan(args.input),
    }
    record_path = args.record_dir / "launch.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if not args.launch:
        return
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=False,
    )
    record["status"] = "launched" if completed.returncode == 0 else "submit_failed"
    record["returncode"] = completed.returncode
    record["stdout"] = completed.stdout[-4000:]
    record["stderr"] = completed.stderr[-4000:]
    record_path.write_text(json.dumps(record, indent=2) + "\n")
    print(completed.stdout, flush=True)
    print(completed.stderr, flush=True)
    if completed.returncode:
        raise RuntimeError(f"Iris submission failed with exit code {completed.returncode}")


if __name__ == "__main__":
    main()
