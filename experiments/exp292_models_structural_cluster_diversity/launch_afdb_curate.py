"""Submit the AFDB production curation job to the marin Iris cluster.

Reads the validated shards, applies the frozen policy in `production_policy.py`,
and writes the selected-addition manifest plus per-shard provenance. Validated
input, the frozen screen tooling and every output are in ``us-central1``, so the
job crosses no region boundary.

Unlike the source-validation job this stage is CPU-bound on pairwise alignment,
so each worker takes several cores and parallelizes clusters across them rather
than leaving them idle.
"""

import argparse
import json
import shlex
import subprocess
from pathlib import Path

import gcsfs

BUNDLE_EXCLUDES = ("^data/", "^gallery/", "^plots/", "^\\.venv/")


def resolve_input(pattern: str) -> dict:
    """Count the validated shards and bytes the job will read."""
    filesystem = gcsfs.GCSFileSystem()
    matches = filesystem.glob(pattern)
    if not matches:
        raise ValueError(f"Validated pattern matched no objects: {pattern}")
    return {
        "input_files": len(matches),
        "input_bytes": sum(filesystem.info(path)["size"] for path in matches),
    }


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
        "curate_afdb_cli.py",
        "--input",
        args.input,
        "--output",
        args.output,
        "--aux-prefix",
        args.aux_prefix,
        "--mmseqs-archive",
        args.mmseqs_archive,
        "--max-workers",
        str(args.max_workers),
        "--screen-threads",
        str(args.screen_threads),
        "--select-workers",
        str(args.select_workers),
        "--worker-cpu",
        str(args.worker_cpu),
        "--worker-memory",
        args.worker_memory,
        "--worker-disk",
        args.worker_disk,
        "--region",
        args.region,
    ]
    for reference in args.reference:
        worker.extend(["--reference", reference])
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
    parser.add_argument("--aux-prefix", required=True)
    parser.add_argument("--mmseqs-archive", required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--cluster", default="marin")
    parser.add_argument("--region", default="us-central1")
    parser.add_argument("--max-workers", type=int, default=256)
    parser.add_argument("--screen-threads", type=int, default=4)
    parser.add_argument("--select-workers", type=int, default=8)
    parser.add_argument("--worker-cpu", type=float, default=8)
    parser.add_argument("--worker-memory", default="16g")
    parser.add_argument("--worker-disk", default="8g")
    parser.add_argument("--priority", default="batch")
    parser.add_argument(
        "--preemptible", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not args.output.endswith(".parquet"):
        raise ValueError("Selection output must name a parquet pattern")
    if "{shard" not in args.output:
        raise ValueError("A full run must write one manifest parquet per input shard")
    if args.select_workers > args.worker_cpu:
        raise ValueError("Selection processes must not exceed the worker's cores")
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
        "aux_prefix": args.aux_prefix,
        "mmseqs_archive": args.mmseqs_archive,
        "reference": args.reference,
        "region": args.region,
        "priority": args.priority,
        "preemptible": args.preemptible,
        "max_workers": args.max_workers,
        "screen_threads": args.screen_threads,
        "select_workers": args.select_workers,
        "worker_cpu": args.worker_cpu,
        "worker_memory": args.worker_memory,
        "worker_disk": args.worker_disk,
        "command": shlex.join(command),
        **resolve_input(args.input),
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
        raise RuntimeError(
            f"Iris submission failed with exit code {completed.returncode}"
        )


if __name__ == "__main__":
    main()
