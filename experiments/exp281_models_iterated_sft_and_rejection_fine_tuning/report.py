"""Attach bounded result files to the existing W&B run for workstation review.

Large candidate parquets and checkpoints stay on co-located storage. Only
explicitly requested small manifests, metrics, and timing files are collected.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import fsspec
import wandb

from common import files, identity, write_json
from train import record_history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patterns", nargs="+", required=True)
    parser.add_argument("--max-bytes", type=int, default=10_000_000)
    args = parser.parse_args()
    paths = sorted({p for pattern in args.patterns for p in files(pattern)})
    total = 0
    for path in paths:
        fs, source = fsspec.core.url_to_fs(path)
        total += fs.size(source)
    if total > args.max_bytes:
        raise ValueError(f"report is {total} bytes, exceeding explicit small-artifact budget")
    run = wandb.init(entity="open-athena", project="MarinFold", id=args.run_name,
                     name=args.run_name, resume="must", job_type="report")
    record_history(run, args.output)
    with tempfile.TemporaryDirectory(prefix="exp281-report-") as directory:
        artifact = wandb.Artifact(f"{args.run_name}-report", type="experiment-report")
        index = {}
        for uri in paths:
            name = identity(uri)[:12] + "-" + uri.rsplit("/", 1)[-1]
            path = Path(directory) / name
            with fsspec.open(uri, "rb") as src, path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            artifact.add_file(str(path))
            index[name] = uri
        path = Path(directory) / "index.json"
        write_json(str(path), index)
        artifact.add_file(str(path))
        run.log_artifact(artifact).wait()
    run.finish()


if __name__ == "__main__":
    main()
