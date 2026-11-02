"""Copy one in-cluster S3 result prefix to the public HF bucket."""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fsspec


def main() -> None:
    """Download S3 objects locally in-cluster, then upload them with the HF CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN is required")
    filesystem, root = fsspec.core.url_to_fs(
        args.source,
        config_kwargs={"s3": {"addressing_style": "virtual"}},
    )
    files = sorted(filesystem.find(root))
    if not files:
        raise FileNotFoundError(args.source)
    with tempfile.TemporaryDirectory() as scratch:
        local_root = Path(scratch)
        for remote in files:
            relative = remote[len(root.rstrip("/")) + 1 :]
            local = local_root / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            filesystem.get_file(remote, str(local))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "huggingface_hub.cli.hf",
                "buckets",
                "sync",
                str(local_root),
                args.destination,
            ],
            check=True,
        )
    print(f"[exp333] exported {len(files)} files to {args.destination}", flush=True)


if __name__ == "__main__":
    main()
