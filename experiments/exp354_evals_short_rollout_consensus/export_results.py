"""Publish S3 artifacts from a small in-region CPU job using the HF CLI."""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import fsspec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    filesystem, root = fsspec.core.url_to_fs(args.source)
    deadline = time.monotonic() + args.wait_seconds
    previous = -1
    while True:
        filesystem.invalidate_cache()
        files = filesystem.find(root)
        markers = [p for p in files if "/complete/" in p and p.endswith(".json")]
        if len(markers) != previous:
            print(json.dumps({"event": "progress", "complete": len(markers), "expected": args.expected}), flush=True)
            previous = len(markers)
        if len(markers) >= args.expected or time.monotonic() >= deadline:
            break
        time.sleep(20)
    if len(markers) != args.expected:
        raise RuntimeError(f"Expected {args.expected} complete proteins; found {len(markers)}")
    with tempfile.TemporaryDirectory() as directory:
        for remote in files:
            local = Path(directory) / remote.removeprefix(root.rstrip("/") + "/")
            local.parent.mkdir(parents=True, exist_ok=True)
            filesystem.get_file(remote, str(local))
        manifest = [{"path": str(p.relative_to(directory)), "bytes": p.stat().st_size}
                    for p in Path(directory).rglob("*") if p.is_file()]
        (Path(directory) / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
        subprocess.run([sys.executable, "-m", "huggingface_hub.cli.hf", "buckets", "sync",
                        directory, args.destination], check=True)
    print(json.dumps({"event": "published", "destination": args.destination, "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()
