"""Stage the co-located exp277 checkpoint into a CoreWeave worker pod."""

import argparse
import os
from pathlib import Path

import fsspec


def main() -> None:
    """Copy one flat HF checkpoint prefix to local pod storage."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    filesystem, root = fsspec.core.url_to_fs(args.source)
    files = [item for item in filesystem.ls(root, detail=True) if item["type"] == "file"]
    if not files:
        raise FileNotFoundError(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    for item in files:
        filesystem.get_file(item["name"], str(args.out / os.path.basename(item["name"])))
    size = sum(int(item["size"]) for item in files)
    print(f"[exp321] staged {len(files)} files ({size / 2**30:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
