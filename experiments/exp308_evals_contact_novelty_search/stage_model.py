#!/usr/bin/env python
"""Stage the co-located exp277 checkpoint into a CoreWeave worker pod."""

import argparse
import os
from pathlib import Path

import fsspec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fs, root = fsspec.core.url_to_fs(args.source)
    files = [item for item in fs.ls(root, detail=True) if item["type"] == "file"]
    if not files:
        raise FileNotFoundError(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    for item in files:
        fs.get_file(item["name"], str(args.out / os.path.basename(item["name"])))
    print(f"[exp306] staged {len(files)} checkpoint files", flush=True)


if __name__ == "__main__":
    main()
