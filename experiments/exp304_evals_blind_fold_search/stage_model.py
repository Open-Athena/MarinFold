#!/usr/bin/env python
"""Stage the public HF model from an isolated hub>=1.5 environment."""

import argparse
from pathlib import Path

import fsspec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache-out", help="optional co-located S3 model cache to populate")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(args.model)
    files = [item for item in fs.ls(root, detail=True) if item["type"] == "file"]
    if not files:
        raise FileNotFoundError(f"no model files under {args.model}")
    for item in files:
        destination = args.out / Path(item["name"]).name
        if destination.exists() and destination.stat().st_size == item["size"]:
            continue
        fs.get_file(item["name"], str(destination))
    print(f"staged {len(files)} files from {args.model} to {args.out}", flush=True)
    if args.cache_out:
        cache_fs, cache_root = fsspec.core.url_to_fs(args.cache_out)
        for item in files:
            local = args.out / Path(item["name"]).name
            remote = f"{cache_root}/{local.name}"
            if cache_fs.exists(remote) and cache_fs.info(remote)["size"] == local.stat().st_size:
                continue
            cache_fs.put_file(str(local), remote)
        print(f"cached {len(files)} model files under {args.cache_out}", flush=True)


if __name__ == "__main__":
    main()
