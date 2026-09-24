#!/usr/bin/env python
"""Mirror small CoreWeave result parquets once for local analysis."""

import argparse
from pathlib import Path

import fsspec

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = "s3://marin-us-east-02a/MarinFold/exp304/blind-search-v1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=HERE / "_cache" / "raw")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(args.source)
    paths = [path for path in fs.glob(f"{root}/*.parquet") if path.endswith(".parquet")]
    if not paths:
        raise FileNotFoundError(f"no parquet files under {args.source}")
    for path in paths:
        destination = args.out / Path(path).name
        if destination.exists() and destination.stat().st_size == fs.info(path)["size"]:
            continue
        fs.get_file(path, str(destination))
    print(f"mirrored {len(paths)} parquets to {args.out}")


if __name__ == "__main__":
    main()
