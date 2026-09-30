"""Fetch only the manifest-pinned public raw files needed by this analysis."""

import argparse
import csv
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
PREFIX = "https://huggingface.co/buckets/open-athena/MarinFold/resolve/data/exp321/null-sequence-guidance-v1"


def fetch(row: dict[str, str], destination: Path) -> None:
    """Download one public parquet and require its frozen checksum."""
    path = destination / row["relative_path"]
    if path.exists():
        payload = path.read_bytes()
    else:
        with urlopen(f"{PREFIX}/{row['relative_path']}", timeout=120) as response:
            payload = response.read()
    if len(payload) != int(row["bytes"]) or hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise ValueError(f"raw artifact changed: {row['relative_path']}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def main() -> None:
    """Recover the 35.9 MB raw input set without authentication."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, default=HERE / "_cache")
    args = parser.parse_args()
    with (HERE / "data/raw_manifest.csv").open() as source:
        rows = list(csv.DictReader(source))
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda row: fetch(row, args.destination), rows))
    print(f"verified {len(rows)} raw files")


if __name__ == "__main__":
    main()
