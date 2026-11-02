"""Collect durable raw-rollout timing sidecars into the experiment CSV."""

import argparse
import csv
from pathlib import Path

import fsspec


def collect(pattern: str, output: Path) -> int:
    """Combine measured timing sidecars, rejecting missing or duplicate inputs.

    Args:
        pattern: Local or remote glob matching the worker's timing CSV files.
        output: Destination for the committed per-input timing table.

    Returns:
        Number of input measurements collected.
    """
    fs, path = fsspec.core.url_to_fs(pattern)
    sources = sorted(fs.glob(path))
    if not sources:
        raise FileNotFoundError(f"No timing sidecars matched {pattern}")
    rows = []
    fields = None
    seen = set()
    for source in sources:
        with fs.open(source, "rt") as handle:
            reader = csv.DictReader(handle)
            if fields is None:
                fields = reader.fieldnames
            if fields != reader.fieldnames:
                raise ValueError(f"Timing schemas differ: {source}")
            records = list(reader)
        if len(records) != 1:
            raise ValueError(f"Expected one per-input timing record: {source}")
        row = records[0]
        identity = (row["model_source"], row["stem"], row["mode"])
        if identity in seen:
            raise ValueError(f"Duplicate timing measurement: {identity}")
        seen.add(identity)
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Glob of *.timings.csv sidecars")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data/timings.csv")
    args = parser.parse_args()
    print(f"Collected {collect(args.input, args.output)} measurements into {args.output}")


if __name__ == "__main__":
    main()
