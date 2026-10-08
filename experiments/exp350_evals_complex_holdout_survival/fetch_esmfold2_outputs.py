"""Recover the saved ESMFold2 structures and timings from Modal."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import modal


def fetch_file(
    volume: modal.Volume, output: Path, entry: modal.volume.FileEntry
) -> None:
    """Download one file atomically, preserving the source directory layout."""
    destination = output / entry.path
    if destination.exists() and destination.stat().st_size == entry.size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with temporary.open("wb") as handle:
        for chunk in volume.read_file(entry.path):
            handle.write(chunk)
    if temporary.stat().st_size != entry.size:
        raise ValueError(f"Size mismatch: {entry.path}")
    temporary.replace(destination)


def main() -> None:
    """Download files with their original hierarchy, checking byte counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    volume = modal.Volume.from_name("exp350-four-model-structures")
    entries = [
        e
        for e in volume.iterdir("/esmfold2", recursive=True)
        if e.type == modal.volume.FileEntryType.FILE
    ]

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(partial(fetch_file, volume, args.output), entries))
    print(
        "Downloaded",
        len(entries),
        "files;",
        len(list((args.output / "esmfold2").glob("*/provenance.json"))),
        "completed targets",
    )


if __name__ == "__main__":
    main()
