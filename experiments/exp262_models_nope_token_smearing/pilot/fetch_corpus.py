# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Fetch a slice of the decontaminated contacts-v1 corpus for the local pilot.

The pilot (see ``pilot/README.md``) trains ~30M-parameter models on a local
GPU, so it needs a few hundred million tokens rather than the 4.4B of the full
AFDB corpus. Shards are pulled from the public ``open-athena/MarinFold`` bucket,
which is readable with no authentication.

Shards are taken from the two ends of the shard ordering: the low-numbered ones
train and the high-numbered ones validate, so a document never appears in both.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from marinfold.registry import _download_bucket_file, _list_bucket_files

REPO = "open-athena/MarinFold"
PREFIX = "data/document_structures/contacts_v1_decontam/train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-shards", type=int, default=160)
    parser.add_argument("--val-shards", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, default=Path("/data/tim/exp262_pilot/corpus"))
    arguments = parser.parse_args()

    entries = sorted(
        (entry for entry in _list_bucket_files(REPO, PREFIX) if entry.path.endswith(".parquet")),
        key=lambda entry: entry.path,
    )
    print(f"[fetch] bucket lists {len(entries)} parquet shards")
    if arguments.train_shards + arguments.val_shards > len(entries):
        raise SystemExit("asked for more shards than the bucket lists")

    wanted = [
        (entry, "train") for entry in entries[: arguments.train_shards]
    ] + [
        (entry, "val") for entry in entries[-arguments.val_shards :]
    ]

    def fetch(item) -> int:
        entry, split = item
        destination = arguments.out / split / Path(entry.path).name
        if destination.is_file() and destination.stat().st_size == entry.size:
            return 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        _download_bucket_file(REPO, entry, destination)
        return entry.size

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        sizes = list(pool.map(fetch, wanted))
    print(f"[fetch] downloaded {sum(sizes) / 1e9:.2f} GB into {arguments.out}")


if __name__ == "__main__":
    main()
