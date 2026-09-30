# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fsspec",
#   "gcsfs",
#   "huggingface-hub>=1.6,<2",
#   "pyarrow",
# ]
# ///

"""Stage premade-contact shards to an in-region GCS prefix."""

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import PurePosixPath
from typing import Any

import fsspec
from huggingface_hub import HfFileSystem

HF_ROOT = "buckets/open-athena/MarinFold"
CONTACTS_PREFIX = f"{HF_ROOT}/data/contacts/esm_atlas_esmfold2_distill"
DEFAULT_OUTPUT = (
    "gs://marin-us-east5/protein-structure/MarinFold/"
    "exp147_on_the_fly_contacts_v1_pilot/pilot_data"
)
MAX_TRANSFER_BYTES = 10_000_000_000


def _shard_name(index: int, total: int = 3338) -> str:
    return f"shard-{index:05d}-of-{total:05d}.parquet"


def _copy_file(source: Any, destination: str) -> None:
    with fsspec.open(destination, "wb") as target:
        shutil.copyfileobj(source, target, length=32 << 20)


def _destination_path(output_prefix: str, shard_name: str) -> str:
    return f"{output_prefix.rstrip('/')}/contacts/{shard_name}"


def _stage_one(
    shard_name: str,
    *,
    output_prefix: str,
    expected_size: int,
) -> tuple[str, int]:
    hf_fs = HfFileSystem(token=False)
    contacts_source = str(PurePosixPath(CONTACTS_PREFIX) / shard_name)
    contacts_destination = _destination_path(output_prefix, shard_name)

    destination_fs, destination_path = fsspec.core.url_to_fs(contacts_destination)
    if destination_fs.exists(destination_path):
        actual_size = int(destination_fs.info(destination_path)["size"])
        if actual_size != expected_size:
            raise ValueError(
                f"Existing destination {contacts_destination} has size "
                f"{actual_size}, expected {expected_size}"
            )
        return shard_name, 0

    with hf_fs.open(contacts_source, "rb") as source:
        _copy_file(source, contacts_destination)

    actual_size = int(destination_fs.info(destination_path)["size"])
    if actual_size != expected_size:
        raise ValueError(
            f"Copied destination {contacts_destination} has size {actual_size}, "
            f"expected {expected_size}"
        )
    return shard_name, actual_size


def _source_sizes(hf_fs: HfFileSystem) -> dict[str, int]:
    entries = hf_fs.glob(
        str(PurePosixPath(CONTACTS_PREFIX) / "*.parquet"),
        detail=True,
    )
    if not isinstance(entries, dict):
        raise TypeError("Detailed Hugging Face glob did not return file metadata")
    return {
        PurePosixPath(path).name: int(info["size"])
        for path, info in entries.items()
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--num-shards", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Perform the transfer. Without this flag, only print the plan.",
    )
    parser.add_argument(
        "--allow-large-transfer",
        action="store_true",
        help="Acknowledge an approved transfer above 10 GB.",
    )
    args = parser.parse_args(argv)
    if args.num_shards <= 0 or args.num_shards > 3338:
        parser.error("--num-shards must be in [1, 3338]")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    hf_fs = HfFileSystem(token=False)
    shard_names = [_shard_name(index) for index in range(args.num_shards)]
    source_sizes = _source_sizes(hf_fs)
    missing_sources = sorted(set(shard_names) - source_sizes.keys())
    if missing_sources:
        raise FileNotFoundError(
            f"{len(missing_sources)} expected source shards are absent; "
            f"first missing shard: {missing_sources[0]}"
        )

    contacts_destination = _destination_path(args.output, "")
    destination_fs, destination_prefix = fsspec.core.url_to_fs(
        contacts_destination
    )
    existing_entries = destination_fs.glob(
        f"{destination_prefix.rstrip('/')}/*.parquet",
        detail=True,
    )
    if not isinstance(existing_entries, dict):
        raise TypeError("Detailed destination glob did not return file metadata")
    existing_sizes = {
        PurePosixPath(path).name: int(info["size"])
        for path, info in existing_entries.items()
    }
    for shard_name in sorted(set(shard_names) & existing_sizes.keys()):
        if existing_sizes[shard_name] != source_sizes[shard_name]:
            raise ValueError(
                f"Existing destination {_destination_path(args.output, shard_name)} "
                f"has size {existing_sizes[shard_name]}, expected "
                f"{source_sizes[shard_name]}"
            )

    missing_shards = [
        shard_name for shard_name in shard_names if shard_name not in existing_sizes
    ]
    transfer_bytes = sum(source_sizes[name] for name in missing_shards)
    if (
        args.execute
        and transfer_bytes > MAX_TRANSFER_BYTES
        and not args.allow_large_transfer
    ):
        raise ValueError(
            f"Transfer would copy {transfer_bytes / 1e9:.2f} GB, above the "
            f"{MAX_TRANSFER_BYTES / 1e9:.0f} GB approval threshold; an explicitly "
            f"approved run must pass --allow-large-transfer"
        )

    print(
        f"Plan: retain {len(shard_names) - len(missing_shards)} existing shards "
        f"and copy {len(missing_shards)} missing shards "
        f"({transfer_bytes / 1e9:.2f} GB) to {args.output}",
        file=sys.stderr,
    )
    if not args.execute:
        print(
            "Preview only; source and destination metadata were read, but no "
            "data was transferred. Re-run with --execute to apply this plan.",
            file=sys.stderr,
        )
        return 0

    if not missing_shards:
        print("All requested shards are already staged.", file=sys.stderr)
        return 0

    completed_shards = 0
    completed_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _stage_one,
                shard_name,
                output_prefix=args.output,
                expected_size=source_sizes[shard_name],
            ): shard_name
            for shard_name in missing_shards
        }
        for future in as_completed(futures):
            _, copied_bytes = future.result()
            completed_shards += 1
            completed_bytes += copied_bytes
            if completed_shards % 50 == 0 or completed_shards == len(missing_shards):
                print(
                    f"Progress: {completed_shards}/{len(missing_shards)} shards, "
                    f"{completed_bytes / 1e9:.2f}/{transfer_bytes / 1e9:.2f} GB",
                    file=sys.stderr,
                    flush=True,
                )
    print(
        f"Staged all {len(shard_names)} contact shards under {args.output}/contacts",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
