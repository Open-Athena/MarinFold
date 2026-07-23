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

"""Stage a small premade-contacts pilot to GCS."""

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
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


def _stage_one(
    shard_name: str,
    *,
    output_prefix: str,
) -> str:
    hf_fs = HfFileSystem(token=False)
    contacts_source = str(PurePosixPath(CONTACTS_PREFIX) / shard_name)
    contacts_destination = str(PurePosixPath(output_prefix) / "contacts" / shard_name)

    destination_fs, destination_path = fsspec.core.url_to_fs(contacts_destination)
    if not destination_fs.exists(destination_path):
        with hf_fs.open(contacts_source, "rb") as source:
            _copy_file(source, contacts_destination)

    return shard_name


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
    args = parser.parse_args(argv)
    if args.num_shards <= 0 or args.num_shards > 3338:
        parser.error("--num-shards must be in [1, 3338]")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    hf_fs = HfFileSystem(token=False)
    shard_names = [_shard_name(index) for index in range(args.num_shards)]
    transfer_bytes = sum(
        int(hf_fs.info(str(PurePosixPath(CONTACTS_PREFIX) / shard_name))["size"])
        for shard_name in shard_names
    )
    if transfer_bytes > MAX_TRANSFER_BYTES:
        raise ValueError(
            f"Pilot would transfer {transfer_bytes / 1e9:.2f} GB, above the "
            f"{MAX_TRANSFER_BYTES / 1e9:.0f} GB approval threshold"
        )

    print(
        f"Plan: stage {len(shard_names)} shards "
        f"({transfer_bytes / 1e9:.2f} GB maximum) "
        f"to {args.output}",
        file=sys.stderr,
    )
    if not args.execute:
        print(
            "Preview only; no destination was contacted and no data was "
            "transferred. Re-run with --execute to apply this plan.",
            file=sys.stderr,
        )
        return 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        staged = list(
            pool.map(
                lambda name: _stage_one(name, output_prefix=args.output),
                shard_names,
            )
        )
    print(
        f"Staged {len(staged)} contact shards under {args.output}/contacts",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
