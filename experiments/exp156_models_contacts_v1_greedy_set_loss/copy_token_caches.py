"""Resumable exp156 contacts-v1 token-cache mirror from GCS to CoreWeave S3.

Run inside the dedicated CoreWeave copy Pod. The Pod mounts a short-lived GCS
reader credential at ``GOOGLE_APPLICATION_CREDENTIALS`` and inherits the
cluster's existing CoreWeave S3 credentials through ``iris-task-env``. It
copies cache completion markers last, so a marker at the destination means its
entire cache root was copied successfully.
"""

import argparse
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import gcsfs
import s3fs

logger = logging.getLogger(__name__)

CHUNK_BYTES = 8 * 1024 * 1024
EXECUTOR_STATUS_FILENAME = ".executor_status"


@dataclass(frozen=True)
class CopyEntry:
    """One immutable source object and its relative path under a cache root."""

    relative_path: str
    source_size: int


def parse_args() -> argparse.Namespace:
    """Parse cache-root pairs and copy parallelism."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        action="append",
        nargs=2,
        metavar=("SOURCE", "DESTINATION"),
        required=True,
        help="One source/destination cache-root pair; repeat for train and validation.",
    )
    parser.add_argument("--workers", type=int, default=16, help="Concurrent object copies.")
    return parser.parse_args()


def relative_entries(gcs: gcsfs.GCSFileSystem, source_root: str) -> list[CopyEntry]:
    """List source objects except the completion marker, sorted by path."""
    source_prefix = gcs._strip_protocol(source_root).rstrip("/")
    entries: list[CopyEntry] = []
    for source_path, info in gcs.find(source_root, detail=True).items():
        if source_path == source_prefix:
            continue
        if not source_path.startswith(source_prefix + "/"):
            raise RuntimeError(f"unexpected GCS listing entry {source_path!r} under {source_root!r}")
        relative_path = source_path[len(source_prefix) + 1 :]
        if relative_path == EXECUTOR_STATUS_FILENAME:
            continue
        size = info.get("size")
        if not isinstance(size, int):
            raise RuntimeError(f"GCS entry {source_path!r} has no integer size: {info!r}")
        entries.append(CopyEntry(relative_path=relative_path, source_size=size))
    return sorted(entries, key=lambda entry: entry.relative_path)


def destination_has_matching_size(s3: s3fs.S3FileSystem, destination: str, source_size: int) -> bool:
    """Return whether an existing destination object has the expected byte size."""
    if not s3.exists(destination):
        return False
    size = s3.info(destination).get("size")
    return size == source_size


def copy_entry(
    gcs: gcsfs.GCSFileSystem,
    s3: s3fs.S3FileSystem,
    source_root: str,
    destination_root: str,
    entry: CopyEntry,
) -> int:
    """Copy one object atomically, skipping a size-matched prior result."""
    source = f"{source_root.rstrip('/')}/{entry.relative_path}"
    destination = f"{destination_root.rstrip('/')}/{entry.relative_path}"
    if destination_has_matching_size(s3, destination, entry.source_size):
        return 0

    temporary = f"{destination}.tmp.{uuid.uuid4().hex}"
    copied = 0
    try:
        with gcs.open(source, "rb") as source_file, s3.open(temporary, "wb") as destination_file:
            while chunk := source_file.read(CHUNK_BYTES):
                destination_file.write(chunk)
                copied += len(chunk)
        if copied != entry.source_size:
            raise RuntimeError(f"copied {copied} bytes, expected {entry.source_size} for {source!r}")
        s3.mv(temporary, destination)
    except BaseException:
        if s3.exists(temporary):
            s3.rm(temporary)
        raise
    return copied


def copy_completion_marker(
    gcs: gcsfs.GCSFileSystem,
    s3: s3fs.S3FileSystem,
    source_root: str,
    destination_root: str,
) -> None:
    """Copy the source completion marker after every data object has succeeded."""
    source = f"{source_root.rstrip('/')}/{EXECUTOR_STATUS_FILENAME}"
    destination = f"{destination_root.rstrip('/')}/{EXECUTOR_STATUS_FILENAME}"
    source_size = gcs.info(source).get("size")
    if not isinstance(source_size, int):
        raise RuntimeError(f"GCS completion marker {source!r} has no integer size")
    if destination_has_matching_size(s3, destination, source_size):
        return
    temporary = f"{destination}.tmp.{uuid.uuid4().hex}"
    try:
        with gcs.open(source, "rb") as source_file, s3.open(temporary, "wb") as destination_file:
            while chunk := source_file.read(CHUNK_BYTES):
                destination_file.write(chunk)
        s3.mv(temporary, destination)
    except BaseException:
        if s3.exists(temporary):
            s3.rm(temporary)
        raise


def copy_cache(gcs: gcsfs.GCSFileSystem, s3: s3fs.S3FileSystem, source_root: str, destination_root: str, workers: int) -> None:
    """Copy one cache root with bounded parallelism and resumable size checks."""
    entries = relative_entries(gcs, source_root)
    if not entries:
        raise RuntimeError(f"source cache {source_root!r} contains no data objects")
    logger.info("Copying %d objects: %s -> %s", len(entries), source_root, destination_root)
    copied_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(copy_entry, gcs, s3, source_root, destination_root, entry)
            for entry in entries
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            copied_bytes += future.result()
            if completed % 100 == 0 or completed == len(entries):
                logger.info("Completed %d/%d objects (%.2f GiB copied)", completed, len(entries), copied_bytes / 2**30)
    copy_completion_marker(gcs, s3, source_root, destination_root)
    logger.info("Completed cache: %s -> %s", source_root, destination_root)


def main() -> None:
    """Mirror all requested roots using the mounted GCS and inherited S3 credentials."""
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS must point to the mounted GCS reader credential")
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gcs = gcsfs.GCSFileSystem()
    s3 = s3fs.S3FileSystem(anon=False)
    for source_root, destination_root in args.cache:
        copy_cache(gcs, s3, source_root, destination_root, args.workers)


if __name__ == "__main__":
    main()
