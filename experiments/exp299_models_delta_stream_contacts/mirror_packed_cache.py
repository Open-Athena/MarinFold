"""Mirror or verify the exp299 packed cache between regional S3 buckets.

The copy uses S3's CopyObject / multipart-copy operations through s3fs, so the
control process does not download cache objects to local disk. Run it as an
Iris job in the destination compute region.
"""

import argparse
import shutil
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import fsspec

DEFAULT_SOURCE = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/packed_cache/2026.09.16.1"
)
DEFAULT_DESTINATION = (
    "s3://rhoarnet-us-east-08a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/packed_cache/2026.09.16.1"
)


def _paths(root: str) -> tuple[object, str, list[str]]:
    fs, path = fsspec.core.url_to_fs(root.rstrip("/"))
    files = sorted(fs.find(path))
    if not files:
        raise ValueError(f"no objects under {root}")
    return fs, path, files


def _destination_paths(source_path: str, source_root: str, destination_root: str, files: Iterable[str]) -> list[str]:
    prefix = source_path.rstrip("/") + "/"
    return [f"{destination_root.rstrip('/')}/{path.removeprefix(prefix)}" for path in files]


def verify(source: str, destination: str) -> tuple[int, int]:
    source_fs, source_path, source_files = _paths(source)
    destination_fs, destination_path, destination_files = _paths(destination)
    source_relative = [path.removeprefix(source_path.rstrip("/") + "/") for path in source_files]
    destination_relative = [path.removeprefix(destination_path.rstrip("/") + "/") for path in destination_files]
    if source_relative != destination_relative:
        raise ValueError("source and destination object sets differ")

    total_bytes = 0
    for source_file, destination_file in zip(source_files, destination_files, strict=True):
        source_info = source_fs.info(source_file)
        destination_info = destination_fs.info(destination_file)
        if int(source_info["size"]) != int(destination_info["size"]):
            raise ValueError(f"size mismatch: {source_file}")
        total_bytes += int(source_info["size"])
    return len(source_files), total_bytes


def _stream_copy_one(fs: object, source: str, destination: str) -> None:
    if fs.exists(destination) and int(fs.info(source)["size"]) == int(fs.info(destination)["size"]):
        return
    with fs.open(source, "rb") as reader, fs.open(destination, "wb") as writer:
        shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)


def _stream_copy(fs: object, source_files: list[str], destination_paths: list[str], workers: int) -> None:
    # Regional S3 has no cross-region CopyObject path. Streaming through this
    # Iris worker is the intentional, one-time cross-region transfer.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for completed, _ in enumerate(executor.map(lambda pair: _stream_copy_one(fs, *pair), zip(source_files, destination_paths)), 1):
            if completed % 1000 == 0:
                print(f"streamed {completed}/{len(source_files)} objects", flush=True)


def mirror(source: str, destination: str, batch_size: int) -> None:
    source_fs, source_path, source_files = _paths(source)
    destination_paths = _destination_paths(source_path, source, destination, source_files)
    print(f"copying {len(source_files)} objects from {source} to {destination}", flush=True)
    _stream_copy(source_fs, source_files, destination_paths, workers=min(batch_size, 8))
    count, total_bytes = verify(source, destination)
    print(f"verified {count} objects / {total_bytes} bytes", flush=True)


def delete(root: str, batch_size: int) -> None:
    fs, _, files = _paths(root)
    print(f"deleting {len(files)} objects under {root}", flush=True)
    fs.rm(files, batch_size=batch_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("mirror", "verify", "delete"))
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--destination", default=DEFAULT_DESTINATION)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if args.action == "mirror":
        mirror(args.source, args.destination, args.batch_size)
    elif args.action == "verify":
        count, total_bytes = verify(args.source, args.destination)
        print(f"verified {count} objects / {total_bytes} bytes", flush=True)
    else:
        delete(args.source, args.batch_size)


if __name__ == "__main__":
    main()
