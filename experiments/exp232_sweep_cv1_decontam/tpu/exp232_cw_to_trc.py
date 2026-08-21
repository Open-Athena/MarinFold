# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Ingress exp232 data and one full-state checkpoint from CW into TRC regions.

Every regional copy reads the original CoreWeave S3 objects directly and writes
only to that region's canonical GCS bucket. GCS is never a transfer source.
Objects stream through bounded memory on the region-pinned Iris CPU worker; they
are not staged on the development VM or the worker's disk.
"""

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import gcsfs
import s3fs

SUPPORTED_REGIONS = ("europe-west4", "us-east1", "us-east5", "us-west4")
REGION_BUCKETS = {
    "europe-west4": "marin-eu-west4",
    "us-east1": "marin-us-east1",
    "us-east5": "marin-us-east5",
    "us-west4": "marin-us-west4",
}
REGIONAL_EXPERIMENT_RELATIVE = "protein-structure/MarinFold/exp232_train_trc"
INGRESS_VERSION = "2026.08.21.1"
SOURCE_RUN_ID = (
    "prot-exp232-cw-cv1-decontam-recover-a03-skipstep-m2-p06-srcpeak-augcont"
)
SOURCE_RUN_VERSION = "2026.08.20.1"
SOURCE_CHECKPOINT_STEP = 333_960

COPY_BUFFER_BYTES = 16 * 2**20
GCS_BLOCK_BYTES = 32 * 2**20
MAX_ATTEMPTS = 4


@dataclass(frozen=True)
class Artifact:
    key: str
    source: str
    destination_relative: str
    expected_objects: int
    expected_bytes: int


ARTIFACTS = (
    Artifact(
        key="afdb",
        source=(
            "s3://marin-us-east-02a/MarinFold/"
            "exp232_sweep_cv1_decontam/tokenized/contacts_v1/afdb/2026.08.14"
        ),
        destination_relative="tokenized/contacts_v1/afdb/2026.08.14",
        expected_objects=755,
        expected_bytes=6_164_768_697,
    ),
    Artifact(
        key="esm",
        source=(
            "s3://marin-us-east-02a/MarinFold/"
            "exp232_sweep_cv1_decontam/tokenized/contacts_v1/esm/2026.08.14"
        ),
        destination_relative="tokenized/contacts_v1/esm/2026.08.14",
        expected_objects=10_019,
        expected_bytes=95_596_299_057,
    ),
    Artifact(
        key="validation",
        source=(
            "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/"
            "tokenized/contacts-v1-val/2026.07.25"
        ),
        destination_relative="tokenized/contacts-v1-val/2026.07.25",
        expected_objects=17,
        expected_bytes=66_618_514,
    ),
    Artifact(
        key="checkpoint",
        source=(
            "s3://marin-us-east-02a/MarinFold/"
            "exp232_train_cv1_decontam_recover/checkpoints/protein/"
            f"{SOURCE_RUN_ID}/{SOURCE_RUN_VERSION}/checkpoints/"
            f"step-{SOURCE_CHECKPOINT_STEP}"
        ),
        destination_relative=(
            "checkpoints/protein/exp232-trc-init/"
            f"{SOURCE_RUN_ID}/{INGRESS_VERSION}/checkpoints/"
            f"step-{SOURCE_CHECKPOINT_STEP}"
        ),
        expected_objects=28,
        expected_bytes=17_656_643_205,
    ),
)


def _strip_scheme(path: str, scheme: str) -> str:
    prefix = f"{scheme}://"
    if not path.startswith(prefix):
        raise ValueError(f"expected {prefix} path, got {path!r}")
    return path.removeprefix(prefix).rstrip("/")


def regional_root(region: str) -> str:
    try:
        bucket = REGION_BUCKETS[region]
    except KeyError:
        raise ValueError(f"unsupported region {region!r}") from None
    return f"gs://{bucket}/{REGIONAL_EXPERIMENT_RELATIVE}"


def destination(region: str, artifact: Artifact) -> str:
    return f"{regional_root(region)}/{artifact.destination_relative}"


def _s3(workers: int) -> s3fs.S3FileSystem:
    missing = [
        name for name in ("CW_KEY_ID", "CW_KEY_SECRET") if not os.environ.get(name)
    ]
    if missing:
        raise ValueError(f"missing CoreWeave credentials: {', '.join(missing)}")
    return s3fs.S3FileSystem(
        endpoint_url="https://cwobject.com",
        key=os.environ["CW_KEY_ID"],
        secret=os.environ["CW_KEY_SECRET"],
        client_kwargs={"region_name": "US-EAST-02A"},
        config_kwargs={
            "max_pool_connections": max(32, workers * 2),
            "s3": {"addressing_style": "virtual"},
        },
    )


def _inventory(fs, root: str, scheme: str) -> dict[str, tuple[str, int]]:
    stripped = _strip_scheme(root, scheme)
    if not fs.exists(stripped):
        return {}
    inventory: dict[str, tuple[str, int]] = {}
    for path, info in fs.find(stripped, detail=True, withdirs=False).items():
        if info.get("type") == "directory":
            continue
        relative = path.removeprefix(f"{stripped}/")
        if relative == path:
            raise ValueError(f"object {path!r} is outside inventory root {stripped!r}")
        inventory[relative] = (path, int(info["size"]))
    return inventory


def _fingerprint(inventory: dict[str, tuple[str, int]]) -> str:
    digest = hashlib.sha256()
    for relative, (_, size) in sorted(inventory.items()):
        digest.update(f"{relative}\0{size}\n".encode())
    return digest.hexdigest()


def _validate_expected(
    artifact: Artifact, inventory: dict[str, tuple[str, int]]
) -> None:
    observed = (len(inventory), sum(size for _, size in inventory.values()))
    expected = (artifact.expected_objects, artifact.expected_bytes)
    if observed != expected:
        raise RuntimeError(
            f"{artifact.key} source inventory changed: {observed=}, {expected=}"
        )


def _copy_one(
    s3: s3fs.S3FileSystem,
    gcs: gcsfs.GCSFileSystem,
    *,
    source_path: str,
    destination_path: str,
    expected_size: int,
) -> int:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            with (
                s3.open(
                    source_path,
                    "rb",
                    block_size=COPY_BUFFER_BYTES,
                    cache_type="none",
                ) as source_file,
                gcs.open(
                    destination_path,
                    "wb",
                    block_size=GCS_BLOCK_BYTES,
                ) as destination_file,
            ):
                shutil.copyfileobj(
                    source_file,
                    destination_file,
                    length=COPY_BUFFER_BYTES,
                )
            actual_size = int(gcs.info(destination_path)["size"])
            if actual_size != expected_size:
                raise OSError(
                    f"short GCS write for {destination_path}: "
                    f"{actual_size} != {expected_size}"
                )
            return actual_size
        except Exception as error:
            if attempt == MAX_ATTEMPTS:
                raise RuntimeError(
                    f"failed to ingress {source_path} after {attempt} attempts"
                ) from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def copy_artifact(
    *,
    s3: s3fs.S3FileSystem,
    gcs: gcsfs.GCSFileSystem,
    region: str,
    artifact: Artifact,
    workers: int,
) -> dict[str, int | str]:
    if not artifact.source.startswith("s3://"):
        raise ValueError("CW ingress source must be S3")
    destination_url = destination(region, artifact)
    if not destination_url.startswith(f"gs://{REGION_BUCKETS[region]}/"):
        raise ValueError(f"destination is not region-local: {destination_url}")

    source_inventory = _inventory(s3, artifact.source, "s3")
    _validate_expected(artifact, source_inventory)
    destination_inventory = _inventory(gcs, destination_url, "gs")
    pending = sorted(
        relative
        for relative, (_, expected_size) in source_inventory.items()
        if destination_inventory.get(relative, ("", -1))[1] != expected_size
    )
    total_pending_bytes = sum(source_inventory[path][1] for path in pending)
    print(
        f"[{region}/{artifact.key}] {len(pending)}/{len(source_inventory)} objects "
        f"({total_pending_bytes / 2**30:.2f} GiB) require direct CW ingress",
        flush=True,
    )

    destination_root = _strip_scheme(destination_url, "gs")
    completed_bytes = 0
    completed_objects = 0
    progress_lock = threading.Lock()

    def copy(relative: str) -> tuple[str, int]:
        source_path, expected_size = source_inventory[relative]
        target_path = f"{destination_root}/{relative}"
        size = _copy_one(
            s3,
            gcs,
            source_path=source_path,
            destination_path=target_path,
            expected_size=expected_size,
        )
        return relative, size

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(copy, relative) for relative in pending]
        for future in as_completed(futures):
            _, size = future.result()
            with progress_lock:
                completed_objects += 1
                completed_bytes += size
                if completed_objects % 100 == 0 or completed_objects == len(pending):
                    print(
                        f"[{region}/{artifact.key}] copied "
                        f"{completed_objects}/{len(pending)} objects, "
                        f"{completed_bytes / 2**30:.2f}/{total_pending_bytes / 2**30:.2f} GiB",
                        flush=True,
                    )

    observed = _inventory(gcs, destination_url, "gs")
    source_sizes = {path: size for path, (_, size) in source_inventory.items()}
    observed_sizes = {path: size for path, (_, size) in observed.items()}
    if observed_sizes != source_sizes:
        missing_or_wrong = sorted(
            path
            for path, size in source_sizes.items()
            if observed_sizes.get(path) != size
        )
        unexpected = sorted(set(observed_sizes) - set(source_sizes))
        raise RuntimeError(
            f"{region}/{artifact.key} inventory mismatch: "
            f"missing_or_wrong={missing_or_wrong[:5]}, unexpected={unexpected[:5]}"
        )
    fingerprint = _fingerprint(source_inventory)
    print(
        f"[{region}/{artifact.key}] COMPLETE: {len(observed)} objects, "
        f"{sum(observed_sizes.values()) / 2**30:.2f} GiB, sha256={fingerprint}",
        flush=True,
    )
    return {
        "objects": len(observed),
        "bytes": sum(observed_sizes.values()),
        "inventory_sha256": fingerprint,
        "source": artifact.source,
        "destination": destination_url,
    }


def copy_region(*, region: str, workers: int) -> None:
    s3 = _s3(workers)
    gcs = gcsfs.GCSFileSystem()
    results = {
        artifact.key: copy_artifact(
            s3=s3,
            gcs=gcs,
            region=region,
            artifact=artifact,
            workers=workers,
        )
        for artifact in ARTIFACTS
    }
    marker = {
        "contract": "direct-cw-s3-to-region-local-gcs",
        "ingress_version": INGRESS_VERSION,
        "region": region,
        "artifacts": results,
    }
    marker_path = _strip_scheme(
        f"{regional_root(region)}/ingress/{INGRESS_VERSION}.json",
        "gs",
    )
    gcs.pipe(marker_path, json.dumps(marker, indent=2, sort_keys=True).encode())
    print(f"[{region}] all ingress complete; marker=gs://{marker_path}", flush=True)


def verify_region(*, region: str, workers: int) -> None:
    s3 = _s3(workers)
    gcs = gcsfs.GCSFileSystem()
    for artifact in ARTIFACTS:
        source_inventory = _inventory(s3, artifact.source, "s3")
        _validate_expected(artifact, source_inventory)
        observed = _inventory(gcs, destination(region, artifact), "gs")
        source_sizes = {path: size for path, (_, size) in source_inventory.items()}
        observed_sizes = {path: size for path, (_, size) in observed.items()}
        if observed_sizes != source_sizes:
            raise RuntimeError(
                f"{region}/{artifact.key} has not passed ingress validation"
            )
        print(
            f"[{region}/{artifact.key}] VERIFIED: {len(observed)} objects, "
            f"sha256={_fingerprint(observed)}",
            flush=True,
        )


def show_plan() -> None:
    plan = {
        "contract": "each destination independently reads CoreWeave S3",
        "regions": {
            region: {
                artifact.key: {
                    "source": artifact.source,
                    "destination": destination(region, artifact),
                    "objects": artifact.expected_objects,
                    "bytes": artifact.expected_bytes,
                }
                for artifact in ARTIFACTS
            }
            for region in SUPPORTED_REGIONS
        },
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("plan")
    for command in ("copy-region", "verify-region"):
        regional = subparsers.add_parser(command)
        regional.add_argument("--region", choices=SUPPORTED_REGIONS, required=True)
        regional.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if getattr(args, "workers", 1) < 1:
        raise SystemExit("--workers must be positive")
    if args.command == "plan":
        show_plan()
    elif args.command == "copy-region":
        copy_region(region=args.region, workers=args.workers)
    else:
        verify_region(region=args.region, workers=args.workers)


if __name__ == "__main__":
    main()
