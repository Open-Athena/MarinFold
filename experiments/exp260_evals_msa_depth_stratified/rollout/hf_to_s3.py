# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify CoreWeave checkpoints and mirror small public eval inputs."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from datetime import UTC, datetime

import fsspec
from checkpoint_specs import Checkpoint

CHUNK_BYTES = 8 * 1024**2
PROGRESS_BYTES = 1024**3


def write_json(filesystem, path: str, data: dict) -> None:
    """Write canonical JSON through fsspec."""

    with filesystem.open(path, "wt") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def expected_manifest(checkpoint: Checkpoint) -> dict:
    """Return the pinned HF-directory identity for one checkpoint."""

    return {
        "source": {"kind": "coreweave-s3", "uri": checkpoint.coreweave_uri},
        "checkpoint": {
            "label": checkpoint.label,
            "run_name": checkpoint.run_name,
            "step": checkpoint.step,
        },
        "files": [
            {
                "name": file.name,
                "size": file.size,
                "digest": file.digest,
                "digest_kind": file.digest_kind,
            }
            for file in checkpoint.files
        ],
    }


def verify_checkpoint_at_uri(
    *, checkpoint: Checkpoint, source_uri: str, verification_uri: str
) -> dict:
    """Verify a pre-existing CoreWeave HF directory without copying it."""

    if not source_uri.startswith("s3://") or source_uri.startswith("gs://"):
        raise ValueError(f"expected CoreWeave S3, received {source_uri!r}")
    if not checkpoint.files:
        raise ValueError(f"{checkpoint.label} has no pinned HF export manifest")

    filesystem, source_root = fsspec.core.url_to_fs(source_uri)
    expected = expected_manifest(checkpoint)
    expected_files = {file.name: file for file in checkpoint.files}
    objects = [
        entry
        for entry in filesystem.ls(source_root, detail=True)
        if entry["type"] == "file"
    ]
    actual_sizes = {os.path.basename(entry["name"]): entry["size"] for entry in objects}
    expected_sizes = {name: file.size for name, file in expected_files.items()}
    if actual_sizes != expected_sizes:
        raise ValueError(
            f"CoreWeave checkpoint file set changed: {actual_sizes} != {expected_sizes}"
        )

    verified_files = []
    for entry in sorted(objects, key=lambda item: item["name"]):
        name = os.path.basename(entry["name"])
        file = expected_files[name]
        read = 0
        started = time.monotonic()
        if file.digest_kind == "s3-etag":
            actual_digest = str(entry.get("ETag") or entry.get("etag") or "").strip('"')
            if not actual_digest:
                raise ValueError(f"CoreWeave object has no S3 ETag: {entry}")
            read = entry["size"]
        else:
            sha256 = hashlib.sha256()
            git_sha1 = hashlib.sha1(f"blob {file.size}\0".encode())
            next_progress = PROGRESS_BYTES
            with filesystem.open(entry["name"], "rb") as handle:
                while chunk := handle.read(CHUNK_BYTES):
                    sha256.update(chunk)
                    git_sha1.update(chunk)
                    read += len(chunk)
                    if read >= next_progress:
                        print(
                            f"[verify] {name}: {read / 2**30:.1f}/"
                            f"{file.size / 2**30:.1f} GiB",
                            flush=True,
                        )
                        next_progress += PROGRESS_BYTES
            actual_digest = (
                sha256.hexdigest()
                if file.digest_kind == "sha256"
                else git_sha1.hexdigest()
            )
        if read != file.size or actual_digest != file.digest:
            raise ValueError(
                f"checkpoint mismatch for {source_uri}/{name}: "
                f"size={read}, {file.digest_kind}={actual_digest}"
            )
        verified_files.append(
            {
                "name": name,
                "size": read,
                "digest": actual_digest,
                "digest_kind": file.digest_kind,
                "elapsed_seconds": time.monotonic() - started,
            }
        )

    record = {
        **expected,
        "coreweave_uri": source_uri,
        "verified_at": datetime.now(UTC).isoformat(),
        "verification": "verified in place; no checkpoint copy was created",
        "verified_files": verified_files,
    }
    verification_filesystem, verification_path = fsspec.core.url_to_fs(verification_uri)
    write_json(verification_filesystem, verification_path, record)
    return record


def verify_levanter_source(
    *, checkpoint: Checkpoint, verification_uri: str
) -> dict | None:
    """Verify the recursive source prefix used for an eval-local HF export."""

    if checkpoint.levanter_source_uri is None:
        return None
    filesystem, root = fsspec.core.url_to_fs(checkpoint.levanter_source_uri)
    details = filesystem.find(root, detail=True)
    objects = []
    for name, entry in details.items():
        if entry["type"] != "file":
            continue
        objects.append(
            {
                "name": name.removeprefix(root.rstrip("/") + "/"),
                "size": int(entry["size"]),
                "etag": str(entry.get("ETag") or entry.get("etag") or "").strip('"'),
            }
        )
    objects.sort(key=lambda item: item["name"])
    canonical = json.dumps(objects, sort_keys=True, separators=(",", ":"))
    actual = {
        "objects": len(objects),
        "bytes": sum(item["size"] for item in objects),
        "manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
    }
    expected = {
        "objects": checkpoint.levanter_source_objects,
        "bytes": checkpoint.levanter_source_bytes,
        "manifest_sha256": checkpoint.levanter_source_manifest_sha256,
    }
    if actual != expected:
        raise ValueError(f"Levanter source identity changed: {actual} != {expected}")
    record = {
        "source_uri": checkpoint.levanter_source_uri,
        "verified_at": datetime.now(UTC).isoformat(),
        **actual,
        "metadata": next(
            (item for item in objects if item["name"] == "metadata.json"), None
        ),
    }
    out_filesystem, out_path = fsspec.core.url_to_fs(verification_uri)
    write_json(out_filesystem, out_path, record)
    return record


def mirror_public_input(
    *, url: str, destination_uri: str, expected_size: int, expected_sha256: str
) -> dict:
    """Stream one immutable public input directly into CoreWeave S3."""

    if url.startswith("gs://") or destination_uri.startswith("gs://"):
        raise ValueError("GCS is forbidden for this evaluation")
    filesystem, destination = fsspec.core.url_to_fs(destination_uri)
    marker = f"{destination}.verified.json"
    expected = {
        "size": expected_size,
        "digest": expected_sha256,
        "digest_kind": "sha256",
        "source_url": url,
    }
    if filesystem.exists(destination) and filesystem.exists(marker):
        with filesystem.open(marker, "rt") as handle:
            prior = json.load(handle)
        if prior == expected and filesystem.info(destination)["size"] == expected_size:
            return expected

    request = urllib.request.Request(url, headers={"User-Agent": "MarinFold-exp232"})
    digest = hashlib.sha256()
    copied = 0
    with (
        urllib.request.urlopen(request) as source,
        filesystem.open(destination, "wb") as target,
    ):
        while chunk := source.read(CHUNK_BYTES):
            target.write(chunk)
            digest.update(chunk)
            copied += len(chunk)
    actual_sha256 = digest.hexdigest()
    if copied != expected_size or actual_sha256 != expected_sha256:
        filesystem.rm(destination)
        raise ValueError(f"public input changed: size={copied}, sha256={actual_sha256}")
    write_json(filesystem, marker, expected)
    return expected
