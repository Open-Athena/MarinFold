# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream immutable Hugging Face artifacts directly into CoreWeave S3."""

import hashlib
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import fsspec
from huggingface_hub import snapshot_download

from checkpoint_specs import Checkpoint, HfFile, model_s3_uri

CHUNK_BYTES = 8 * 1024**2
PROGRESS_BYTES = 1024**3


def hf_file_url(checkpoint: Checkpoint, file: HfFile) -> str:
    """Return an immutable resolve URL for one checkpoint file."""

    path = urllib.parse.quote(f"{checkpoint.hf_subfolder}/{file.name}", safe="/")
    return (
        f"https://huggingface.co/{checkpoint.hf_repo_id}/resolve/"
        f"{checkpoint.hf_revision}/{path}"
    )


def authorization_headers() -> dict[str, str]:
    """Return optional HF authentication without requiring it for public files."""

    token = os.environ.get("HF_TOKEN")
    headers = {"User-Agent": "MarinFold-exp199-rollout-v2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def expected_manifest(checkpoint: Checkpoint) -> dict:
    """Return the immutable source identity expected at the mirror."""

    return {
        "source": {
            "repo_id": checkpoint.hf_repo_id,
            "revision": checkpoint.hf_revision,
            "subfolder": checkpoint.hf_subfolder,
        },
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
    """Verify an existing CoreWeave checkpoint against the pinned HF manifest.

    This streams each object through the CoreWeave-resident driver and writes only a
    small verification record. It does not copy the checkpoint.
    """

    if not source_uri.startswith("s3://") or source_uri.startswith("gs://"):
        raise ValueError(f"expected an existing CoreWeave S3 checkpoint: {source_uri}")
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
            f"CoreWeave checkpoint file set differs from pinned HF: "
            f"{actual_sizes} != {expected_sizes}"
        )

    verified_files = []
    for entry in sorted(objects, key=lambda item: item["name"]):
        name = os.path.basename(entry["name"])
        file = expected_files[name]
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1(f"blob {file.size}\0".encode())
        read = 0
        next_progress = PROGRESS_BYTES
        start = time.monotonic()
        with filesystem.open(entry["name"], "rb") as handle:
            while chunk := handle.read(CHUNK_BYTES):
                sha256.update(chunk)
                git_sha1.update(chunk)
                read += len(chunk)
                if read >= next_progress:
                    print(
                        f"[verify] {source_uri}/{name}: "
                        f"{read / 2**30:.1f}/{file.size / 2**30:.1f} GiB",
                        flush=True,
                    )
                    next_progress += PROGRESS_BYTES
        actual_digest = (
            sha256.hexdigest() if file.digest_kind == "sha256" else git_sha1.hexdigest()
        )
        if read != file.size or actual_digest != file.digest:
            raise ValueError(
                f"CoreWeave object does not match pinned HF {file.digest_kind}: "
                f"{source_uri}/{name} size={read} digest={actual_digest}"
            )
        verified_files.append(
            {
                "name": name,
                "size": read,
                "digest": actual_digest,
                "digest_kind": file.digest_kind,
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        print(f"[verify] matched pinned HF: {source_uri}/{name}", flush=True)

    record = {
        **expected,
        "coreweave_uri": source_uri,
        "verified_at": datetime.now(UTC).isoformat(),
        "verification": (
            "streamed and hashed inside CoreWeave; no checkpoint copy was created"
        ),
        "verified_files": verified_files,
    }
    verification_filesystem, verification_path = fsspec.core.url_to_fs(verification_uri)
    write_json(verification_filesystem, verification_path, record)
    return record


def read_json(filesystem, path: str) -> dict | None:
    """Read JSON when it exists."""

    if not filesystem.exists(path):
        return None
    with filesystem.open(path, "rt") as file:
        return json.load(file)


def write_json(filesystem, path: str, data: dict) -> None:
    """Write canonical JSON to object storage."""

    with filesystem.open(path, "wt") as file:
        json.dump(data, file, indent=2, sort_keys=True)
        file.write("\n")


def verified_file_exists(
    filesystem, destination: str, marker: str, expected: dict
) -> bool:
    """Return whether a prior HF stream verified this exact destination."""

    record = read_json(filesystem, marker)
    if record != expected or not filesystem.exists(destination):
        return False
    return filesystem.info(destination)["size"] == expected["size"]


def copy_https_to_s3(
    *,
    url: str,
    filesystem,
    destination: str,
    expected_size: int,
    expected_digest: str,
    digest_kind: str,
) -> dict:
    """Stream one HTTPS object to S3 while validating its immutable digest."""

    if destination.startswith("gs://") or url.startswith("gs://"):
        raise ValueError("GCS is forbidden in the HF-to-CoreWeave staging path")
    request = urllib.request.Request(url, headers=authorization_headers())
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {expected_size}\0".encode())
    copied = 0
    next_progress = PROGRESS_BYTES
    start = time.monotonic()
    if filesystem.exists(destination):
        filesystem.rm(destination)

    with (
        urllib.request.urlopen(request, timeout=180) as response,
        filesystem.open(destination, "wb", block_size=64 * 1024**2) as output,
    ):
        while chunk := response.read(CHUNK_BYTES):
            output.write(chunk)
            sha256.update(chunk)
            git_sha1.update(chunk)
            copied += len(chunk)
            if copied >= next_progress:
                print(
                    f"[stage] {destination}: {copied / 2**30:.1f}/{expected_size / 2**30:.1f} GiB",
                    flush=True,
                )
                next_progress += PROGRESS_BYTES

    actual_digest = (
        sha256.hexdigest() if digest_kind == "sha256" else git_sha1.hexdigest()
    )
    if copied != expected_size:
        raise ValueError(f"size mismatch for {url}: {copied} != {expected_size}")
    if actual_digest != expected_digest:
        raise ValueError(
            f"{digest_kind} mismatch for {url}: {actual_digest} != {expected_digest}"
        )
    stored_size = filesystem.info(destination)["size"]
    if stored_size != expected_size:
        raise ValueError(
            f"stored size mismatch for {destination}: {stored_size} != {expected_size}"
        )
    elapsed = time.monotonic() - start
    print(
        f"[stage] verified {destination}: {expected_size / 2**30:.2f} GiB in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "size": copied,
        "digest": actual_digest,
        "digest_kind": digest_kind,
        "elapsed_seconds": elapsed,
        "source_url": url,
    }


def digest_file(path: Path, *, expected_size: int, digest_kind: str) -> str:
    """Hash one HF download as either content SHA-256 or Git blob SHA-1."""

    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(f"blob {expected_size}\0".encode())
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            sha256.update(chunk)
            git_sha1.update(chunk)
    return sha256.hexdigest() if digest_kind == "sha256" else git_sha1.hexdigest()


def upload_verified_file(
    *,
    source: Path,
    filesystem,
    destination: str,
    expected: HfFile,
) -> dict:
    """Verify a local HF download, upload it, and verify its stored size."""

    size = source.stat().st_size
    if size != expected.size:
        raise ValueError(f"size mismatch for {source}: {size} != {expected.size}")
    actual_digest = digest_file(
        source,
        expected_size=expected.size,
        digest_kind=expected.digest_kind,
    )
    if actual_digest != expected.digest:
        raise ValueError(
            f"{expected.digest_kind} mismatch for {source}: {actual_digest} != {expected.digest}"
        )
    start = time.monotonic()
    filesystem.put_file(str(source), destination)
    if filesystem.info(destination)["size"] != expected.size:
        raise ValueError(f"stored size mismatch for {destination}")
    elapsed = time.monotonic() - start
    print(
        f"[stage] uploaded and verified {destination}: {size / 2**30:.2f} GiB "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    return {
        "size": size,
        "digest": actual_digest,
        "digest_kind": expected.digest_kind,
        "elapsed_seconds": elapsed,
    }


def mirror_checkpoint(run_id: str, checkpoint: Checkpoint) -> dict:
    """Mirror and verify one complete HF export in the shared CW namespace."""

    destination_uri = model_s3_uri(run_id, checkpoint)
    filesystem, destination_root = fsspec.core.url_to_fs(destination_uri)
    identity_path = f"{destination_root}/identity.json"
    expected = expected_manifest(checkpoint)
    existing = read_json(filesystem, identity_path)
    if (
        existing is not None
        and {key: existing.get(key) for key in expected} == expected
        and all(
            filesystem.exists(f"{destination_root}/{file.name}")
            and filesystem.info(f"{destination_root}/{file.name}")["size"] == file.size
            for file in checkpoint.files
        )
    ):
        print(f"[stage] verified mirror already exists: {destination_uri}", flush=True)
        return existing

    streamed: dict[str, dict] = {}
    pending: list[tuple[HfFile, str, str, dict]] = []
    for file in checkpoint.files:
        destination = f"{destination_root}/{file.name}"
        marker = f"{destination_root}/verified/{file.name}.json"
        file_expected = {
            "name": file.name,
            "size": file.size,
            "digest": file.digest,
            "digest_kind": file.digest_kind,
            "source_url": hf_file_url(checkpoint, file),
        }
        if verified_file_exists(filesystem, destination, marker, file_expected):
            print(
                f"[stage] verified file already exists: {destination_uri}/{file.name}",
                flush=True,
            )
            streamed[file.name] = file_expected
            continue
        pending.append((file, destination, marker, file_expected))

    if pending:
        download_start = time.monotonic()
        with TemporaryDirectory(prefix="exp199-hf-") as temporary_directory:
            local_root = Path(
                snapshot_download(
                    repo_id=checkpoint.hf_repo_id,
                    revision=checkpoint.hf_revision,
                    allow_patterns=[
                        f"{checkpoint.hf_subfolder}/{file.name}"
                        for file, _, _, _ in pending
                    ],
                    local_dir=temporary_directory,
                    max_workers=8,
                    token=os.environ.get("HF_TOKEN"),
                )
            )
            print(
                f"[stage] downloaded {len(pending)} file(s) for {checkpoint.label} "
                f"from pinned HF in {time.monotonic() - download_start:.1f}s",
                flush=True,
            )
            for file, destination, marker, file_expected in pending:
                source = local_root / checkpoint.hf_subfolder / file.name
                result = upload_verified_file(
                    source=source,
                    filesystem=filesystem,
                    destination=destination,
                    expected=file,
                )
                result["source_url"] = file_expected["source_url"]
                write_json(filesystem, marker, file_expected)
                streamed[file.name] = result

    identity = {
        **expected,
        "destination": destination_uri,
        "staged_at": datetime.now(UTC).isoformat(),
        "verified_from_huggingface": True,
        "transfer": (
            "pinned Hugging Face snapshot download inside CoreWeave, then CoreWeave S3 upload"
        ),
        "streamed": streamed,
    }
    write_json(filesystem, identity_path, identity)
    return identity


def mirror_public_input(
    *,
    url: str,
    destination_uri: str,
    expected_size: int,
    expected_sha256: str,
) -> dict:
    """Mirror and verify one public eval input from HF to CoreWeave S3."""

    filesystem, destination = fsspec.core.url_to_fs(destination_uri)
    marker = f"{destination}.verified.json"
    expected = {
        "size": expected_size,
        "digest": expected_sha256,
        "digest_kind": "sha256",
        "source_url": url,
    }
    if verified_file_exists(filesystem, destination, marker, expected):
        print(f"[stage] verified input already exists: {destination_uri}", flush=True)
        return expected
    result = copy_https_to_s3(
        url=url,
        filesystem=filesystem,
        destination=destination,
        expected_size=expected_size,
        expected_digest=expected_sha256,
        digest_kind="sha256",
    )
    write_json(filesystem, marker, expected)
    return result
