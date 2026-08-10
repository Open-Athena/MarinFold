# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy selected CoreWeave checkpoints to the exp199 Hugging Face repository."""

import argparse
import hashlib
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
from huggingface_hub import HfApi, RepoFile, hf_hub_download
from huggingface_hub.errors import RemoteEntryNotFoundError
from rigging.filesystem.s3_compat import configure_coreweave_s3

RUN_NAME = "prot-exp199-cw-cv1-s02-m1-p06-aug"
SOURCE_ROOT = (
    "s3://marin-us-east-02a/marin/protein-structure/MarinFold/"
    "exp199_optimize_contacts_v1_afdb_esm/checkpoints/protein/"
    f"{RUN_NAME}/2026.08.07.2"
)
DEFAULT_HF_REPO = "open-athena/marinfold-exp199"
DEFAULT_SCRATCH = f"/app/scratch/hf-upload/{RUN_NAME}"
COPY_CHUNK_BYTES = 64 * 1024 * 1024
MINIMUM_FREE_BYTES = 24 * 1024**3


@dataclass(frozen=True)
class TransferSpec:
    """One immutable source subtree and its matching HF destination."""

    relative_path: str
    expected_files: int
    expected_bytes: int

    @property
    def source_prefix(self) -> str:
        return f"{SOURCE_ROOT}/{self.relative_path}/"

    @property
    def destination_prefix(self) -> str:
        return f"{RUN_NAME}/{self.relative_path}"


@dataclass(frozen=True)
class SourceFile:
    """One source object with its destination-relative path."""

    uri: str
    relative_path: str
    size: int


@dataclass(frozen=True)
class StagedFile:
    """One locally staged object with its content digest."""

    source: SourceFile
    local_path: Path
    sha256: str


TRANSFERS = (
    TransferSpec(
        relative_path="checkpoints/step-116160",
        expected_files=25,
        expected_bytes=17_656_716_004,
    ),
    TransferSpec(
        relative_path="checkpoints/step-145199",
        expected_files=36,
        expected_bytes=17_656_751_789,
    ),
    TransferSpec(
        relative_path="hf/step-145199",
        expected_files=6,
        expected_bytes=5_885_614_712,
    ),
)


def source_inventory(fs: Any, spec: TransferSpec) -> list[SourceFile]:
    """List and validate the exact source objects selected for transfer."""

    source_key_prefix = spec.source_prefix.removeprefix("s3://")
    details = fs.find(spec.source_prefix, detail=True)
    files = []
    for key, info in sorted(details.items()):
        if info.get("type") != "file":
            continue
        if not key.startswith(source_key_prefix):
            raise ValueError(f"source object escaped prefix: {key}")
        relative_path = key.removeprefix(source_key_prefix)
        if not relative_path:
            continue
        files.append(
            SourceFile(
                uri=f"s3://{key}",
                relative_path=relative_path,
                size=int(info["size"]),
            )
        )
    total_bytes = sum(item.size for item in files)
    if len(files) != spec.expected_files or total_bytes != spec.expected_bytes:
        raise ValueError(
            f"source inventory changed for {spec.relative_path}: "
            f"{len(files)} files / {total_bytes} bytes; expected "
            f"{spec.expected_files} files / {spec.expected_bytes} bytes"
        )
    return files


def remote_files(api: HfApi, repo_id: str, prefix: str) -> dict[str, RepoFile]:
    """Return all HF files below a destination prefix, keyed relatively."""

    try:
        entries = api.list_repo_tree(
            repo_id,
            path_in_repo=prefix,
            repo_type="model",
            recursive=True,
            expand=True,
        )
        return {
            entry.path.removeprefix(f"{prefix}/"): entry
            for entry in entries
            if isinstance(entry, RepoFile)
        }
    except RemoteEntryNotFoundError:
        return {}


def safe_scratch(path: str) -> Path:
    """Resolve a task-local scratch directory and reject broad targets."""

    scratch = Path(path).resolve()
    if scratch in {Path("/"), Path("/app"), Path("/app/scratch")}:
        raise ValueError(f"scratch path is too broad: {scratch}")
    if len(scratch.parts) < 4:
        raise ValueError(f"scratch path is too shallow: {scratch}")
    return scratch


def clear_batch(batch_root: Path, scratch: Path) -> None:
    """Remove one verified staging batch without touching its parent."""

    resolved_batch = batch_root.resolve()
    resolved_scratch = scratch.resolve()
    if resolved_scratch not in resolved_batch.parents:
        raise ValueError(f"batch path escaped scratch: {resolved_batch}")
    if resolved_batch.exists():
        shutil.rmtree(resolved_batch)


def stage_files(
    fs: Any,
    batch_root: Path,
    spec: TransferSpec,
    sources: list[SourceFile],
) -> list[StagedFile]:
    """Download and hash one subtree serially while preserving its path."""

    destination = batch_root / RUN_NAME / spec.relative_path
    staged = []
    for index, source in enumerate(sources, start=1):
        local_path = destination / source.relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        print(
            f"[download {index}/{len(sources)}] {source.uri} ({source.size} bytes)",
            flush=True,
        )
        with fs.open(source.uri, "rb") as remote, local_path.open("wb") as local:
            while chunk := remote.read(COPY_CHUNK_BYTES):
                local.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        if size != source.size:
            raise ValueError(
                f"download size mismatch for {source.uri}: {size} != {source.size}"
            )
        staged.append(
            StagedFile(
                source=source,
                local_path=local_path,
                sha256=digest.hexdigest(),
            )
        )
    return staged


def remote_sha256(
    entry: RepoFile,
    *,
    api: HfApi,
    repo_id: str,
    cache_dir: Path,
) -> str:
    """Return the uploaded content digest without redownloading LFS objects."""

    if entry.lfs is not None:
        return entry.lfs.sha256
    path = hf_hub_download(
        repo_id,
        entry.path,
        repo_type="model",
        token=api.token,
        cache_dir=cache_dir,
        force_download=True,
    )
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while chunk := source.read(COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def validate_remote(
    api: HfApi,
    repo_id: str,
    spec: TransferSpec,
    staged: list[StagedFile],
    cache_dir: Path,
    *,
    require_complete: bool,
) -> None:
    """Reject destination differences and optionally require every file."""

    remote = remote_files(api, repo_id, spec.destination_prefix)
    expected = {item.source.relative_path: item for item in staged}
    extra = sorted(set(remote) - set(expected))
    missing = sorted(set(expected) - set(remote))
    if extra:
        raise ValueError(
            f"unexpected destination files in {spec.destination_prefix}: {extra}"
        )
    if require_complete and missing:
        raise ValueError(
            f"missing destination files in {spec.destination_prefix}: {missing}"
        )
    for relative_path, entry in remote.items():
        item = expected[relative_path]
        if entry.size != item.source.size:
            raise ValueError(
                f"destination size mismatch for {entry.path}: "
                f"{entry.size} != {item.source.size}"
            )
        digest = remote_sha256(
            entry,
            api=api,
            repo_id=repo_id,
            cache_dir=cache_dir,
        )
        if digest != item.sha256:
            raise ValueError(
                f"destination hash mismatch for {entry.path}: {digest} != {item.sha256}"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO)
    parser.add_argument("--scratch", default=DEFAULT_SCRATCH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.repo_info(args.hf_repo, repo_type="model")

    inventories = []
    for spec in TRANSFERS:
        sources = source_inventory(fs, spec)
        existing = remote_files(api, args.hf_repo, spec.destination_prefix)
        print(
            f"[inventory] {spec.source_prefix} -> "
            f"hf://models/{args.hf_repo}/{spec.destination_prefix}: "
            f"{len(sources)} files / {sum(item.size for item in sources)} bytes; "
            f"{len(existing)} destination files",
            flush=True,
        )
        inventories.append((spec, sources))
    if args.dry_run:
        return 0
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN must contain the open-athena write token")

    scratch = safe_scratch(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(scratch).free < MINIMUM_FREE_BYTES:
        raise ValueError(
            f"scratch has less than {MINIMUM_FREE_BYTES} bytes free: {scratch}"
        )
    batch_root = scratch / "batch"
    verify_cache = scratch / "verify-cache"

    for spec, sources in inventories:
        clear_batch(batch_root, scratch)
        staged = stage_files(fs, batch_root, spec, sources)
        validate_remote(
            api,
            args.hf_repo,
            spec,
            staged,
            verify_cache,
            require_complete=False,
        )
        print(f"[upload] {spec.destination_prefix}", flush=True)
        api.upload_large_folder(
            args.hf_repo,
            folder_path=batch_root,
            repo_type="model",
            num_workers=1,
            print_report=True,
            print_report_every=60,
        )
        validate_remote(
            api,
            args.hf_repo,
            spec,
            staged,
            verify_cache,
            require_complete=True,
        )
        print(
            f"[verified] {spec.destination_prefix}: "
            f"{len(staged)} files / {sum(item.source.size for item in staged)} bytes",
            flush=True,
        )
        clear_batch(batch_root, scratch)

    print(
        f"[complete] https://huggingface.co/{args.hf_repo}/tree/main/{RUN_NAME}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
