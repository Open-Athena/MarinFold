# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror the published #294 complex corpus into CoreWeave storage, in-region.

The corpus is 19.6 GB on the public HF bucket. The workstation uplink is about
2 MB/s, so it must not pass through it; this runs on a CoreWeave CPU pod, where
the download is a public HTTPS read and the write is a same-region S3 PUT.

The published shards split here rather than later: shards other than
`VALIDATION_SHARD` land under `documents/train/` and that one lands under
`documents/validation/`, so `prepare.py`'s two sources are plain globs and no
downstream step has to remember which shard is held out.

Emits `stage-manifest.json` with every shard's published path, size, digest and
parquet row count. The row counts are what pin the train/validation document
totals in `config.py` -- they are read from the staged footers, not assumed from
the corpus's nominal 20,000-row shard size.
"""

import hashlib
import json
import logging
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import click
import fsspec
import pyarrow.parquet as pq
from huggingface_hub import download_bucket_files, list_bucket_tree

from experiments.exp343_models_complex_corpus_training.config import (
    COMPLEX_BUCKET,
    COMPLEX_BUCKET_PREFIX,
    COMPLEX_DOCUMENTS,
    COMPLEX_PUBLISHED_TOKENS,
    COMPLEX_SHARDS,
    STAGE_MANIFEST,
    STAGED_TRAIN,
    STAGED_VALIDATION,
    VALIDATION_SHARD,
)

#: HF throttles a bucket prefix under heavy concurrency and this is one pod, so
#: the win from more threads is bounded by the S3 write side anyway.
WORKERS = 12
#: Columns the training pipeline reads. The rest of the published schema is
#: provenance and sampling metadata that tokenization never touches, but it is
#: cheap to keep and losing it would make the staged copy non-reproducible
#: against the published one, so shards are mirrored byte-for-byte.
REQUIRED_COLUMNS = ("document", "document_id", "num_tokens", "source_arm", "sha1")


def _log(message: str) -> None:
    print(f"[exp343-stage] {message}", file=sys.stderr, flush=True)


def published_shards(token: str | bool | None) -> list[tuple[int, str, int]]:
    """Every published shard as `(index, bucket path, size)`, index-ordered.

    The index comes from the `shard-NNNNN-of-00171` name rather than from
    enumeration order, because the held-out shard is selected by index and a
    listing that silently reordered would move it.
    """
    prefix = f"{COMPLEX_BUCKET_PREFIX}/train/"
    shards = []
    for entry in list_bucket_tree(
        COMPLEX_BUCKET, prefix, recursive=True, token=token
    ):
        name = Path(entry.path).name
        if not name.startswith("shard-") or not name.endswith(".parquet"):
            raise ValueError(f"Unexpected file under {prefix}: {entry.path}")
        index, _, total = name[len("shard-") : -len(".parquet")].partition("-of-")
        if int(total) != COMPLEX_SHARDS:
            raise ValueError(f"{name} claims {total} shards, expected {COMPLEX_SHARDS}")
        shards.append((int(index), entry.path, entry.size))
    shards.sort()
    if [index for index, _, _ in shards] != list(range(COMPLEX_SHARDS)):
        raise ValueError(f"Published shard indices are not 0..{COMPLEX_SHARDS - 1}")
    return shards


def destination(index: int, name: str) -> str:
    root = STAGED_VALIDATION if index == VALIDATION_SHARD else STAGED_TRAIN
    return f"{root}/{name}"


def stage_one(index: int, path: str, size: int, token: str | bool | None) -> dict:
    """Copy one shard through local disk and audit the staged result.

    Already-staged shards are re-audited rather than re-copied, so a preempted
    job resumes instead of restarting. The audit reads the staged object, not
    the local temporary, so a truncated write cannot pass.
    """
    name = Path(path).name
    target = destination(index, name)
    fs, target_key = fsspec.core.url_to_fs(target)
    if not (fs.exists(target_key) and fs.size(target_key) == size):
        with tempfile.TemporaryDirectory() as scratch:
            local = Path(scratch) / name
            download_bucket_files(
                COMPLEX_BUCKET, files=[(path, str(local))], token=token
            )
            if local.stat().st_size != size:
                raise ValueError(
                    f"{name}: downloaded {local.stat().st_size} bytes, "
                    f"listing says {size}"
                )
            blob = local.read_bytes()
        with fs.open(target_key, "wb") as handle:
            handle.write(blob)
        if fs.size(target_key) != size:
            raise ValueError(f"{name}: staged {fs.size(target_key)} bytes, want {size}")
    digest = hashlib.sha256()
    with fs.open(target_key, "rb") as handle:
        metadata = pq.ParquetFile(handle).metadata
        handle.seek(0)
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    columns = set(metadata.schema.names)
    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    if missing:
        raise ValueError(f"{name}: staged copy is missing columns {missing}")
    return {
        "index": index,
        "published": path,
        "staged": target,
        "bytes": size,
        "rows": metadata.num_rows,
        "sha256": digest.hexdigest(),
    }


def stage(token: str | bool | None) -> dict:
    """Mirror every published shard and reconcile the staged corpus."""
    shards = published_shards(token)
    total_bytes = sum(size for _, _, size in shards)
    _log(f"{len(shards)} shards, {total_bytes / 1e9:.1f} GB -> {STAGED_TRAIN}")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [
            pool.submit(stage_one, index, path, size, token)
            for index, path, size in shards
        ]
        rows = []
        for done, future in enumerate(futures, 1):
            rows.append(future.result())
            if done % 20 == 0 or done == len(futures):
                _log(f"{done}/{len(futures)} shards staged")
    rows.sort(key=lambda row: row["index"])
    documents = sum(row["rows"] for row in rows)
    if documents != COMPLEX_DOCUMENTS:
        raise ValueError(
            f"Staged corpus has {documents} documents, published corpus has "
            f"{COMPLEX_DOCUMENTS}"
        )
    train = sum(row["rows"] for row in rows if row["index"] != VALIDATION_SHARD)
    manifest = {
        "bucket": COMPLEX_BUCKET,
        "bucket_prefix": COMPLEX_BUCKET_PREFIX,
        "documents": documents,
        "published_tokens": COMPLEX_PUBLISHED_TOKENS,
        "validation_shard": VALIDATION_SHARD,
        "train_documents": train,
        "validation_documents": documents - train,
        "train_shards": len(rows) - 1,
        "bytes": total_bytes,
        "shards": rows,
    }
    with fsspec.open(STAGE_MANIFEST, "w") as handle:
        json.dump(manifest, handle, indent=2)
    _log(
        f"STAGED {documents} documents: {train} train across {len(rows) - 1} shards, "
        f"{documents - train} validation in shard {VALIDATION_SHARD}"
    )
    _log(f"manifest -> {STAGE_MANIFEST}")
    return manifest


@click.command()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    # The bucket is public, so `False` -- explicitly anonymous -- is the correct
    # fallback. A token is used when one is present only because an
    # authenticated read is not rate-limited as aggressively.
    stage(os.environ.get("HF_TOKEN") or False)


if __name__ == "__main__":
    main()
