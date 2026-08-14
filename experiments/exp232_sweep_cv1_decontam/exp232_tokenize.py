# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror exp232 training data to CoreWeave S3, then tokenize the S3 copy.

The mirror is a hard phase boundary: tokenization starts only after every
Hugging Face bucket object is present at the expected size in CoreWeave S3.
Tokenization workers therefore read only ``s3://marin-us-east-02a`` paths.
"""

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime

import click
import fsspec
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from huggingface_hub import HfFileSystem
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, ShardedCacheLayout
from marin.processing.tokenize._core import (
    bundle_files_by_size,
    compute_target_group_bytes,
    drop_sidecars,
    expand_tokenize_paths,
    glob_with_sizes,
    parquet_window_hint,
    tokenize_pipeline,
)
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.processing.tokenize.store_builder import (
    build_from_datasets,
    write_stats_json,
)
from marin.processing.tokenize.tokenize import TokenizeConfig
from rigging.filesystem import StoragePath, marin_prefix, prefix_join
from zephyr.dataset import Dataset, FileEntry
from zephyr.execution import ZephyrContext
from zephyr.readers import load_file

logger = logging.getLogger(__name__)

CACHE_VERSION = "2026.08.14"
TOKENIZER = "eczech/contacts-v1-tokenizer-5d68a24a899f"
TEXT_KEY = "document"

EXPERIMENT_PREFIX = "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam"
DATA_PREFIX = f"{EXPERIMENT_PREFIX}/data"
TOKENIZED_PREFIX = f"{EXPERIMENT_PREFIX}/tokenized/contacts_v1"

AFDB_DATA = f"{DATA_PREFIX}/afdb"
ESM_DATA = f"{DATA_PREFIX}/esm"
AFDB_CACHE = f"{TOKENIZED_PREFIX}/afdb/{CACHE_VERSION}"
ESM_CACHE = f"{TOKENIZED_PREFIX}/esm/{CACHE_VERSION}"

AFDB_HF_PREFIX = (
    "buckets/open-athena/MarinFold/data/document_structures/contacts_v1_decontam/train"
)
ESM_HF_PREFIX = (
    "buckets/open-athena/MarinFold/data/document_structures/"
    "contacts_v1_esm_atlas_decontam/train"
)
MIRROR_MANIFEST = f"{DATA_PREFIX}/mirror-{CACHE_VERSION}.json"
SMOKE_PREFIX = f"{EXPERIMENT_PREFIX}/tmp/tokenization-smoke/{CACHE_VERSION}"
SMOKE_MANIFEST = f"{SMOKE_PREFIX}/mirror.json"

COPY_BUFFER_BYTES = 32 * 1024 * 1024
COPY_RETRIES = 5
COORDINATOR_RESOURCES = ResourceConfig(
    cpu=1,
    ram="6g",
    disk="16g",
    preemptible=False,
)


@dataclass(frozen=True)
class Corpus:
    key: str
    hf_prefix: str
    s3_prefix: str
    cache_path: str
    expected_documents: int
    expected_parquet_files: int
    expected_parquet_bytes: int


@dataclass(frozen=True)
class MirrorFile:
    relative_path: str
    size: int
    xet_hash: str | None


CORPORA = (
    Corpus(
        "afdb",
        AFDB_HF_PREFIX,
        AFDB_DATA,
        AFDB_CACHE,
        expected_documents=3_963_003,
        expected_parquet_files=2_067,
        expected_parquet_bytes=12_128_381_115,
    ),
    Corpus(
        "esm",
        ESM_HF_PREFIX,
        ESM_DATA,
        ESM_CACHE,
        expected_documents=65_553_178,
        expected_parquet_files=3_338,
        expected_parquet_bytes=130_662_915_211,
    ),
)


def _smoke_corpora() -> tuple[Corpus, ...]:
    """Point the real corpus definitions at isolated, disposable S3 outputs."""
    return tuple(
        replace(
            corpus,
            s3_prefix=f"{SMOKE_PREFIX}/data/{corpus.key}",
            cache_path=f"{SMOKE_PREFIX}/tokenized/contacts_v1/{corpus.key}",
            expected_documents=0,
        )
        for corpus in CORPORA
    )


def _validate_launch_prefix() -> None:
    configured = marin_prefix().rstrip("/")
    if configured != EXPERIMENT_PREFIX:
        raise ValueError(
            f"MARIN_PREFIX must be exactly {EXPERIMENT_PREFIX!r}, got {configured!r}"
        )


def _hf_files(hf_fs: HfFileSystem, corpus: Corpus) -> list[MirrorFile]:
    entries = hf_fs.ls(corpus.hf_prefix, detail=True)
    files = [
        MirrorFile(
            relative_path=entry["name"].removeprefix(f"{corpus.hf_prefix}/"),
            size=int(entry.get("size") or 0),
            xet_hash=entry.get("xet_hash"),
        )
        for entry in entries
        if entry.get("type") == "file"
    ]
    if not files or not any(file.relative_path.endswith(".parquet") for file in files):
        raise ValueError(f"no parquet files found under hf://{corpus.hf_prefix}")
    if any(
        "/" in file.relative_path or file.relative_path in {"", ".", ".."}
        for file in files
    ):
        raise ValueError(
            f"unexpected nested or invalid file under hf://{corpus.hf_prefix}"
        )
    parquet_files = [file for file in files if file.relative_path.endswith(".parquet")]
    parquet_bytes = sum(file.size for file in parquet_files)
    if (
        len(parquet_files) != corpus.expected_parquet_files
        or parquet_bytes != corpus.expected_parquet_bytes
    ):
        raise ValueError(
            f"{corpus.key} source changed: found {len(parquet_files)} parquet files "
            f"and {parquet_bytes} bytes; expected {corpus.expected_parquet_files} "
            f"files and {corpus.expected_parquet_bytes} bytes"
        )
    return sorted(files, key=lambda file: file.relative_path)


def _destination_path(s3_root: str, relative_path: str) -> str:
    return f"{s3_root.removeprefix('s3://').rstrip('/')}/{relative_path}"


def _copy_one(
    hf_fs: HfFileSystem,
    s3_fs: fsspec.AbstractFileSystem,
    corpus: Corpus,
    source: MirrorFile,
) -> bool:
    source_path = f"{corpus.hf_prefix}/{source.relative_path}"
    destination_path = _destination_path(corpus.s3_prefix, source.relative_path)

    for attempt in range(1, COPY_RETRIES + 1):
        try:
            if s3_fs.exists(destination_path):
                destination_size = int(s3_fs.info(destination_path).get("size") or 0)
                if destination_size == source.size:
                    return False

            with (
                hf_fs.open(source_path, "rb") as source_file,
                s3_fs.open(destination_path, "wb") as destination_file,
            ):
                shutil.copyfileobj(
                    source_file,
                    destination_file,
                    length=COPY_BUFFER_BYTES,
                )
            destination_size = int(s3_fs.info(destination_path).get("size") or 0)
            if destination_size != source.size:
                raise OSError(
                    f"size mismatch for {destination_path}: "
                    f"expected {source.size}, found {destination_size}"
                )
            return True
        except Exception as exc:
            if attempt == COPY_RETRIES:
                raise RuntimeError(
                    f"failed to mirror hf://{source_path} to s3://{destination_path}"
                ) from exc
            time.sleep(min(2**attempt, 30))
    raise AssertionError("copy retry loop did not return or raise")


def _validate_s3_mirror(
    s3_fs: fsspec.AbstractFileSystem,
    corpus: Corpus,
    sources: list[MirrorFile],
) -> None:
    destination_root = corpus.s3_prefix.removeprefix("s3://").rstrip("/")
    destination_entries = {
        path.removeprefix(f"{destination_root}/"): int(info.get("size") or 0)
        for path, info in s3_fs.find(destination_root, detail=True).items()
        if info.get("type") == "file"
    }
    expected = {source.relative_path: source.size for source in sources}
    missing = sorted(expected.keys() - destination_entries.keys())
    wrong_size = sorted(
        name
        for name, size in expected.items()
        if destination_entries.get(name) is not None
        and destination_entries[name] != size
    )
    unexpected = sorted(destination_entries.keys() - expected.keys())
    if missing or wrong_size or unexpected:
        raise ValueError(
            f"invalid {corpus.key} mirror: {len(missing)} missing, "
            f"{len(wrong_size)} wrong-size, {len(unexpected)} unexpected"
        )


def _select_sources(
    sources: list[MirrorFile], parquet_file_limit: int | None
) -> list[MirrorFile]:
    if parquet_file_limit is None:
        return sources
    parquet_files = [
        source
        for source in sources
        if source.relative_path.endswith(".parquet") and source.size > 0
    ]
    selected = sorted(
        parquet_files,
        key=lambda source: (source.size, source.relative_path),
    )[:parquet_file_limit]
    if len(selected) != parquet_file_limit:
        raise ValueError(
            f"requested {parquet_file_limit} parquet files, found {len(selected)}"
        )
    return sorted(selected, key=lambda source: source.relative_path)


def mirror_training_data(
    copy_workers: int,
    *,
    corpora: tuple[Corpus, ...] = CORPORA,
    parquet_file_limit: int | None = None,
    manifest_path: str = MIRROR_MANIFEST,
) -> dict[str, list[MirrorFile]]:
    """Mirror both issue #232 corpora and return their immutable file catalogs."""
    hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or False)
    s3_fs = fsspec.filesystem("s3")
    catalogs: dict[str, list[MirrorFile]] = {}

    for corpus in corpora:
        sources = _select_sources(
            _hf_files(hf_fs, corpus),
            parquet_file_limit,
        )
        catalogs[corpus.key] = sources
        copied = 0
        completed = 0
        with ThreadPoolExecutor(
            max_workers=copy_workers,
            thread_name_prefix=f"mirror-{corpus.key}",
        ) as pool:
            futures = [
                pool.submit(_copy_one, hf_fs, s3_fs, corpus, source)
                for source in sources
            ]
            for future in as_completed(futures):
                copied += int(future.result())
                completed += 1
                if completed % 100 == 0 or completed == len(futures):
                    logger.info(
                        "%s mirror: %d/%d checked, %d copied",
                        corpus.key,
                        completed,
                        len(futures),
                        copied,
                    )
        _validate_s3_mirror(s3_fs, corpus, sources)

    manifest = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "cache_version": CACHE_VERSION,
        "corpora": {
            corpus.key: {
                "hf_prefix": corpus.hf_prefix,
                "s3_prefix": corpus.s3_prefix,
                "files": [asdict(source) for source in catalogs[corpus.key]],
                "total_bytes": sum(source.size for source in catalogs[corpus.key]),
            }
            for corpus in corpora
        },
    }
    manifest_fs, manifest_fs_path = fsspec.core.url_to_fs(manifest_path)
    with manifest_fs.open(manifest_fs_path, "wt") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    logger.info("mirror complete: %s", manifest_path)
    return catalogs


def validate_training_data_mirror(
    *,
    corpora: tuple[Corpus, ...] = CORPORA,
    parquet_file_limit: int | None = None,
) -> dict[str, list[MirrorFile]]:
    """Require a complete S3 mirror before any tokenizer worker is created."""
    hf_fs = HfFileSystem(token=os.environ.get("HF_TOKEN") or False)
    s3_fs = fsspec.filesystem("s3")
    catalogs = {
        corpus.key: _select_sources(
            _hf_files(hf_fs, corpus),
            parquet_file_limit,
        )
        for corpus in corpora
    }
    for corpus in corpora:
        _validate_s3_mirror(s3_fs, corpus, catalogs[corpus.key])
    return catalogs


def _with_observed_document_counts(
    corpora: tuple[Corpus, ...],
    catalogs: dict[str, list[MirrorFile]],
) -> tuple[Corpus, ...]:
    """Read parquet footers so smoke caches must preserve every input row."""
    s3_fs = fsspec.filesystem("s3")
    updated: list[Corpus] = []
    for corpus in corpora:
        documents = 0
        for source in catalogs[corpus.key]:
            if not source.relative_path.endswith(".parquet"):
                continue
            path = _destination_path(corpus.s3_prefix, source.relative_path)
            with s3_fs.open(path, "rb") as parquet_file:
                documents += pq.ParquetFile(parquet_file).metadata.num_rows
        if documents <= 0:
            raise ValueError(f"no documents found in {corpus.key} smoke input")
        logger.info("%s smoke input contains %d documents", corpus.key, documents)
        updated.append(replace(corpus, expected_documents=documents))
    return tuple(updated)


def _validate_ledger(corpus: Corpus, ledger: CacheLedger) -> None:
    total_tokens = ledger.field_counts.get("input_ids", 0)
    if ledger.total_num_rows != corpus.expected_documents or total_tokens <= 0:
        raise ValueError(
            f"invalid {corpus.key} cache ledger: {ledger.total_num_rows} documents "
            f"and {total_tokens} tokens; expected {corpus.expected_documents} "
            "documents and positive tokens"
        )


def _cache_complete(corpus: Corpus) -> bool:
    split_path = prefix_join(corpus.cache_path, "train")
    ledger_path = ShardedCacheLayout.parse(split_path).ledger
    if not StoragePath(ledger_path).exists():
        return False
    try:
        stats = read_tokenized_cache_stats(corpus.cache_path, "train")
    except FileNotFoundError:
        ledger = CacheLedger.load(split_path)
        _validate_ledger(corpus, ledger)
        write_stats_json(split_path, ledger)
        stats = read_tokenized_cache_stats(corpus.cache_path, "train")
    if stats.total_elements != corpus.expected_documents or stats.total_tokens <= 0:
        raise ValueError(
            f"invalid completed {corpus.key} cache stats: {stats}; "
            f"expected {corpus.expected_documents} documents and positive tokens"
        )
    return True


# Pinned Marin's tokenize() constructs a 1GB Zephyr coordinator internally.
# Keep its pipeline helpers but construct the context here with CoreWeave's 6GB floor.
def _file_groups(config: TokenizeConfig) -> list[list[str]]:
    patterns = expand_tokenize_paths(list(config.train_paths))
    files: list[FileEntry] = sorted(
        drop_sidecars(glob_with_sizes(patterns)),
        key=lambda entry: entry.path,
    )
    if not files:
        raise ValueError(f"no mirrored training files matched {patterns}")
    total_bytes = sum(file.size for file in files)
    target_group_bytes = compute_target_group_bytes(
        total_bytes,
        config.num_shards or config.max_workers,
    )
    groups = list(bundle_files_by_size(files, target_group_bytes))
    logger.info(
        "grouped %d files (%.2f GB) into %d tokenizer shards",
        len(files),
        total_bytes / 1e9,
        len(groups),
    )
    return groups


def tokenize_corpus(corpus: Corpus, *, max_workers: int, num_input_files: int) -> None:
    """Tokenize one mirrored corpus with an explicitly sized CW coordinator."""
    if _cache_complete(corpus):
        logger.info("cache already complete, skipping: %s", corpus.cache_path)
        return

    config = TokenizeConfig(
        train_paths=[f"{corpus.s3_prefix}/*.parquet"],
        validation_paths=[],
        cache_path=corpus.cache_path,
        tokenizer=TOKENIZER,
        format=TextLmDatasetFormat(text_key=TEXT_KEY),
        tags=["protein", "contacts-v1", "decontaminated", corpus.key],
        max_workers=max_workers,
        num_shards=num_input_files,
        worker_resources=ResourceConfig(
            cpu=1,
            ram="8g",
            disk="8g",
            preemptible=False,
        ),
    )
    groups = _file_groups(config)
    dataset = (
        Dataset.from_list(groups).flat_map(lambda paths: paths).flat_map(load_file)
    )
    tokenized_dataset, batch_size = tokenize_pipeline(
        dataset,
        data_format=config.format,
        sample_count=config.sample_count,
        sample_parquet_path=parquet_window_hint(groups),
        levanter_batch_size=config.levanter_batch_size,
    )
    context = ZephyrContext(
        resources=config.worker_resources,
        max_workers=min(config.max_workers, len(groups)),
        coordinator_resources=COORDINATOR_RESOURCES,
        chunk_storage_prefix=(
            f"{EXPERIMENT_PREFIX}/tmp/zephyr/"
            f"{corpus.cache_path.removeprefix(EXPERIMENT_PREFIX).strip('/').replace('/', '-')}"
        ),
        name=f"exp232-tokenize-{corpus.key}",
    )
    context.put("tokenizer_name", config.tokenizer)
    context.put("tokenizer_backend", config.tokenizer_backend)
    ledger = build_from_datasets(
        ctx=context,
        dataset=tokenized_dataset,
        output_path=prefix_join(config.cache_path, "train"),
        batch_size=batch_size,
        task_resources=config.map_task_resources,
    )
    _validate_ledger(corpus, ledger)
    stats_path, _ = write_stats_json(
        prefix_join(config.cache_path, "train"),
        ledger,
    )
    logger.info(
        "%s tokenization complete: %d documents; stats at %s",
        corpus.key,
        ledger.total_num_rows,
        stats_path,
    )


@click.command(help=__doc__)
@click.option(
    "--phase",
    type=click.Choice(("all", "mirror", "tokenize"), case_sensitive=False),
    default="all",
    show_default=True,
)
@click.option(
    "--copy-workers", type=click.IntRange(min=1), default=16, show_default=True
)
@click.option(
    "--tokenize-workers",
    type=click.IntRange(min=1),
    default=128,
    show_default=True,
)
@click.option(
    "--smoke-test",
    is_flag=True,
    help=(
        "Mirror the smallest parquet shard from each corpus and tokenize it under "
        "the isolated tmp/tokenization-smoke prefix."
    ),
)
def main(
    phase: str,
    copy_workers: int,
    tokenize_workers: int,
    smoke_test: bool,
) -> None:
    logging.basicConfig(level=logging.INFO)
    _validate_launch_prefix()
    corpora = _smoke_corpora() if smoke_test else CORPORA
    parquet_file_limit = 1 if smoke_test else None
    if phase in {"all", "mirror"}:
        catalogs = mirror_training_data(
            copy_workers,
            corpora=corpora,
            parquet_file_limit=parquet_file_limit,
            manifest_path=SMOKE_MANIFEST if smoke_test else MIRROR_MANIFEST,
        )
    else:
        catalogs = validate_training_data_mirror(
            corpora=corpora,
            parquet_file_limit=parquet_file_limit,
        )
    if phase in {"all", "tokenize"}:
        if smoke_test:
            corpora = _with_observed_document_counts(corpora, catalogs)
        for corpus in corpora:
            parquet_count = sum(
                source.relative_path.endswith(".parquet")
                for source in catalogs[corpus.key]
            )
            tokenize_corpus(
                corpus,
                max_workers=tokenize_workers,
                num_input_files=parquet_count,
            )


if __name__ == "__main__":
    main()
