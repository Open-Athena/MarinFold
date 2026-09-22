# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build resumable variable-length Levanter caches from converted V2 parquets."""

import argparse
import json
import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import fsspec
import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.store.cache import ShardedCacheLayout
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.processing.tokenize.store_builder import (
    build_from_datasets,
    write_stats_json,
)
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from convert_exp277_caches_to_delta_stream import CORPORA, OUTPUT_ROOT, VOCAB_SIZE

CACHE_ROOT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/"
    "exp277_full_epoch_tokenized_cache/2026.09.22.1"
)


def row_count(path: str) -> int:
    """Read one parquet footer without materializing token arrays."""
    with fsspec.open(path, "rb") as handle:
        return pq.ParquetFile(handle).metadata.num_rows


def read_token_ids(path: str, *, max_rows: int | None) -> Iterator[dict[str, np.ndarray]]:
    """Stream and validate token arrays from one converted document shard."""
    emitted = 0
    with fsspec.open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(batch_size=512, columns=["token_ids", "token_count", "vocab_size"]):
            for row in batch.to_pylist():
                token_ids = np.asarray(row["token_ids"], dtype=np.int32)
                if token_ids.ndim != 1 or token_ids.size != int(row["token_count"]):
                    raise ValueError(f"bad token length in {path}")
                if int(row["vocab_size"]) != VOCAB_SIZE:
                    raise ValueError(f"unexpected vocab size in {path}: {row['vocab_size']}")
                if token_ids.size == 0 or int(token_ids.min()) < 0 or int(token_ids.max()) >= VOCAB_SIZE:
                    raise ValueError(f"out-of-range token in {path}")
                yield {"input_ids": token_ids}
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    return


def verify_cache(cache_root: str, *, expected_documents: int | None) -> tuple[int, int] | None:
    """Return completed cache counts, or ``None`` when no ledger exists."""
    split = f"{cache_root.rstrip('/')}/train"
    fs, ledger = fsspec.core.url_to_fs(ShardedCacheLayout.parse(split).ledger)
    if not fs.exists(ledger):
        return None
    stats = read_tokenized_cache_stats(cache_root, "train")
    if expected_documents is not None and stats.total_elements != expected_documents:
        raise ValueError(f"cache has {stats.total_elements} documents, expected {expected_documents}")
    if stats.total_tokens <= 0:
        raise ValueError("completed cache has no tokens")
    print(f"VERIFIED {cache_root}: {stats.total_elements} documents, {stats.total_tokens} tokens", flush=True)
    return stats.total_elements, stats.total_tokens


def build(args: argparse.Namespace) -> None:
    corpus = CORPORA[args.corpus]
    input_pattern = f"{args.documents_root.rstrip('/')}/{corpus.name}/shard-*.parquet"
    fs, pattern = fsspec.core.url_to_fs(input_pattern)
    protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    paths = [f"{protocol}://{path}" for path in sorted(fs.glob(pattern))]
    if args.max_input_shards is not None:
        paths = paths[: args.max_input_shards]
    if not paths:
        raise ValueError(f"no input shards matched {input_pattern}")
    with ThreadPoolExecutor(max_workers=min(32, len(paths))) as pool:
        source_counts = list(pool.map(row_count, paths))
    source_documents = sum(source_counts)
    expected_documents = (
        sum(min(count, args.max_rows_per_input_shard) for count in source_counts)
        if args.max_rows_per_input_shard is not None
        else source_documents
    )
    if args.max_input_shards is None and args.max_rows_per_input_shard is None and source_documents != corpus.documents:
        raise ValueError(f"{corpus.name}: found {source_documents} documents, expected {corpus.documents}")

    cache_root = f"{args.cache_root.rstrip('/')}/{corpus.name}"
    completed = verify_cache(cache_root, expected_documents=expected_documents)
    if completed is not None:
        return

    dataset = Dataset.from_list(paths).reshard(len(paths)).flat_map(
        lambda path: read_token_ids(path, max_rows=args.max_rows_per_input_shard)
    )
    context = ZephyrContext(
        resources=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_memory, disk=args.worker_disk),
        max_workers=min(args.max_workers, len(paths)),
        coordinator_resources=ResourceConfig(cpu=1, ram="8GB", disk="16GB"),
        chunk_storage_prefix=f"{cache_root}/_zephyr_chunks",
        name=f"exp299-cache-exp277-v2-{corpus.name}",
        max_execution_retries=2,
    )
    ledger = build_from_datasets(
        ctx=context,
        dataset=dataset,
        output_path=f"{cache_root}/train",
        batch_size=args.write_batch_size,
        task_resources=None,
    )
    if ledger.total_num_rows != expected_documents:
        raise ValueError(f"cache wrote {ledger.total_num_rows} documents, expected {expected_documents}")
    write_stats_json(f"{cache_root}/train", ledger)
    documents, tokens = verify_cache(cache_root, expected_documents=expected_documents) or (0, 0)
    with fsspec.open(f"{cache_root}/source-manifest.json", "w") as handle:
        json.dump(
            {
                "corpus": corpus.name,
                "input_paths": paths,
                "documents": documents,
                "tokens": tokens,
                "vocab_size": VOCAB_SIZE,
                "max_rows_per_input_shard": args.max_rows_per_input_shard,
            },
            handle,
            indent=2,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    parser.add_argument("--documents-root", default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", default=CACHE_ROOT)
    parser.add_argument("--max-input-shards", type=int)
    parser.add_argument("--max-rows-per-input-shard", type=int)
    parser.add_argument("--write-batch-size", type=int, default=512)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EXP299_CACHE_MAX_WORKERS", "256")))
    parser.add_argument("--worker-cpu", type=float, default=1.0)
    parser.add_argument("--worker-memory", default="8GB")
    parser.add_argument("--worker-disk", default="16GB")
    args = parser.parse_args()
    if args.max_input_shards is not None and args.max_input_shards <= 0:
        parser.error("--max-input-shards must be positive")
    if args.max_rows_per_input_shard is not None and args.max_rows_per_input_shard <= 0:
        parser.error("--max-rows-per-input-shard must be positive")
    build(args)


if __name__ == "__main__":
    main()
