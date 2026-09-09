"""Tokenize the existing regional MPNN documents without copying the corpus."""

import json
import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import click
import fsspec
import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.data.text.formats import TextLmDatasetFormat, preprocessor_for_format
from levanter.store.cache import ShardedCacheLayout, TreeCache
from levanter.tokenizers import load_tokenizer
from marin.processing.tokenize._core import parquet_window_hint, tokenize_pipeline
from marin.processing.tokenize.cache_stats import read_tokenized_cache_stats
from marin.processing.tokenize.store_builder import (
    build_from_datasets,
    write_stats_json,
)
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.exp277_models_single_mpnn_pilot.config import (
    CORPORA,
    PREFIX,
    TOKENIZER,
    Corpus,
)


EXPECTED_TOKEN_IDS = {
    "<pad>": 0,
    "<eos>": 1,
    "<contacts-v1>": 2,
    "<begin_sequence>": 8,
    "<begin_statements>": 9,
    "<end>": 10,
    "<UNK>": 2844,
}


def _load_validated_tokenizer():
    tokenizer = load_tokenizer(TOKENIZER, backend="hf")
    vocab = tokenizer.get_vocab()
    if len(tokenizer) != 2845 or any(
        vocab.get(k) != v for k, v in EXPECTED_TOKEN_IDS.items()
    ):
        raise ValueError("Tokenizer differs from the contacts-v1 contract")
    return tokenizer


def _validate_tokenized_record(record: dict) -> dict:
    ids = record["input_ids"]
    if len(ids) < 5 or min(ids) <= 0 or max(ids) >= 2844:
        raise ValueError("Empty, padded, unknown, or out-of-range tokenized record")
    if ids[0] != 2 or 8 not in ids or 9 not in ids or list(ids[-2:]) != [10, 1]:
        raise ValueError("Malformed contacts-v1 record boundaries")
    return record


def read_documents(path: str) -> Iterator[dict[str, str]]:
    """Stream only document text; long AFDB row groups must not become a table."""
    with fsspec.open(path, "rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches(
            batch_size=128, columns=["document"]
        ):
            yield from batch.to_pylist()


def row_count(path: str) -> int:
    """Read only the footer, through the explicitly configured S3 filesystem."""
    with fsspec.open(path, "rb") as handle:
        return pq.ParquetFile(handle).metadata.num_rows


def verify_cache(corpus: Corpus) -> bool:
    """Accept only complete caches with the pinned document cardinality."""
    fs, ledger = fsspec.core.url_to_fs(
        ShardedCacheLayout.parse(f"{corpus.cache}/train").ledger
    )
    if not fs.exists(ledger):
        return False
    stats = read_tokenized_cache_stats(corpus.cache, "train")
    if stats.total_elements != corpus.documents or stats.total_tokens <= 0:
        raise ValueError(f"Invalid cache for {corpus.name}: {stats}")
    if corpus.tokens is not None and stats.total_tokens != corpus.tokens:
        raise ValueError(f"Token count changed for {corpus.name}: {stats}")
    print(
        f"VERIFIED {corpus.name}: {stats.total_elements} documents, {stats.total_tokens} tokens",
        flush=True,
    )
    return True


def audit_smoke(corpus: Corpus, paths: list[str]) -> None:
    """Independently compare every smoke row against fresh tokenization."""
    processor = preprocessor_for_format(
        TextLmDatasetFormat(text_key="document"), _load_validated_tokenizer()
    )
    cache = TreeCache.load(
        f"{corpus.cache}/train", {"input_ids": np.zeros((1,), dtype=np.int32)}
    )
    offset = 0
    for path in paths:
        with fsspec.open(path, "rb") as handle:
            for batch in pq.ParquetFile(handle).iter_batches(
                batch_size=64, columns=["document"]
            ):
                fresh = processor(batch.to_pylist())
                cached = cache.get_batch_sync(slice(offset, offset + len(fresh)))
                for expected, actual in zip(fresh, cached, strict=True):
                    if not np.array_equal(expected["input_ids"], actual["input_ids"]):
                        raise ValueError(
                            f"Fresh tokenization mismatch at {corpus.name} row {offset}"
                        )
                    offset += 1
    if offset != corpus.documents or len(cache) != offset:
        raise ValueError(
            f"Smoke audit cardinality mismatch: {offset}, {corpus.documents}, {len(cache)}"
        )
    print(
        f"AUDITED {corpus.name}: all {offset} rows exactly match fresh tokenization",
        flush=True,
    )


def prepare(corpus: Corpus, smoke: bool) -> None:
    """Audit source cardinality and build a resumable, validated token cache."""
    if not smoke and verify_cache(corpus):
        return
    fs, pattern = fsspec.core.url_to_fs(corpus.source)
    files = fs.glob(pattern, detail=True)
    if len(files) != corpus.shards:
        raise ValueError(
            f"Expected {corpus.shards} source shards for {corpus.name}, found {len(files)}"
        )
    paths = [f"s3://{path}" for path in sorted(files)]
    if smoke:
        smallest = min(files, key=lambda path: files[path]["size"])
        paths = [f"s3://{smallest}"]
    with ThreadPoolExecutor(max_workers=32) as pool:
        documents = sum(pool.map(row_count, paths))
    if smoke:
        corpus = replace(
            corpus, documents=documents, cache=f"{PREFIX}/smoke-tokenized/{corpus.name}"
        )
    elif documents != corpus.documents:
        raise ValueError(
            f"{corpus.name}: source has {documents} rows, expected {corpus.documents}"
        )
    if verify_cache(corpus):
        if smoke:
            audit_smoke(corpus, paths)
        return
    print(
        f"TOKENIZING {corpus.name}: {len(paths)} shards, {documents} documents -> {corpus.cache}",
        flush=True,
    )
    dataset = Dataset.from_list(paths).flat_map(read_documents)
    tokenized, batch_size = tokenize_pipeline(
        dataset,
        data_format=TextLmDatasetFormat(text_key="document"),
        sample_count=None,
        sample_parquet_path=parquet_window_hint([[path] for path in paths]),
        levanter_batch_size=1024,
    )
    tokenized = tokenized.map(_validate_tokenized_record)
    # ESM shards have bounded sizes and measured peak memory below 2 GB.
    # Reserve the larger memory envelope for AFDB's long-document tail.
    esm = corpus.name == "mpnn-esm"
    context = ZephyrContext(
        resources=ResourceConfig(cpu=1, ram="8g" if esm else "32g", disk="16g"),
        max_workers=min(512 if esm else 128, len(paths)),
        coordinator_resources=ResourceConfig(cpu=1, ram="6g", disk="16g"),
        chunk_storage_prefix=f"{PREFIX}/tmp/zephyr/{'smoke' if smoke else 'production'}/{corpus.name}",
        name=f"exp277-tokenize-{corpus.name}",
        max_execution_retries=1,
    )
    context.put("tokenizer_name", TOKENIZER)
    context.put("tokenizer_backend", "hf")
    ledger = build_from_datasets(
        ctx=context,
        dataset=tokenized,
        output_path=f"{corpus.cache}/train",
        batch_size=batch_size,
        task_resources=None,
    )
    if ledger.total_num_rows != corpus.documents:
        raise ValueError(
            f"Tokenization changed row count: {ledger.total_num_rows} != {corpus.documents}"
        )
    write_stats_json(f"{corpus.cache}/train", ledger)
    verify_cache(corpus)
    if smoke:
        audit_smoke(corpus, paths)
    with fsspec.open(f"{corpus.cache}/source-manifest.json", "w") as handle:
        json.dump(
            {"paths": paths, "documents": documents, "tokenizer": TOKENIZER}, handle
        )


@click.command()
@click.option("--smoke", is_flag=True)
def main(smoke: bool) -> None:
    logging.basicConfig(level=logging.INFO)
    _load_validated_tokenizer()
    for corpus in CORPORA:
        if corpus.source:
            prepare(corpus, smoke)
        elif not verify_cache(corpus):
            raise ValueError(f"Missing native cache: {corpus.cache}")


if __name__ == "__main__":
    main()
