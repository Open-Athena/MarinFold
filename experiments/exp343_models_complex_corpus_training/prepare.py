# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the staged complex corpus; adopt exp277's four caches unchanged.

Only the complex corpus is built here, into two caches: the 170 training shards
under `train`, and the held-out shard under `validation` (levanter reads a
validation set from that subdirectory, so the split name is load-bearing).

The tokenizer is the 2845-token contacts-v1 tokenizer the model's embedding
table was sized for, **not** the 2848-token copy published beside the corpus.
Every record is checked against the contacts-v1 boundary contract and against
that vocabulary, so a document using one of the three appended ids would fail
the build rather than silently index past the embedding table.
"""

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

from experiments.exp343_models_complex_corpus_training.config import (
    COMPLEX_TRAIN,
    COMPLEX_VALIDATION,
    CORPORA,
    PREFIX,
    STAGE_MANIFEST,
    TOKENIZER,
    VALIDATION_SHARD,
    Corpus,
)

#: The contacts-v1 vocabulary the 1.5B model was trained against. `<UNK>` is the
#: last id, so any document that tokenizes to it or beyond is out of contract.
VOCAB_SIZE = 2845
UNK_ID = 2844
EXPECTED_TOKEN_IDS = {
    "<pad>": 0,
    "<eos>": 1,
    "<contacts-v1>": 2,
    "<begin_sequence>": 8,
    "<begin_statements>": 9,
    "<end>": 10,
    "<UNK>": UNK_ID,
}
#: Built corpora the experiment owns. The other entries in CORPORA are adopted
#: from exp232/exp277 by path and are only verified.
BUILT = (COMPLEX_TRAIN, COMPLEX_VALIDATION)


def _load_validated_tokenizer():
    tokenizer = load_tokenizer(TOKENIZER, backend="hf")
    vocab = tokenizer.get_vocab()
    if len(tokenizer) != VOCAB_SIZE or any(
        vocab.get(k) != v for k, v in EXPECTED_TOKEN_IDS.items()
    ):
        raise ValueError("Tokenizer differs from the contacts-v1 contract")
    return tokenizer


def _validate_tokenized_record(record: dict) -> dict:
    ids = record["input_ids"]
    if len(ids) < 5 or min(ids) <= 0 or max(ids) >= UNK_ID:
        raise ValueError("Empty, padded, unknown, or out-of-range tokenized record")
    if ids[0] != 2 or 8 not in ids or 9 not in ids or list(ids[-2:]) != [10, 1]:
        raise ValueError("Malformed contacts-v1 record boundaries")
    return record


def read_documents(path: str) -> Iterator[dict[str, str]]:
    """Stream only document text; a whole complex shard must not become a table."""
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
    """Accept only a complete cache with the pinned document cardinality."""
    fs, ledger = fsspec.core.url_to_fs(
        ShardedCacheLayout.parse(f"{corpus.cache}/{corpus.split}").ledger
    )
    if not fs.exists(ledger):
        return False
    stats = read_tokenized_cache_stats(corpus.cache, corpus.split)
    if stats.total_elements != corpus.documents or stats.total_tokens <= 0:
        raise ValueError(f"Invalid cache for {corpus.name}: {stats}")
    if corpus.tokens is not None and stats.total_tokens != corpus.tokens:
        raise ValueError(f"Token count changed for {corpus.name}: {stats}")
    print(
        f"VERIFIED {corpus.name}: {stats.total_elements} documents, "
        f"{stats.total_tokens} tokens",
        flush=True,
    )
    return True


def verify_stage_manifest() -> dict:
    """Require the staged corpus, and that it still splits the way config says."""
    with fsspec.open(STAGE_MANIFEST, "r") as handle:
        manifest = json.load(handle)
    if manifest["validation_shard"] != VALIDATION_SHARD:
        raise ValueError(
            f"Staged manifest holds out shard {manifest['validation_shard']}, "
            f"config holds out {VALIDATION_SHARD}"
        )
    for corpus, key in ((COMPLEX_TRAIN, "train"), (COMPLEX_VALIDATION, "validation")):
        if manifest[f"{key}_documents"] != corpus.documents:
            raise ValueError(
                f"{corpus.name}: manifest has {manifest[f'{key}_documents']} "
                f"documents, config pins {corpus.documents}"
            )
    print(
        f"VERIFIED stage manifest: {manifest['documents']} documents in "
        f"{len(manifest['shards'])} shards",
        flush=True,
    )
    return manifest


def audit_smoke(corpus: Corpus, paths: list[str]) -> None:
    """Independently compare every smoke row against fresh tokenization."""
    processor = preprocessor_for_format(
        TextLmDatasetFormat(text_key="document"), _load_validated_tokenizer()
    )
    cache = TreeCache.load(
        f"{corpus.cache}/{corpus.split}", {"input_ids": np.zeros((1,), dtype=np.int32)}
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
            f"Smoke audit cardinality mismatch: {offset}, {corpus.documents}, "
            f"{len(cache)}"
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
            f"Expected {corpus.shards} staged shards for {corpus.name}, "
            f"found {len(files)}"
        )
    paths = [f"s3://{path}" for path in sorted(files)]
    if smoke:
        smallest = min(files, key=lambda path: files[path]["size"])
        paths = [f"s3://{smallest}"]
    with ThreadPoolExecutor(max_workers=32) as pool:
        documents = sum(pool.map(row_count, paths))
    if smoke:
        corpus = replace(
            corpus,
            documents=documents,
            cache=f"{PREFIX}/smoke-tokenized/{corpus.name}",
            split="train",
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
        f"TOKENIZING {corpus.name}: {len(paths)} shards, {documents} documents "
        f"-> {corpus.cache}/{corpus.split}",
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
    # A complex shard is 20,000 documents averaging 3,270 tokens, with a tail at
    # the 8192 cap -- an order of magnitude smaller per shard than exp277's AFDB
    # tail, which is what forced 32 GB there.
    context = ZephyrContext(
        resources=ResourceConfig(cpu=1, ram="12g", disk="16g"),
        max_workers=min(180, len(paths)),
        coordinator_resources=ResourceConfig(cpu=1, ram="6g", disk="16g"),
        chunk_storage_prefix=(
            f"{PREFIX}/tmp/zephyr/{'smoke' if smoke else 'production'}/{corpus.name}"
        ),
        name=f"exp343-tokenize-{corpus.name}",
        max_execution_retries=1,
    )
    context.put("tokenizer_name", TOKENIZER)
    context.put("tokenizer_backend", "hf")
    ledger = build_from_datasets(
        ctx=context,
        dataset=tokenized,
        output_path=f"{corpus.cache}/{corpus.split}",
        batch_size=batch_size,
        task_resources=None,
    )
    if ledger.total_num_rows != corpus.documents:
        raise ValueError(
            f"Tokenization changed row count: {ledger.total_num_rows} != "
            f"{corpus.documents}"
        )
    write_stats_json(f"{corpus.cache}/{corpus.split}", ledger)
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
    verify_stage_manifest()
    for corpus in BUILT:
        prepare(corpus, smoke)
    if smoke:
        return
    for corpus in CORPORA:
        if corpus not in BUILT and not verify_cache(corpus):
            raise ValueError(f"Missing adopted cache: {corpus.cache}")


if __name__ == "__main__":
    main()
