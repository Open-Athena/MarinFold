# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Census converted exp277 V2 documents and enforce the 8,192-token limit."""

import argparse
import json
import os
from collections.abc import Iterable
from functools import partial

import fsspec
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

from convert_exp277_caches_to_delta_stream import (
    CORPORA,
    MAX_DOCUMENT_TOKENS,
    OUTPUT_ROOT,
    VOCAB_SIZE,
)

CENSUS_ROOT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/exp277_full_epoch_census/2026.09.22.1"
)
TOKEN_BOUNDS = (1024, 2048, 4096, 6144, 7168, 8192)


def list_parquets(documents_root: str, corpus: str) -> list[str]:
    """List one corpus's converted parquet shards."""
    fs, pattern = fsspec.core.url_to_fs(f"{documents_root.rstrip('/')}/{corpus}/shard-*.parquet")
    protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    paths = [f"{protocol}://{path}" for path in sorted(fs.glob(pattern))]
    if not paths:
        raise ValueError(f"no converted shards found for {corpus}")
    return paths


def census_file(path: str, *, corpus: str) -> dict[str, object]:
    """Compute vectorized counts for one converted parquet shard."""
    columns = [
        "source_global_row",
        "seq_len",
        "contacts_undirected",
        "source_token_count",
        "token_count",
        "max_abs_delta",
        "uses_extended_delta",
        "vocab_size",
    ]
    with fsspec.open(path, "rb") as handle:
        table = pq.read_table(handle, columns=columns)
    rows = table.num_rows
    if rows == 0:
        raise ValueError(f"empty converted shard: {path}")
    vocab_min = int(pc.min(table["vocab_size"]).as_py())
    vocab_max = int(pc.max(table["vocab_size"]).as_py())
    if vocab_min != VOCAB_SIZE or vocab_max != VOCAB_SIZE:
        raise ValueError(f"unexpected vocabulary in {path}: {vocab_min}..{vocab_max}")

    token_counts = table["token_count"]
    max_index = int(pc.index(token_counts, pc.max(token_counts)).as_py())
    output: dict[str, object] = {
        "corpus": corpus,
        "path": path,
        "documents": rows,
        "sequence_residues": int(pc.sum(table["seq_len"]).as_py()),
        "contacts_undirected": int(pc.sum(table["contacts_undirected"]).as_py()),
        "source_tokens": int(pc.sum(table["source_token_count"]).as_py()),
        "v2_tokens": int(pc.sum(token_counts).as_py()),
        "extended_delta_documents": int(pc.sum(pc.cast(table["uses_extended_delta"], "int64")).as_py()),
        "over_context_documents": int(pc.sum(pc.cast(pc.greater(token_counts, MAX_DOCUMENT_TOKENS), "int64")).as_py()),
        "at_context_documents": int(pc.sum(pc.cast(pc.equal(token_counts, MAX_DOCUMENT_TOKENS), "int64")).as_py()),
        "max_sequence_length": int(pc.max(table["seq_len"]).as_py()),
        "max_contacts_undirected": int(pc.max(table["contacts_undirected"]).as_py()),
        "max_abs_delta": int(pc.max(table["max_abs_delta"]).as_py()),
        "max_v2_tokens": int(pc.max(token_counts).as_py()),
        "max_source_global_row": int(table["source_global_row"][max_index].as_py()),
    }
    for bound in TOKEN_BOUNDS:
        output[f"documents_le_{bound}"] = int(pc.sum(pc.cast(pc.less_equal(token_counts, bound), "int64")).as_py())
    return output


def process_paths(
    paths: Iterable[dict[str, str]], _shard_info: object, *, corpus: str
) -> Iterable[dict[str, object]]:
    """Census a Zephyr work shard of parquet paths."""
    for row in paths:
        yield census_file(row["path"], corpus=corpus)


def aggregate(corpus: str, expected_documents: int, rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate per-file census records and enforce accounting invariants."""
    additive = {
        "documents",
        "sequence_residues",
        "contacts_undirected",
        "source_tokens",
        "v2_tokens",
        "extended_delta_documents",
        "over_context_documents",
        "at_context_documents",
        *(f"documents_le_{bound}" for bound in TOKEN_BOUNDS),
    }
    maximum = {"max_sequence_length", "max_contacts_undirected", "max_abs_delta", "max_v2_tokens"}
    summary: dict[str, object] = {"corpus": corpus, "parquet_shards": len(rows), "vocab_size": VOCAB_SIZE}
    summary.update({key: sum(int(row[key]) for row in rows) for key in additive})
    summary.update({key: max(int(row[key]) for row in rows) for key in maximum})
    max_row = max(rows, key=lambda row: int(row["max_v2_tokens"]))
    summary["longest_document_path"] = max_row["path"]
    summary["longest_document_source_global_row"] = max_row["max_source_global_row"]
    if int(summary["documents"]) != expected_documents:
        raise ValueError(f"{corpus}: found {summary['documents']} documents, expected {expected_documents}")
    if int(summary["over_context_documents"]) != 0:
        raise ValueError(f"{corpus}: {summary['over_context_documents']} documents exceed {MAX_DOCUMENT_TOKENS}")
    return summary


def run(args: argparse.Namespace) -> None:
    corpus = CORPORA[args.corpus]
    paths = list_parquets(args.documents_root, corpus.name)
    records = Dataset.from_list([{"path": path} for path in paths]).reshard(len(paths)).map_shard(
        partial(process_paths, corpus=corpus.name)
    )
    detail_root = f"{args.census_root.rstrip('/')}/{corpus.name}/shards"
    output = records.write_parquet(f"{detail_root}/shard-{{shard:05d}}-of-{{total:05d}}.parquet")
    context = ZephyrContext(
        max_workers=min(args.max_workers, len(paths)),
        resources=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_memory, disk=args.worker_disk),
        coordinator_resources=ResourceConfig(cpu=1, ram="8GB", disk="16GB"),
        name=f"exp299-census-exp277-v2-{corpus.name}",
        chunk_storage_prefix=f"{args.census_root.rstrip('/')}/{corpus.name}/_zephyr_chunks",
        max_execution_retries=2,
    )
    context.execute(output)

    fs, pattern = fsspec.core.url_to_fs(f"{detail_root}/shard-*.parquet")
    protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    detail_paths = [f"{protocol}://{path}" for path in sorted(fs.glob(pattern))]
    detail_rows: list[dict[str, object]] = []
    for path in detail_paths:
        with fsspec.open(path, "rb") as handle:
            detail_rows.extend(pq.read_table(handle).to_pylist())
    summary = aggregate(corpus.name, corpus.documents, detail_rows)
    summary_path = f"{args.census_root.rstrip('/')}/{corpus.name}/summary.json"
    with fsspec.open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=sorted(CORPORA), required=True)
    parser.add_argument("--documents-root", default=OUTPUT_ROOT)
    parser.add_argument("--census-root", default=CENSUS_ROOT)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EXP299_CENSUS_MAX_WORKERS", "128")))
    parser.add_argument("--worker-cpu", type=float, default=1.0)
    parser.add_argument("--worker-memory", default="6GB")
    parser.add_argument("--worker-disk", default="8GB")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
