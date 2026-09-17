# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Build fixed-length Levanter prebuilt caches for exp299 delta-stream training."""

import argparse
import json
import os
from collections.abc import Iterable
from functools import partial

import fsspec
import numpy as np
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from levanter.store.cache import CACHE_LAYOUT_SHARDED, CacheLedger, CacheMetadata, SerialCacheWriter
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset

DEFAULT_INPUT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/documents/2026.09.16.1/shard-*-of-03338.parquet"
)
DEFAULT_CACHE_ROOT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/packed_cache/2026.09.16.1"
)
DEFAULT_META_ROOT = (
    "s3://marin-us-east-02a/protein-structure/MarinFold/"
    "exp299_contacts_delta_stream_v2_sequence_prefix/packed_cache_meta/2026.09.16.1"
)
VOCAB_SIZE = 2080
SEQ_LEN = 8192
PAD_TOKEN_ID = 0


def _list_inputs(pattern: str, *, max_shards: int | None = None) -> list[str]:
    fs, glob = fsspec.core.url_to_fs(pattern)
    paths = sorted(fs.glob(glob))
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise ValueError(f"no input shards matched {pattern}")
    protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    return [f"{protocol}://{path}" for path in paths]


def _token_arrays_from_parquet(path: str, *, max_rows: int | None = None) -> Iterable[np.ndarray]:
    emitted = 0
    with fsspec.open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(batch_size=2048, columns=["token_ids", "vocab_size", "token_count"]):
            for row in batch.to_pylist():
                if int(row["vocab_size"]) != VOCAB_SIZE:
                    raise ValueError(f"unexpected vocab_size={row['vocab_size']} in {path}")
                token_ids = np.asarray(row["token_ids"], dtype=np.int32)
                if token_ids.ndim != 1 or token_ids.size != int(row["token_count"]):
                    raise ValueError(f"bad token_ids length in {path}")
                if token_ids.size == 0 or int(token_ids.min()) < 0 or int(token_ids.max()) >= VOCAB_SIZE:
                    raise ValueError(f"token ids out of range in {path}")
                yield token_ids
                emitted += 1
                if max_rows is not None and emitted >= max_rows:
                    return


def _flush_example(buffer: list[int], valid: int) -> dict[str, np.ndarray]:
    if len(buffer) != SEQ_LEN:
        raise ValueError(f"expected full buffer length {SEQ_LEN}, got {len(buffer)}")
    loss_weights = np.zeros((SEQ_LEN,), dtype=np.float32)
    loss_weights[:valid] = 1.0
    return {"input_ids": np.asarray(buffer, dtype=np.int32), "loss_weights": loss_weights}


def _packed_records(path: str, *, max_rows: int | None = None) -> Iterable[dict[str, np.ndarray]]:
    buffer: list[int] = []
    valid = 0
    for token_ids in _token_arrays_from_parquet(path, max_rows=max_rows):
        offset = 0
        while offset < token_ids.size:
            space = SEQ_LEN - len(buffer)
            take = min(space, token_ids.size - offset)
            buffer.extend(int(x) for x in token_ids[offset : offset + take])
            valid += take
            offset += take
            if len(buffer) == SEQ_LEN:
                yield _flush_example(buffer, valid)
                buffer = []
                valid = 0
    if buffer:
        valid = len(buffer)
        buffer.extend([PAD_TOKEN_ID] * (SEQ_LEN - len(buffer)))
        yield _flush_example(buffer, valid)


def write_cache_shard(row: dict, *, cache_split: str, max_rows_per_input_shard: int | None = None) -> list[dict]:
    source = row["path"]
    shard_index = int(row["shard_index"])
    shard_name = f"shards/shard-{shard_index:05d}"
    shard_path = f"{cache_split}/{shard_name}"
    rows = 0
    valid_tokens = 0
    exemplar = {
        "input_ids": np.zeros((SEQ_LEN,), dtype=np.int32),
        "loss_weights": np.zeros((SEQ_LEN,), dtype=np.float32),
    }
    with SerialCacheWriter(shard_path, exemplar, shard_name=shard_name, mode="w") as writer:
        batch = []
        for record in _packed_records(source, max_rows=max_rows_per_input_shard):
            rows += 1
            valid_tokens += int(record["loss_weights"].sum())
            batch.append(record)
            if len(batch) >= 256:
                writer.write_batch(batch)
                batch = []
        if batch:
            writer.write_batch(batch)
    return [
        {
            "source": source,
            "shard_index": shard_index,
            "shard_name": shard_name,
            "rows": rows,
            "valid_tokens": valid_tokens,
            "stored_tokens": rows * SEQ_LEN,
        }
    ]


def _read_metadata_rows(meta_path: str) -> list[dict]:
    fs, glob = fsspec.core.url_to_fs(f"{meta_path.rstrip('/')}/*.parquet")
    paths = sorted(fs.glob(glob))
    if not paths:
        raise ValueError(f"no metadata shards found at {meta_path}")
    protocol = fs.protocol[0] if isinstance(fs.protocol, tuple) else fs.protocol
    rows: list[dict] = []
    for path in paths:
        with fsspec.open(f"{protocol}://{path}", "rb") as handle:
            rows.extend(pq.read_table(handle).to_pylist())
    rows.sort(key=lambda r: int(r["shard_index"]))
    return rows


def finalize_ledger(cache_split: str, meta_path: str) -> None:
    rows = _read_metadata_rows(meta_path)
    shard_rows = {str(row["shard_name"]): int(row["rows"]) for row in rows}
    field_counts_by_shard = {
        str(row["shard_name"]): {
            "input_ids": int(row["stored_tokens"]),
            "loss_weights": int(row["stored_tokens"]),
        }
        for row in rows
    }
    stored_tokens = sum(int(row["stored_tokens"]) for row in rows)
    valid_tokens = sum(int(row["valid_tokens"]) for row in rows)
    ledger = CacheLedger(
        total_num_rows=sum(shard_rows.values()),
        shard_rows=shard_rows,
        is_finished=True,
        finished_shards=list(shard_rows),
        field_counts={"input_ids": stored_tokens, "loss_weights": stored_tokens},
        field_counts_by_shard=field_counts_by_shard,
        layout=CACHE_LAYOUT_SHARDED,
        metadata=CacheMetadata.empty(),
    )
    ledger._serialize_and_commit(cache_split)
    with fsspec.open(f"{cache_split}/stats.json", "w") as handle:
        json.dump(
            {
                "total_examples": ledger.total_num_rows,
                "stored_tokens": stored_tokens,
                "valid_tokens": valid_tokens,
                "seq_len": SEQ_LEN,
                "vocab_size": VOCAB_SIZE,
            },
            handle,
            indent=2,
        )
    print(
        f"[exp299] finalized {cache_split}: {ledger.total_num_rows} examples, "
        f"{valid_tokens} valid tokens, {len(rows)} shards",
        flush=True,
    )


def process_cache_rows(
    shard: Iterable[dict],
    _shard_info: object,
    *,
    cache_split: str,
    max_rows_per_input_shard: int | None,
) -> list[dict]:
    return [
        output
        for row in shard
        for output in write_cache_shard(
            row,
            cache_split=cache_split,
            max_rows_per_input_shard=max_rows_per_input_shard,
        )
    ]


def build_split(
    *,
    paths: list[str],
    cache_split: str,
    meta_path: str,
    max_workers: int,
    max_rows_per_input_shard: int | None,
    worker_cpu: float,
    worker_memory: str,
    worker_disk: str,
    preemptible: bool,
    region: str,
) -> None:
    records = [{"path": path, "shard_index": i} for i, path in enumerate(paths)]
    dataset = Dataset.from_list(records).map_shard(
        partial(process_cache_rows, cache_split=cache_split, max_rows_per_input_shard=max_rows_per_input_shard)
    )
    output = dataset.write_parquet(f"{meta_path.rstrip('/')}/shard-{{shard:05d}}-of-{{total:05d}}.parquet")
    ctx = ZephyrContext(
        max_workers=max_workers,
        resources=ResourceConfig(
            cpu=worker_cpu,
            ram=worker_memory,
            disk=worker_disk,
            regions=[region],
            preemptible=preemptible,
        ),
        coordinator_resources=ResourceConfig(cpu=1, ram="6GB", disk="16GB", preemptible=preemptible),
        name="exp299-delta-stream-packed-cache",
        chunk_storage_prefix=f"{meta_path.rstrip('/')}/_zephyr_chunks",
        max_execution_retries=1,
    )
    ctx.execute(output)
    finalize_ledger(cache_split, meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--meta-root", default=DEFAULT_META_ROOT)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--validation-shards", type=int, default=1)
    parser.add_argument("--max-rows-per-input-shard", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=int(os.environ.get("EXP299_PACKED_CACHE_MAX_WORKERS", "128")))
    parser.add_argument("--region", default="us-central1", help="Region for Zephyr workers and source-data locality.")
    parser.add_argument("--worker-cpu", type=float, default=1.0)
    parser.add_argument("--worker-memory", default="10GB")
    parser.add_argument("--worker-disk", default="32GB")
    parser.add_argument("--preemptible", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    input_paths = _list_inputs(args.input, max_shards=args.max_shards)
    if args.validation_shards <= 0:
        raise ValueError("--validation-shards must be positive")
    if len(input_paths) <= args.validation_shards:
        raise ValueError(
            f"need more input shards than validation shards: {len(input_paths)} <= {args.validation_shards}"
        )
    val_paths = input_paths[: args.validation_shards]
    train_paths = input_paths[args.validation_shards :]
    print(f"[exp299] train shards: {len(train_paths)} -> {args.cache_root}/train", flush=True)
    build_split(
        paths=train_paths,
        cache_split=f"{args.cache_root}/train",
        meta_path=f"{args.meta_root}/train",
        max_workers=args.max_workers,
        max_rows_per_input_shard=args.max_rows_per_input_shard,
        worker_cpu=args.worker_cpu,
        worker_memory=args.worker_memory,
        worker_disk=args.worker_disk,
        preemptible=args.preemptible,
        region=args.region,
    )
    print(f"[exp299] validation shards: {len(val_paths)} -> {args.cache_root}/validation", flush=True)
    build_split(
        paths=val_paths,
        cache_split=f"{args.cache_root}/validation",
        meta_path=f"{args.meta_root}/validation",
        max_workers=min(args.max_workers, len(val_paths)),
        max_rows_per_input_shard=args.max_rows_per_input_shard,
        worker_cpu=args.worker_cpu,
        worker_memory=args.worker_memory,
        worker_disk=args.worker_disk,
        preemptible=args.preemptible,
        region=args.region,
    )


if __name__ == "__main__":
    main()
