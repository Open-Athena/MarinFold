"""Locate production ESM candidates in the pinned Atlas Lance dataset.

The production plan contains tens of millions of hashes. Repeating indexed
lookups for that population is much slower than one parallel scan of the Atlas
``protein_hash`` column. This stage uses a shared sorted uint64 membership array
for the scan, then performs an exact full-hash join before publishing row IDs.
"""

import argparse
import gc
import json
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path
from time import perf_counter

import duckdb
import lance
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from build_esm_plan import require_source_region, sql_string

ATLAS_URI = "s3://esm-protein-atlas/v1/folds/folds_1B.lance"
ATLAS_VERSION = 3
ATLAS_ROWS = 1_095_530_880
ATLAS_STORAGE = {
    "aws_skip_signature": "true",
    "region": "us-west-2",
    "timeout": "120s",
    "connect_timeout": "20s",
}
_HASH_KEYS: np.ndarray | None = None


def open_atlas() -> lance.LanceDataset:
    """Open the exact Atlas snapshot used by exp91."""
    return lance.dataset(ATLAS_URI, version=ATLAS_VERSION, storage_options=ATLAS_STORAGE)


def hash_keys(hashes: list[str]) -> np.ndarray:
    """Return high-64-bit integer keys for lowercase MD5 protein hashes."""
    return np.fromiter(
        (int(value[:16], 16) for value in hashes),
        dtype=np.uint64,
        count=len(hashes),
    )


def prepare_wanted(plan_glob: str, wanted: Path) -> int:
    """Flatten anchor and reservoir hashes with their exact cluster lineage."""
    con = duckdb.connect()
    try:
        con.execute("SET threads=16")
        con.execute(
            f"""
            COPY (
                SELECT
                    shard,
                    cluster_id,
                    anchor_id AS protein_hash,
                    true AS is_anchor,
                    0::INTEGER AS reservoir_rank,
                    anchor_seq_len,
                    anchor_mean_plddt,
                    anchor_ptm,
                    cluster_size,
                    unique_omitted_members,
                    membership_rows_observed,
                    anchor_multiplicity,
                    duplicate_membership_rows
                FROM read_parquet({sql_string(plan_glob)}, hive_partitioning=true)
                UNION ALL
                SELECT
                    shard,
                    cluster_id,
                    candidate.protein_hash,
                    false AS is_anchor,
                    CAST(candidate.reservoir_rank AS INTEGER),
                    anchor_seq_len,
                    anchor_mean_plddt,
                    anchor_ptm,
                    cluster_size,
                    unique_omitted_members,
                    membership_rows_observed,
                    anchor_multiplicity,
                    duplicate_membership_rows
                FROM read_parquet({sql_string(plan_glob)}, hive_partitioning=true),
                UNNEST(candidate_ids) WITH ORDINALITY
                    AS candidate(protein_hash, reservoir_rank)
            ) TO {sql_string(wanted)} (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
        counts = con.execute(
            f"SELECT count(*), count(DISTINCT protein_hash) FROM read_parquet({sql_string(wanted)})"
        ).fetchone()
    finally:
        con.close()
    if counts[0] != counts[1]:
        raise ValueError(
            f"Production plan assigns {counts[0] - counts[1]} hashes more than once"
        )
    return counts[0]


def scan_range(spec: dict) -> tuple[int, int, str]:
    """Scan one Atlas row range and stream matching hashes to parquet."""
    memory = None
    if spec.get("shared_memory"):
        memory = shared_memory.SharedMemory(name=spec["shared_memory"])
        keys = np.ndarray((spec["key_count"],), dtype=np.uint64, buffer=memory.buf)
    else:
        keys = _HASH_KEYS
    if keys is None or not len(keys):
        raise ValueError("Wanted hash-key array is empty")
    start, stop = int(spec["start"]), int(spec["stop"])
    output = Path(spec["output"])
    schema = pa.schema([("row_index", pa.int64()), ("protein_hash", pa.string())])
    writer = pq.ParquetWriter(output, schema, compression="zstd")
    seen = hits = 0
    row_offset = start
    try:
        scanner = open_atlas().scanner(
            columns=["protein_hash"],
            offset=start,
            limit=stop - start,
            batch_size=int(spec["batch_rows"]),
            batch_readahead=4,
            fragment_readahead=2,
        )
        for batch_id, batch in enumerate(scanner.to_batches()):
            hashes = batch.column("protein_hash").to_pylist()
            batch_keys = hash_keys(hashes)
            positions = np.searchsorted(keys, batch_keys)
            valid = positions < len(keys)
            matched = np.zeros(len(hashes), dtype=bool)
            matched[valid] = keys[positions[valid]] == batch_keys[valid]
            indices = np.flatnonzero(matched)
            if len(indices):
                writer.write_table(
                    pa.table(
                        {
                            "row_index": pa.array(
                                row_offset + indices, type=pa.int64()
                            ),
                            "protein_hash": pa.array(
                                [hashes[index] for index in indices], type=pa.string()
                            ),
                        },
                        schema=schema,
                    )
                )
                hits += len(indices)
            seen += len(hashes)
            row_offset += len(hashes)
            if batch_id and batch_id % 20 == 0:
                print(
                    f"scan {spec['tag']}: {seen:,} rows, {hits:,} candidate hits",
                    flush=True,
                )
    finally:
        writer.close()
        if memory is not None:
            memory.close()
    return seen, hits, str(output)


def exact_join(wanted: Path, emits: list[Path], output: Path) -> dict:
    """Drop truncated-key false positives and publish sharded row-index plans."""
    output.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    try:
        con.execute("SET threads=32")
        emits_sql = "[" + ",".join(sql_string(path) for path in emits) + "]"
        con.execute(
            f"""
            COPY (
                SELECT w.*, e.row_index
                FROM read_parquet({sql_string(wanted)}) w
                JOIN read_parquet({emits_sql}) e USING (protein_hash)
            ) TO {sql_string(output)} (
                FORMAT PARQUET,
                PARTITION_BY (shard),
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 100000,
                OVERWRITE_OR_IGNORE
            )
            """
        )
        glob = str(output / "*" / "*.parquet")
        counts = con.execute(
            f"SELECT count(*),count(DISTINCT protein_hash),count(DISTINCT cluster_id) FROM read_parquet({sql_string(glob)}, hive_partitioning=true)"
        ).fetchone()
    finally:
        con.close()
    return {
        "located_rows": counts[0],
        "distinct_hashes": counts[1],
        "clusters": counts[2],
        "parquet_files": len(list(output.rglob("*.parquet"))),
        "parquet_bytes": sum(path.stat().st_size for path in output.rglob("*.parquet")),
    }


def locate(
    plan_glob: str,
    work: Path,
    output: Path,
    *,
    workers: int,
    batch_rows: int,
    atlas_rows: int = ATLAS_ROWS,
) -> dict:
    """Build the wanted set, scan Atlas once, and verify exact completeness."""
    global _HASH_KEYS
    work.mkdir(parents=True, exist_ok=True)
    wanted = work / "wanted.parquet"
    wanted_count = prepare_wanted(plan_glob, wanted)
    hashes = pq.read_table(wanted, columns=["protein_hash"])["protein_hash"].to_pylist()
    _HASH_KEYS = np.unique(hash_keys(hashes))
    del hashes
    gc.collect()
    memory = shared_memory.SharedMemory(create=True, size=_HASH_KEYS.nbytes)
    shared = np.ndarray(_HASH_KEYS.shape, dtype=_HASH_KEYS.dtype, buffer=memory.buf)
    shared[:] = _HASH_KEYS
    step = atlas_rows // workers
    specs = [
        {
            "tag": worker,
            "start": worker * step,
            "stop": atlas_rows if worker == workers - 1 else (worker + 1) * step,
            "batch_rows": batch_rows,
            "output": str(work / f"emits-{worker:03d}.parquet"),
            "shared_memory": memory.name,
            "key_count": len(_HASH_KEYS),
        }
        for worker in range(workers)
    ]
    started = perf_counter()
    try:
        with mp.get_context("spawn").Pool(workers) as pool:
            scan_results = pool.map(scan_range, specs)
    finally:
        memory.close()
        memory.unlink()
    seen = sum(result[0] for result in scan_results)
    if seen != atlas_rows:
        raise ValueError(f"Atlas scan returned {seen:,} rows, expected {atlas_rows:,}")
    emits = [Path(result[2]) for result in scan_results]
    stats = exact_join(wanted, emits, output)
    if stats["located_rows"] != wanted_count or stats["distinct_hashes"] != wanted_count:
        raise ValueError(
            f"Located {stats['located_rows']:,}/{wanted_count:,} exact production hashes"
        )
    for path in emits:
        path.unlink()
    return {
        "wanted_rows": wanted_count,
        "atlas_rows_scanned": seen,
        "truncated_key_hits": sum(result[1] for result in scan_results),
        "scan_workers": workers,
        "batch_rows": batch_rows,
        "atlas_uri": ATLAS_URI,
        "atlas_version": ATLAS_VERSION,
        "elapsed_seconds": perf_counter() - started,
        **stats,
    }


def main() -> None:
    """Run the region-guarded production locator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-glob", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-rows", type=int, default=262144)
    args = parser.parse_args()
    require_source_region()
    stats = locate(
        args.plan_glob,
        args.work,
        args.output,
        workers=args.workers,
        batch_rows=args.batch_rows,
    )
    record = {"status": "complete", **stats}
    (args.output.parent / "locator.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
