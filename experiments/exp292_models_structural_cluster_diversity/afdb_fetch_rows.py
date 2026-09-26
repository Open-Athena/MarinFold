"""Fetch and validate AFDB structures for one Zephyr plan shard."""

import hashlib
import socket
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import cache, partial
from time import perf_counter

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq

from structure_audit import STANDARD_AMINO_ACIDS, parse_structure


@cache
def filesystem() -> gcsfs.GCSFileSystem:
    """Reuse one authenticated GCS client throughout a worker process."""
    return gcsfs.GCSFileSystem()


def parse_row(row: dict, content: bytes) -> dict:
    """Validate one complete source object and emit compact structural arrays."""
    started = perf_counter()
    protein = parse_structure(content, int(row["seq_len"]))
    return {
        **row,
        "sequence": protein.sequence,
        "ca_coords": protein.coords.astype("float32").tolist(),
        "per_residue_plddt": protein.plddt.astype("float32").tolist(),
        "noncanonical_residues": sum(
            residue not in STANDARD_AMINO_ACIDS for residue in protein.sequence
        ),
        "source_bytes": len(content),
        "source_sha256": hashlib.sha256(content).hexdigest(),
        "fetch_parse_seconds": perf_counter() - started,
        "hostname": socket.gethostname(),
    }


def fetch_row(row: dict) -> dict:
    """Full-GET and validate one original AFDB v4 mmCIF."""
    started = perf_counter()
    content = filesystem().cat_file(row["gcs_uri"])
    output = parse_row(row, content)
    output["fetch_parse_seconds"] = perf_counter() - started
    return output


def fetch_shard(
    items: Iterable[dict],
    shard_info=None,
    *,
    fetch_concurrency: int = 32,
) -> Iterator[dict]:
    """Overlap full source GETs while preserving fail-loud row handling."""
    with ThreadPoolExecutor(
        max_workers=fetch_concurrency, thread_name_prefix="exp292-afdb-fetch"
    ) as pool:
        yield from pool.map(partial(fetch_row), items)


def fetch_plan_files(
    paths: Iterable[str],
    shard_info=None,
    *,
    columns: tuple[str, ...],
    fetch_concurrency: int = 32,
    row_limit_per_shard: int | None = None,
) -> Iterator[dict]:
    """Full-GET plan parquets, then fetch and validate every contained row.

    Full object reads avoid a gcsfs ranged-read failure on the small plan files.
    Each source structure is still verified independently by ``fetch_row``.
    """
    rows: list[dict] = []
    fs = filesystem()
    for path in paths:
        content = fs.cat_file(path)
        table = pq.read_table(pa.BufferReader(content), columns=list(columns))
        rows.extend(table.to_pylist())
        if row_limit_per_shard is not None and len(rows) >= row_limit_per_shard:
            rows = rows[:row_limit_per_shard]
            break
    yield from fetch_shard(rows, fetch_concurrency=fetch_concurrency)
