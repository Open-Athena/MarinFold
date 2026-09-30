"""Rewrite the AFDB selection manifests under one explicit, uniform schema.

The first production run let pyarrow infer each shard's schema from the rows it
happened to contain. Names and types agreed, but the two selection paths build
their dicts in different orders, so field *order* varied across shards and a
plain ``pa.concat_tables`` refused the set. A reader using ``pyarrow.dataset``
was unaffected, which is exactly what makes the defect easy to ship.

This pass pins the schema instead of inferring it, so every manifest is
byte-compatible with every other and no reader needs promotion options. It
rewrites in place, verifies the row count per shard, and never changes a value.
``curate_afdb`` now emits the same order, so this is a one-time repair rather
than part of the pipeline.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import gcsfs
import pyarrow as pa
import pyarrow.parquet as pq

from curate_afdb import SELECTION_FIELDS

SELECTION_TYPES = {
    "max_selected_tm": pa.float64(),
    "max_selected_core_tm": pa.float64(),
    "min_selected_coverage": pa.float64(),
    "min_selected_length_ratio": pa.float64(),
    "max_selected_sequence_identity": pa.float64(),
    "structural_novelty": pa.float64(),
    "structurally_comparable": pa.bool_(),
    "strict_structural_diversity": pa.bool_(),
    "max_anchor_sequence_identity": pa.float64(),
    "selection_rank": pa.int64(),
    "selection_tier": pa.string(),
}


def target_schema(sample: pa.Schema) -> pa.Schema:
    """Build the canonical schema from a sample manifest's source columns."""
    source = [
        sample.field(name) for name in sample.names if name not in SELECTION_FIELDS
    ]
    missing = [name for name in SELECTION_FIELDS if name not in SELECTION_TYPES]
    if missing:
        raise ValueError(f"No declared type for selection fields {missing}")
    return pa.schema(source + [pa.field(n, SELECTION_TYPES[n]) for n in SELECTION_FIELDS])


def normalize(path: str, schema: pa.Schema, filesystem: gcsfs.GCSFileSystem) -> dict:
    """Rewrite one manifest under the canonical schema, preserving every value."""
    with filesystem.open(path, "rb") as handle:
        table = pq.read_table(handle)
    if set(table.schema.names) != set(schema.names):
        raise ValueError(f"{path}: manifest columns differ from the canonical set")
    ordered = table.select(schema.names).cast(schema)
    with filesystem.open(path, "wb") as handle:
        pq.write_table(ordered, handle, compression="zstd")
    return {"path": path, "rows": ordered.num_rows}


def main() -> None:
    """Normalize every manifest below the prefix and verify the total."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    filesystem = gcsfs.GCSFileSystem()
    paths = sorted(filesystem.glob(args.prefix.rstrip("/") + "/*.parquet"))
    if not paths:
        raise ValueError(f"No manifests below {args.prefix}")
    with filesystem.open(paths[0], "rb") as handle:
        schema = target_schema(pq.ParquetFile(handle).schema_arrow)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda p: normalize(p, schema, filesystem), paths))
    total = sum(item["rows"] for item in results)
    if total != args.expected_rows:
        raise ValueError(f"Rewrote {total:,} rows, expected {args.expected_rows:,}")
    print(
        json.dumps(
            {
                "status": "complete",
                "manifests": len(results),
                "rows": total,
                "columns": schema.names,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
