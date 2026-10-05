"""Audit every shard of the exp343 training pool in its existing CoreWeave region."""

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import pyarrow.parquet as pq

from common import MAX_LENGTH, MODELS, ROOT

BASE = "s3://marin-us-east-02a/MarinFold"
SOURCES = {
    "native-afdb": (f"{BASE}/exp232_sweep_cv1_decontam/data/afdb", 3_963_003),
    "native-esm": (f"{BASE}/exp232_sweep_cv1_decontam/data/esm", 65_553_178),
    "mpnn-afdb": (f"{BASE}/exp266/documents", 31_702_680),
    "mpnn-esm": (f"{BASE}/exp266/esm_documents", 130_872_044),
    "complex": (
        f"{BASE}/exp343_models_complex_corpus_training/documents/train",
        3_400_000,
    ),
}
DEFAULT_OUT = ROOT + "/data/corpus235m-v1"


def canonical_bytes(value: dict) -> bytes:
    """Encode a manifest deterministically for identity and resume validation."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def audit_shard(uri: str) -> dict:
    """Read metadata only and require the fields used by the streaming trainer."""
    fs, path = fsspec.core.url_to_fs(uri)
    info = fs.info(path)
    with fs.open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        columns = parquet.schema_arrow.names
        lineage = "cluster_key" if "document_id" in columns else "seq_cluster_id"
        identifier = "document_id" if "document_id" in columns else "entry_id"
        required = ["document", identifier, lineage]
        if not set(required).issubset(columns):
            raise ValueError(f"Missing training or lineage columns: {uri}")
        if not info.get("ETag"):
            raise ValueError(f"Missing immutable object identity: {uri}")
        return {
            "uri": uri,
            "rows": parquet.metadata.num_rows,
            "size": info["size"],
            "etag": info["ETag"],
            "columns": required,
            "row_groups": [
                parquet.metadata.row_group(i).num_rows
                for i in range(parquet.metadata.num_row_groups)
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    fs, output = fsspec.core.url_to_fs(args.out)
    if fs.exists(output + "/manifest.json"):
        raise ValueError("A published catalog is immutable; select a new output prefix")
    catalog = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for name, (prefix, expected) in SOURCES.items():
            source_fs, path = fsspec.core.url_to_fs(prefix)
            files = sorted(source_fs.glob(path + "/*.parquet"))
            shards = list(pool.map(audit_shard, ["s3://" + p for p in files]))
            rows = sum(s["rows"] for s in shards)
            if rows != expected:
                raise ValueError(
                    f"{name}: expected {expected} source rows, found {rows}"
                )
            catalog[name] = {"source": prefix, "rows": rows, "shards": shards}
            print(
                json.dumps(
                    {
                        "source": name,
                        "rows": rows,
                        "shards": len(shards),
                        "bytes": sum(s["size"] for s in shards),
                    }
                ),
                flush=True,
            )
    encoded = canonical_bytes(catalog)
    manifest = {
        "kind": "source_documents",
        "source_experiment": 343,
        "created_utc": datetime.now(UTC).isoformat(),
        "max_length": MAX_LENGTH,
        "tokenizer": list(MODELS["0.8B"]),
        "catalog_sha256": hashlib.sha256(encoded).hexdigest(),
        "source_documents": sum(s["rows"] for s in catalog.values()),
        "sources": {
            name: {
                "source": s["source"],
                "rows": s["rows"],
                "shards": len(s["shards"]),
                "bytes": sum(p["size"] for p in s["shards"]),
            }
            for name, s in catalog.items()
        },
        "validation_prefix": ROOT + "/data/v1/validation",
        "filter": "Reserve hash(lineage)%100==0; exclude empty contacts and either format >16384 tokens",
        "shuffle": "Each rank reads disjoint shuffled shards; draw sources by remaining raw row count; shuffle 128-row batches",
        "source_truncation": "Preserve published documents including upstream complex statement-boundary truncation",
        "code_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["common.py", "corpus_catalog.py", "corpus_stream.py"]
        },
    }
    with fs.open(output + "/catalog.json", "wb") as handle:
        handle.write(encoded)
    with fs.open(output + "/manifest.json", "wb") as handle:
        handle.write(canonical_bytes(manifest))
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
