"""Check co-located source metadata without downloading checkpoint weights."""

import argparse
import itertools

import fsspec
import pyarrow.parquet as pq

from common import BASE_MODEL, ROOT, identity, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=f"{ROOT}/inputs")
    args = parser.parse_args()
    config = read_json(f"{BASE_MODEL}/config.json")
    if (config["model_type"], config["hidden_size"], config["num_hidden_layers"]) != ("qwen3", 2048, 24):
        raise ValueError("unexpected base checkpoint architecture")
    fs, model = fsspec.core.url_to_fs(BASE_MODEL)
    model_files = fs.ls(model, detail=True)
    if not any(p["name"].endswith("tokenizer.json") for p in model_files):
        raise ValueError("base checkpoint is missing tokenizer")
    source_root = "s3://marin-us-east-02a/MarinFold/exp232_sweep_cv1_decontam/data"
    sources = []
    by_source = []
    schemas = {}
    for name in ("afdb", "esm"):
        source_fs, prefix = fsspec.core.url_to_fs(f"{source_root}/{name}")
        metadata = source_fs.glob(f"{prefix}/**/*.parquet", detail=True)
        paths = sorted(metadata)
        if not paths:
            raise ValueError(f"no staged {name} decontaminated parquet inputs")
        with source_fs.open(paths[0], "rb") as handle:
            parquet = pq.ParquetFile(handle)
            columns = parquet.schema_arrow.names
            if "document" not in columns or "entry_id" not in columns:
                raise ValueError(f"unexpected {name} schema: {columns}")
            schemas[name] = {"columns": columns, "first_file_rows": parquet.metadata.num_rows,
                             "shards": len(paths)}
        by_source.append([{"name": name, "uri": source_fs.unstrip_protocol(p),
                           "id_column": "entry_id", "document_column": "document",
                           "size_bytes": metadata[p]["size"], "etag": metadata[p].get("ETag")} for p in paths])
    for group in itertools.zip_longest(*by_source):
        sources.extend(s for s in group if s is not None)
    manifest = {"source_revision": "exp232 decontaminated staged corpora; source manifest pinned below",
                "decontamination_reference": "exp225: legacy-554 plus all 1940 FoldBench chains; identity>=0.30, coverage>=0.50 of shorter",
                "sources": sources}
    # The full ordered URI list is hashed, rather than relying on a moving glob.
    manifest["source_revision"] = identity(sources)
    write_json(f"{args.output}/sources.json", manifest)
    write_json(f"{args.output}/preflight.json", {"base_model": BASE_MODEL, "config": config,
               "model_files": [{k: p[k] for k in ("name", "size")} for p in model_files], "schemas": schemas})
    print(f"PASS: model metadata and both source schemas; manifest at {args.output}/sources.json", flush=True)


if __name__ == "__main__":
    main()
