# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""CSR substrate driver — one subcommand for the one-time precompute.

``parse-to-csr`` runs the AFDB-CIF → parsed-structure CSR parquet pipeline
at scale on the marin Iris cluster. Same input shape as exp5's ``generate``
(parquet manifest with ``entry_id`` + ``gcs_uri``), same concurrent-fetch
+ in-region-bucket lessons. The output is the canonical substrate every
downstream doc-format experiment reads.

Doc generation lives elsewhere (per-format experiment, e.g. exp5 for v2)
— this experiment intentionally has no opinion on document format.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext

import csr_store
import parse


# Manifest columns we'll copy verbatim to the CSR output *if* the input
# parquet schema has them. afdb-1.6M ships every one; minimal test
# manifests typically have only a subset. ``gcs_uri`` is special: when
# it's also the data source (the default), it's deduplicated before being
# passed to ``load_parquet(columns=...)``.
_OPTIONAL_PASSTHROUGH: tuple[str, ...] = (
    "split",
    "seq_cluster_id",
    "struct_cluster_id",
    "uniprot_accession",
    "tax_id",
    "organism_name",
    "gcs_uri",
)


def _resolve_passthrough_with_types(input_path: str, cif_col: str
                                    ) -> tuple[list[str], dict]:
    """Peek the first matching parquet file to decide which columns to load
    *and* what arrow types the passthrough columns have.

    The CSR writer needs a stable per-shard schema (one schema across all
    Zephyr shards so the output is concatenable), and the passthrough
    columns' types come from the manifest. We peek once at submission
    time so workers don't all re-fetch.
    """
    import fsspec
    import pyarrow.parquet as pq

    fs, _ = fsspec.core.url_to_fs(input_path)
    matches = sorted(fs.glob(input_path))
    if not matches:
        raise FileNotFoundError(f"No parquet files match {input_path!r}")
    first = fs.unstrip_protocol(matches[0])
    with fsspec.open(first, "rb") as f:
        schema = pq.ParquetFile(f).schema_arrow
    if "entry_id" not in schema.names:
        raise ValueError(f"{first}: missing required 'entry_id' column")
    if cif_col not in schema.names:
        raise ValueError(f"{first}: missing cif column {cif_col!r}")
    passthrough = [c for c in _OPTIONAL_PASSTHROUGH if c in schema.names]
    passthrough_types = {c: schema.field(c).type for c in passthrough}
    columns = ["entry_id", cif_col] + [c for c in passthrough if c not in ("entry_id", cif_col)]
    return columns, passthrough_types


def _parse_row_to_csr(
    row: dict,
    *,
    cif_uri_column: str,
    cif_text_column: str | None,
    passthrough_columns: list[str],
) -> dict | None:
    """Fetch + parse one manifest row, return its CSR dict (or ``None`` on
    failure). Safe inside a Zephyr worker — never raises on transient I/O."""
    entry_id = row.get("entry_id")
    if cif_text_column is not None:
        cif = row.get(cif_text_column)
        if cif is None:
            return None
        structure = parse.try_parse_cif_content(cif, entry_id=entry_id, source=entry_id)
    else:
        uri = row.get(cif_uri_column)
        if uri is None:
            return None
        structure = parse.try_parse_cif_from_uri(uri, entry_id=entry_id)
    if structure is None:
        return None
    passthrough = {col: row.get(col) for col in passthrough_columns}
    return csr_store.parsed_structure_to_row(structure, passthrough=passthrough)


def _parse_to_csr_shard(
    items,
    shard_info,
    *,
    cif_uri_column: str,
    cif_text_column: str | None,
    fetch_concurrency: int,
    passthrough_columns: list[str],
):
    """``map_shard`` body: fetch all rows' cifs concurrently within the shard.

    Same lesson as exp5's generate: the per-row GCS GET (~30-80 ms) dwarfs
    the gemmi parse (~6 ms), so a ThreadPoolExecutor overlapping the GETs
    with parse is the single biggest per-shard speedup. gemmi releases the
    GIL during the C++ parse, so threads make real progress in parallel.
    """
    rows = list(items)
    if not rows:
        return
    worker = partial(
        _parse_row_to_csr,
        cif_uri_column=cif_uri_column,
        cif_text_column=cif_text_column,
        passthrough_columns=passthrough_columns,
    )
    workers = min(fetch_concurrency, len(rows))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exp42-csr") as pool:
        for csr_row in pool.map(worker, rows):
            if csr_row is not None:
                yield csr_row


def cmd_parse_to_csr(args: argparse.Namespace) -> None:
    """One-time precompute: AFDB CIFs → CSR parquet shards on GCS.

    Pipeline mirrors exp5's ``generate``: same per-row fetch concurrency,
    same passthrough wishlist, same per-shard output naming. The output
    rows are columnar :class:`parse.ParsedStructure` views (see
    :mod:`csr_store` for the schema).
    """
    cif_col = args.cif_text_column or args.cif_uri_column
    columns, passthrough_types = _resolve_passthrough_with_types(args.input, cif_col)
    passthrough_columns = list(passthrough_types.keys())

    rows = Dataset.from_files(args.input).load_parquet(columns=columns)
    if args.num_structures is not None:
        rows = rows.reshard(1).take_per_shard(args.num_structures)

    csr_rows = rows.map_shard(partial(
        _parse_to_csr_shard,
        cif_uri_column=args.cif_uri_column,
        cif_text_column=args.cif_text_column,
        fetch_concurrency=args.fetch_concurrency,
        passthrough_columns=passthrough_columns,
    ))

    if "{shard" not in args.out:
        csr_rows = csr_rows.reshard(1)

    # Pin output schema so every shard parquet has identical columns/types
    # → the resulting dataset is directly concatenable by the dataloader.
    schema = csr_store.schema_with_passthrough(passthrough_types)
    ds = csr_rows.write_parquet(args.out, schema=schema)

    ctx = ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(
            cpu=args.worker_cpu,
            ram=args.worker_memory,
            disk=args.worker_disk,
        ),
    )
    ctx.execute(ds)
    print(f"[exp42] wrote CSR parquet to {args.out}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description="CSR substrate precompute (AFDB CIFs → ParsedStructure parquet).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_csr = sub.add_parser(
        "parse-to-csr",
        help="One-time precompute: AFDB CIFs → CSR parquet shards on GCS.",
    )
    p_csr.add_argument(
        "--input", type=str, required=True,
        help="Parquet manifest URI/glob (e.g. "
             "hf://datasets/timodonnell/afdb-1.6M/**/*.parquet). Rows need "
             "an 'entry_id' column plus either --cif-uri-column (default, "
             "fast path) or --cif-text-column (fallback, streams inline cif).",
    )
    p_csr.add_argument(
        "--cif-uri-column", type=str, default="gcs_uri",
        help="Column holding a URI per row. Workers fetch the structure "
             "from that URI directly. Default: 'gcs_uri' (afdb-1.6M column "
             "pointing at the public AFDB bucket).",
    )
    p_csr.add_argument(
        "--cif-text-column", type=str, default=None,
        help="Optional fallback: column holding the mmCIF text inline. "
             "Overrides --cif-uri-column when set (slow path).",
    )
    p_csr.add_argument(
        "--num-structures", type=int, default=None,
        help="Global cap on total structures emitted (writes a single "
             "merged file). For sharded output use a {shard} placeholder.",
    )
    p_csr.add_argument(
        "--out", type=str, required=True,
        help="Output parquet path/pattern. Include {shard:05d} for one "
             "file per input shard (recommended for the full 1.6M run).",
    )

    # ---- Zephyr worker resources -----------------------------------------
    p_csr.add_argument("--worker-cpu", type=float, default=1)
    p_csr.add_argument("--worker-memory", type=str, default="4g")
    p_csr.add_argument("--worker-disk", type=str, default="32g")
    p_csr.add_argument("--max-workers", type=int, default=None)
    p_csr.add_argument(
        "--fetch-concurrency", type=int, default=32,
        help="Concurrent URI fetches per shard (ThreadPoolExecutor inside "
             "map_shard). Overlaps the per-row gs:// GETs with gemmi parse.",
    )
    p_csr.set_defaults(func=cmd_parse_to_csr)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
