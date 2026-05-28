# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-distances-v2-on-zephyr driver.

Two subcommands:

* ``generate`` — runs the v2 generator at scale on the marin Iris
  cluster. Input is the timodonnell/afdb-1.6M parquet manifest on
  HuggingFace; per-row mmCIFs are fetched concurrently from the
  ``gcs_uri`` column inside each Zephyr shard (the perf finding
  from the prior production run: sequential per-row GCS fetches
  were the residual bottleneck once the URI mode killed the
  cross-cloud HF egress).

* ``tokenizer`` — build / save / push the v2 tokenizer (same as
  exp34's; included so a downstream training job can pin a single
  experiment for both the dataset and the tokenizer build).

Run examples are in the experiment README.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import typing
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

from fray import ResourceConfig
from marinfold import build_tokenizer
from zephyr import Dataset, ZephyrContext

import csr_store
import generate
import parse
from vocab import CONTEXT_LENGTH, NAME, all_domain_tokens


# Manifest columns we'll copy verbatim to the output *if* the input parquet
# schema has them. afdb-1.6M ships every one of these; minimal test
# manifests typically have only a subset, which the schema-peek in
# ``_resolve_input_columns`` filters down at submission time.
_OPTIONAL_PASSTHROUGH: tuple[str, ...] = (
    "split",
    "seq_cluster_id",
    "struct_cluster_id",
    "uniprot_accession",
    "tax_id",
    "organism_name",
    # The cif URI itself is provenance — surfaces the gs:// path each doc was
    # generated from. When --cif-uri-column='gcs_uri' (the default) this column
    # is also the data source, so ``_resolve_input_columns`` dedups before
    # passing the column list to ``load_parquet``.
    "gcs_uri",
)

# Contact-statement opener tokens — counted in the emitted doc to populate
# the ``contacts_emitted`` provenance column (matches the published
# reference dataset's column of the same name).
_CONTACT_MARKERS: frozenset[str] = frozenset((
    "<long-range-contact>",
    "<medium-range-contact>",
    "<short-range-contact>",
))


def _resolve_input_columns(input_path: str, cif_col: str) -> tuple[list[str], list[str]]:
    """Peek the first matching parquet file to decide which columns to load.

    Returns ``(columns_to_load, passthrough_present)`` — the second is the
    subset of ``_OPTIONAL_PASSTHROUGH`` actually present in the input schema,
    so the worker only emits keys it can fill. We do this once at submission
    time so the same column list is used for every shard's
    ``load_parquet(columns=...)`` call — keeping the output schema stable.
    """
    import fsspec
    import pyarrow.parquet as pq

    fs, _ = fsspec.core.url_to_fs(input_path)
    # fs.glob handles both a literal path and a glob (with ``**``). Take the
    # first match — schemas are uniform across the manifest's shards.
    matches = sorted(fs.glob(input_path))
    if not matches:
        raise FileNotFoundError(f"No parquet files match {input_path!r}")
    first = fs.unstrip_protocol(matches[0])
    with fsspec.open(first, "rb") as f:
        present = set(pq.ParquetFile(f).schema_arrow.names)
    if "entry_id" not in present:
        raise ValueError(
            f"{first}: input parquet is missing the required 'entry_id' column "
            f"(found {sorted(present)})"
        )
    if cif_col not in present:
        raise ValueError(
            f"{first}: input parquet is missing the cif column {cif_col!r} "
            f"(found {sorted(present)})"
        )
    passthrough = [c for c in _OPTIONAL_PASSTHROUGH if c in present]
    # Dedup — if the cif data-source column ('gcs_uri' by default) is also a
    # passthrough column, we only need to read it once. The output row still
    # gets it via the passthrough loop in _build_output_row.
    base = ["entry_id", cif_col]
    columns = base + [c for c in passthrough if c not in base]
    return columns, passthrough


def _build_output_row(row: dict, doc: str, structure: parse.ParsedStructure,
                      passthrough_columns: list[str]) -> dict:
    """Assemble one output row: doc text + computed provenance + manifest passthrough.

    ``global_plddt`` and ``seq_len`` use the *parsed-structure* values (what
    ``generate_one`` actually serialized into the doc's pLDDT bin and
    sequence-token block), not the manifest's. Keeps each row internally
    consistent — the metadata describes what was emitted.
    """
    contacts_emitted = sum(1 for t in doc.split() if t in _CONTACT_MARKERS)
    out = {
        "entry_id": row.get("entry_id"),
        "structure": NAME,
        "document": doc,
        "sha1": hashlib.sha1(doc.encode()).hexdigest(),
        "seq_len": int(structure.num_residues),
        "global_plddt": float(structure.global_plddt),
        "contacts_emitted": int(contacts_emitted),
    }
    for col in passthrough_columns:
        out[col] = row.get(col)
    return out


# --------------------------------------------------------------------------
# Per-row worker function (pickled + shipped to Zephyr workers)
# --------------------------------------------------------------------------


def _generate_doc_for_row(
    row: dict,
    *,
    cif_uri_column: str,
    cif_text_column: str | None,
    context_length: int,
    cfg: generate.GenerationConfig,
    passthrough_columns: list[str],
) -> dict | None:
    """Fetch + parse + generate one document for one parquet row.

    Returns an output-row dict (with ``document`` + provenance metadata) on
    success, or ``None`` when the structure was unfetchable / unparseable /
    too small to fit the token budget. Designed to be safe inside a Zephyr
    worker — never raises on transient I/O errors.
    """
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
    doc = generate.generate_one(structure, context_length=context_length, cfg=cfg)
    if doc is None:
        return None
    return _build_output_row(row, doc, structure, passthrough_columns)


def _generate_shard(
    items,
    shard_info,
    *,
    cif_uri_column: str,
    cif_text_column: str | None,
    context_length: int,
    cfg: generate.GenerationConfig,
    fetch_concurrency: int,
    passthrough_columns: list[str],
):
    """``map_shard`` body: fetch all rows' cifs concurrently within the shard.

    The CPU work (gemmi parse + doc generation) takes ~6 ms per row;
    the GCS GET is ~30-80 ms on a cold connection. Without intra-shard
    concurrency, per-shard latency is bounded by the sum of GETs
    (sequential I/O), which is what made run 233550 land at ~128 s/shard
    despite vectorized parse. A ThreadPoolExecutor of ``fetch_concurrency``
    workers overlaps the I/O — gemmi releases the GIL during the C++
    parse, so the threads make real progress in parallel.

    Order within the output shard mirrors input row order (we use
    ``executor.map`` not ``as_completed``). That makes the run
    deterministic per-shard, which matters for byte-comparing outputs
    across runs.
    """
    rows = list(items)
    if not rows:
        return
    worker = partial(
        _generate_doc_for_row,
        cif_uri_column=cif_uri_column,
        cif_text_column=cif_text_column,
        context_length=context_length,
        cfg=cfg,
        passthrough_columns=passthrough_columns,
    )
    # Cap concurrency at the actual row count (no point spawning 64 threads
    # for a 2-row test).
    workers = min(fetch_concurrency, len(rows))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exp5-fetch") as pool:
        for out_row in pool.map(worker, rows):
            if out_row is not None:
                yield out_row


# --------------------------------------------------------------------------
# parse-to-csr pipeline (one-time precompute → on-the-fly training dataloader)
# --------------------------------------------------------------------------


def _resolve_passthrough_with_types(input_path: str, cif_col: str
                                    ) -> tuple[list[str], dict]:
    """Like ``_resolve_input_columns`` but also returns the arrow types of the
    passthrough columns.

    The CSR writer needs a stable per-shard schema (one schema across all
    Zephyr shards so the output parquet is concatenable), and the passthrough
    columns' types come from the manifest. We peek the schema once at
    submission time so workers don't all redo the schema fetch.
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
    """Fetch + parse one manifest row, return its CSR dict (or None on failure).

    Same fetch+parse path as ``_generate_doc_for_row`` — kept as a separate
    function to keep the per-subcommand worker body explicit and to make it
    easy to swap the output side (CSR dict vs generated doc) at the Zephyr
    pipeline boundary.
    """
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
    """``map_shard`` body for the CSR precompute. Same thread-pool structure
    as the doc-generation shard worker — the fetch cost dominates either way.
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
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exp5-csr") as pool:
        for csr_row in pool.map(worker, rows):
            if csr_row is not None:
                yield csr_row


# --------------------------------------------------------------------------
# Subcommand handlers
# --------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    cfg = generate.GenerationConfig(
        contact_cutoff_angstrom=args.contact_cutoff_angstrom,
        contact_f_range=tuple(args.contact_f_range),
        contact_rank_mean=args.contact_rank_mean,
        distance_rank_mean=args.distance_rank_mean,
        rank_std=args.rank_std,
        residue_plddt_min=args.residue_plddt_min,
        think_initial_prob=args.think_initial_prob,
        think_initial_geom_p=args.think_initial_geom_p,
        think_additional_count_range=tuple(args.think_additional_count_range),
        think_run_length_geom_p=args.think_run_length_geom_p,
    )

    # Column selection on the parquet manifest. The default cif source is
    # --cif-uri-column='gcs_uri' (the fast path: ~160 KB/shard instead of
    # ~70 MB); --cif-text-column overrides for inline cif_content. Schema
    # peek decides which optional passthrough columns to load (afdb-1.6M
    # has them all; test fixtures often don't).
    cif_col = args.cif_text_column or args.cif_uri_column
    columns, passthrough_columns = _resolve_input_columns(args.input, cif_col)

    rows = Dataset.from_files(args.input).load_parquet(columns=columns)
    if args.num_docs is not None:
        # Global cap; collapses to a single shard so the take is deterministic
        # across runs. Same semantics as exp34's --num-docs.
        rows = rows.reshard(1).take_per_shard(args.num_docs)

    # ``_generate_shard`` yields output-row dicts (document + provenance);
    # no extra .map wrapping needed before write_parquet.
    out_rows = rows.map_shard(partial(
        _generate_shard,
        cif_uri_column=args.cif_uri_column,
        cif_text_column=args.cif_text_column,
        context_length=args.context_length,
        cfg=cfg,
        fetch_concurrency=args.fetch_concurrency,
        passthrough_columns=passthrough_columns,
    ))

    # Output sharding: a {shard} placeholder writes one file per input
    # shard (good for the full 1.6M-structure run; ~1000 output files);
    # otherwise collapse to a single file.
    if "{shard" not in args.out:
        out_rows = out_rows.reshard(1)

    suffix = os.path.splitext(args.out)[1]
    match suffix:
        case ".parquet":
            ds = out_rows.write_parquet(args.out)
        case ".jsonl" | ".json":
            ds = out_rows.write_jsonl(args.out)
        case _:
            typing.assert_never(suffix)

    ctx = ZephyrContext(
        max_workers=args.max_workers,
        resources=ResourceConfig(
            cpu=args.worker_cpu,
            ram=args.worker_memory,
            disk=args.worker_disk,
        ),
    )
    ctx.execute(ds)
    print(f"[{NAME}] wrote {args.out}", file=sys.stderr)


def cmd_parse_to_csr(args: argparse.Namespace) -> None:
    """One-time precompute: AFDB CIFs → CSR parquet (~30-50× smaller than CIFs,
    skips both the GCS GET and the gemmi parse at training time).

    Pipeline shape mirrors ``cmd_generate``: same fetch concurrency, same
    passthrough wishlist, same per-shard output sharding. The output rows are
    columnar :class:`parse.ParsedStructure` views (see :mod:`csr_store` for the
    schema).
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

    # Pin the output schema so every shard parquet has the same columns/types
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
    print(f"[{NAME}] wrote CSR parquet to {args.out}", file=sys.stderr)


def cmd_tokenizer(args: argparse.Namespace) -> None:
    tokenizer = build_tokenizer(all_domain_tokens())
    print(f"[{NAME}] built tokenizer with {len(tokenizer)} tokens", file=sys.stderr)
    did_anything = False
    if args.save_local is not None:
        out_dir = Path(args.save_local)
        out_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(str(out_dir))
        print(f"[{NAME}] saved tokenizer to {out_dir}", file=sys.stderr)
        did_anything = True
    if args.push is not None:
        tokenizer.push_to_hub(args.push, private=args.private)
        print(f"[{NAME}] pushed tokenizer to https://huggingface.co/{args.push}",
              file=sys.stderr)
        did_anything = True
    if not did_anything:
        sample = " ".join(all_domain_tokens()[:8])
        encoded = tokenizer.encode(sample, add_special_tokens=False)
        print(f"sample: {sample!r}")
        print(f"  ids:  {encoded}")
        print(f"  back: {tokenizer.decode(encoded)!r}")
        print("Use --save-local DIR or --push REPO to persist the tokenizer.")


# --------------------------------------------------------------------------
# Argparse
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python cli.py",
        description=f"{NAME} on zephyr — generate / tokenizer.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- generate ----------------------------------------------------------
    p_gen = sub.add_parser("generate", help="Generate v2 training documents at scale.")
    p_gen.add_argument(
        "--input", type=str, required=True,
        help="Parquet file/glob for the input manifest (e.g. "
             "hf://datasets/timodonnell/afdb-1.6M/**/*.parquet). Rows need "
             "an 'entry_id' column plus either --cif-uri-column (default, "
             "fast path) or --cif-text-column (fallback, streams inline cif).",
    )
    p_gen.add_argument(
        "--cif-uri-column", type=str, default="gcs_uri",
        help="Column holding a URI per row. Workers fetch the structure "
             "from that URI directly — much faster than streaming a bulky "
             "cif column cross-cloud. Default: 'gcs_uri' (the column in "
             "afdb-1.6M pointing at the public AFDB bucket).",
    )
    p_gen.add_argument(
        "--cif-text-column", type=str, default=None,
        help="Optional fallback: column holding the mmCIF text inline. "
             "When set, overrides --cif-uri-column and reads the bulk "
             "cif column from the manifest (slow path; for testing or "
             "datasets without URI columns).",
    )
    p_gen.add_argument(
        "--num-docs", type=int, default=None,
        help="Global cap on total documents produced (writes a single "
             "merged file). Caps documents written, not parquet shards "
             "scanned — pair with a bounded --input for a cheap sample.",
    )
    p_gen.add_argument(
        "--context-length", type=int, default=CONTEXT_LENGTH,
        help="Token budget per document (default 8192).",
    )
    p_gen.add_argument(
        "--out", type=str, required=True,
        help="Output path (.parquet or .jsonl), local or cloud URL. "
             "Include a {shard} placeholder (e.g. corpus-{shard:05d}-of-"
             "{total:05d}.parquet) to write one file per input shard; "
             "omit it for a single merged file.",
    )

    # ---- Zephyr worker resources ------------------------------------------
    # Defaults bake in the perf lesson from the prior production runs: 1 CPU
    # per worker (per-shard CPU work is single-threaded once the in-shard
    # ThreadPool releases the GIL on I/O), scale out via --max-workers.
    p_gen.add_argument("--worker-cpu", type=float, default=1)
    p_gen.add_argument("--worker-memory", type=str, default="4g")
    p_gen.add_argument("--worker-disk", type=str, default="32g")
    p_gen.add_argument(
        "--max-workers", type=int, default=None,
        help="Cap concurrent Zephyr workers (default: 128 for distributed, "
             "or ZEPHYR_MAX_WORKERS).",
    )
    p_gen.add_argument(
        "--fetch-concurrency", type=int, default=32,
        help="Concurrent URI fetches per shard (ThreadPoolExecutor inside "
             "map_shard). Overlaps the per-row gs:// GETs with gemmi parse. "
             "Default 32 — the right ballpark for ~30-80 ms GCS GETs and a "
             "~6 ms CPU step per row.",
    )

    # ---- v2 generation knobs (same defaults as exp34) ---------------------
    cfg_defaults = generate.GenerationConfig()
    p_gen.add_argument("--contact-cutoff-angstrom", type=float,
                       default=cfg_defaults.contact_cutoff_angstrom)
    p_gen.add_argument("--residue-plddt-min", type=float,
                       default=cfg_defaults.residue_plddt_min)
    p_gen.add_argument("--contact-f-range", type=float, nargs=2,
                       metavar=("LOW", "HIGH"),
                       default=list(cfg_defaults.contact_f_range))
    p_gen.add_argument("--contact-rank-mean", type=float,
                       default=cfg_defaults.contact_rank_mean)
    p_gen.add_argument("--distance-rank-mean", type=float,
                       default=cfg_defaults.distance_rank_mean)
    p_gen.add_argument("--rank-std", type=float, default=cfg_defaults.rank_std)
    p_gen.add_argument("--think-initial-prob", type=float,
                       default=cfg_defaults.think_initial_prob)
    p_gen.add_argument("--think-initial-geom-p", type=float,
                       default=cfg_defaults.think_initial_geom_p)
    p_gen.add_argument("--think-additional-count-range", type=float, nargs=2,
                       metavar=("LOW", "HIGH"),
                       default=list(cfg_defaults.think_additional_count_range))
    p_gen.add_argument("--think-run-length-geom-p", type=float,
                       default=cfg_defaults.think_run_length_geom_p)
    p_gen.set_defaults(func=cmd_generate)

    # ---- parse-to-csr ------------------------------------------------------
    p_csr = sub.add_parser(
        "parse-to-csr",
        help="One-time precompute: AFDB CIFs → CSR parquet for on-the-fly "
             "training-time doc generation. Drops both the GCS GET and the "
             "gemmi parse from the training-loop hot path.",
    )
    p_csr.add_argument("--input", type=str, required=True,
                       help="Same as `generate --input`: parquet manifest URI/glob.")
    p_csr.add_argument("--cif-uri-column", type=str, default="gcs_uri")
    p_csr.add_argument("--cif-text-column", type=str, default=None)
    p_csr.add_argument(
        "--num-structures", type=int, default=None,
        help="Global cap on total structures emitted (writes a single merged "
             "file). For sharded output use a {shard} placeholder in --out.",
    )
    p_csr.add_argument(
        "--out", type=str, required=True,
        help="Output parquet path/pattern. Include {shard:05d} for one file "
             "per input shard (recommended for the full 1.6M run).",
    )
    p_csr.add_argument("--worker-cpu", type=float, default=1)
    p_csr.add_argument("--worker-memory", type=str, default="4g")
    p_csr.add_argument("--worker-disk", type=str, default="32g")
    p_csr.add_argument("--max-workers", type=int, default=None)
    p_csr.add_argument("--fetch-concurrency", type=int, default=32)
    p_csr.set_defaults(func=cmd_parse_to_csr)

    # ---- tokenizer ---------------------------------------------------------
    p_tok = sub.add_parser("tokenizer", help="Build / save / push the v2 tokenizer.")
    p_tok.add_argument("--save-local", type=Path, default=None)
    p_tok.add_argument("--push", type=str, default=None)
    p_tok.add_argument("--private", action="store_true")
    p_tok.set_defaults(func=cmd_tokenizer)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
