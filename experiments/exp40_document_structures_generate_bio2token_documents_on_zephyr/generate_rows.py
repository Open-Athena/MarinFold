# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-shard worker: fetch a shard's structures and emit their documents.

The worker half of the pipeline (``cli.py`` is the driver). It has **no zephyr
import** so the whole worker path unit-tests locally (against inline
``cif_content``) without the distributed runtime; ``cli.py`` wraps
:func:`generate_shard` in a ``zephyr`` ``map_shard`` to fan out on Iris TPUs.

The per-row cost here is a neural forward pass on an accelerator, not network
I/O, and the Python scan inside the encoder holds the GIL — so running one row
per thread would not parallelize the compute. Each shard is therefore processed
in two phases:

1. **Fetch + parse concurrently** (I/O bound; the GCS GET and the gemmi parse
   both release the GIL) — a thread pool over the shard's rows.
2. **Tokenize the whole shard in bucketed batches** on the accelerator — one
   compiled graph per length bucket, and batching is what keeps the accelerator
   busy (see ``tokenizer.py``).

Fail-loud by default (``on_error="raise"``): a bad fetch/parse kills the worker
with a real traceback so a corrupt corpus is never silently shipped. The only
``None`` is the ``max_context`` filter — a structure whose document exceeds the
token budget is skipped by design. ``--on-error skip`` warns and drops instead.
"""

import warnings
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from adapt import ParsedStructure, parse_structure
from generate import generate_documents
from io_utils import read_object_bytes
from tokenizer import DEFAULT_BUCKETS, DEFAULT_MAX_BATCH, DEFAULT_MAX_BATCH_TOKENS
from vocab import NAME

DEFAULT_CIF_URI_COLUMN = "gcs_uri"

# Manifest provenance columns copied verbatim onto every output row (so a
# generated doc can be traced to its cluster / split / round / source).
# ``entry_id`` is already carried by the document row (it is the same value).
PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "struct_cluster_id",
    "seq_cluster_id",
    "global_plddt",
    "seq_len",
    "split",
    "uniprot_accession",
    "tax_id",
    "organism_name",
    "round",
)


def _fetch_and_parse(
    row: dict,
    *,
    cif_uri_column: str,
    cif_text_column: str | None,
    on_error: str,
) -> tuple[dict, ParsedStructure] | None:
    """Fetch + parse one row into ``(row, ParsedStructure)`` (I/O + gemmi only).

    With ``on_error == "raise"`` (default), a missing source column and any
    fetch or parse failure all raise (fail-loud — a null/empty source column is
    bad data, not a designed-in filter, and silently dropping it would ship an
    empty corpus with a 0 exit). With ``on_error == "skip"`` they are warned and
    dropped instead.
    """
    lenient = on_error == "skip"
    entry_id = row.get("entry_id")
    src_column = cif_text_column if cif_text_column is not None else cif_uri_column
    source = row.get(src_column)
    if not source:
        if not lenient:
            raise ValueError(f"{entry_id}: missing/empty source column {src_column!r}")
        warnings.warn(f"{entry_id}: missing/empty {src_column!r}; skipped", stacklevel=2)
        return None
    if cif_text_column is not None:
        cif = source  # inline cif text (local testing path)
    else:
        # raise-mode: any fetch failure kills the shard; skip-mode: warn + drop.
        cif = read_object_bytes(source, missing_ok=lenient)
        if cif is None:
            return None
    try:
        parsed = parse_structure(
            cif, entry_id=str(entry_id) if entry_id is not None else None)
    except (ValueError, RuntimeError) as exc:
        if on_error != "skip":
            raise
        warnings.warn(f"parse failed for {entry_id}: {exc}", stacklevel=2)
        return None
    return row, parsed


def generate_shard(
    items: Iterable[dict],
    shard_info: Any = None,
    *,
    cif_uri_column: str = DEFAULT_CIF_URI_COLUMN,
    cif_text_column: str | None = None,
    device: str = "xla",
    buckets: tuple[int, ...] = DEFAULT_BUCKETS,
    max_batch: int = DEFAULT_MAX_BATCH,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    max_context: int | None = None,
    fetch_concurrency: int = 32,
    on_error: str = "raise",
    structure_name: str = NAME,
) -> Iterator[dict[str, Any]]:
    """Generate documents for one shard's rows (the ``map_shard`` body).

    Signature ``(items, shard_info, *, ...)`` matches zephyr's ``map_shard``
    contract, but this is plain Python and unit-tested directly. It fetches +
    parses the shard concurrently, tokenizes it in bucketed batches on
    ``device``, then assembles output rows (document + manifest provenance) and
    yields them in input order.
    """
    rows = list(items)
    if not rows:
        return

    fetch = partial(_fetch_and_parse, cif_uri_column=cif_uri_column,
                    cif_text_column=cif_text_column, on_error=on_error)
    if cif_text_column is not None:
        # Inline path (local testing): no network I/O to overlap.
        pairs = [fetch(r) for r in rows]
    else:
        with ThreadPoolExecutor(max_workers=min(fetch_concurrency, len(rows)),
                                thread_name_prefix="exp40-fetch") as pool:
            pairs = list(pool.map(fetch, rows))

    survivors = [p for p in pairs if p is not None]
    if not survivors:
        return
    src_rows = [p[0] for p in survivors]
    parsed_list = [p[1] for p in survivors]

    # Compute: one batched forward pass per length bucket (the throughput lever).
    doc_rows = generate_documents(
        parsed_list, device=device, buckets=buckets, max_batch=max_batch,
        max_batch_tokens=max_batch_tokens, max_context=max_context)

    # Assemble + yield in input order.
    for src, doc in zip(src_rows, doc_rows, strict=True):
        if doc is None:
            continue  # designed-in: exceeded max_context
        out = dict(doc)  # already carries structure/entry_id/sequence/document/…
        out["structure"] = structure_name
        for col in PASSTHROUGH_COLUMNS:
            if col in src:
                out[col] = src.get(col)
        yield out
