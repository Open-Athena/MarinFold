# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-row generation — fetch one structure, emit one contacts-and-coordinates-v1 document.

The *worker* half of the pipeline. It deliberately has **no zephyr import**
so it can be unit-tested locally (against inline mmCIF text) without the
marin distributed runtime. ``cli.py`` next door wraps these functions in a
zephyr ``map_shard`` to fan out on Iris.

It calls **straight into** marinfold's contacts-and-coordinates-v1 generator
(``generate_document`` → ``metadata_row``) — this experiment re-implements no
document logic. Each input row is one selected ``(entry, round)`` record from
exp53's contacts-v1 selection manifest (the *same proteins*); the structure
is fetched from the row's ``gcs_uri`` (the at-scale path) or read from an
inline mmCIF column (the local-test path), parsed with gemmi, and handed to
``generate_document`` with the manifest ``entry_id`` as the deterministic
seed. The output row is the format's ``metadata_row()`` plus the manifest
provenance columns.
"""

import functools
import warnings
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any

import gemmi

from marinfold.document_structures.contacts_and_coordinates_v1 import (
    GenerationConfig,
    generate_document,
)
from marinfold.document_structures.contacts_and_coordinates_v1.vocab import (
    CONTEXT_LENGTH,
    NAME,
)
from marinfold.document_structures.io import (
    read_object_bytes,
    thread_per_row_in_shard,
)

# Manifest columns copied verbatim onto every output row (provenance: lets a
# generated doc be traced back to its cluster / round / split / source).
# ``entry_id`` comes from the format's ``metadata_row`` (same value — it is the
# generation seed), so it is not repeated here.
PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "round",
    "struct_cluster_id",
    "seq_cluster_id",
    "split",
    "uniprot_accession",
    "tax_id",
    "organism_name",
)

DEFAULT_CIF_URI_COLUMN = "gcs_uri"


@functools.cache
def _load_rotamer_library() -> Any | None:
    """Parse pyconfind's Dunbrack rotamer library once per process (best effort).

    Memoized process-wide and passed explicitly to every ``generate_document``
    call so pyconfind does not re-parse the (large) ``EBL.out`` per structure /
    per shard, and so the shard's fetch threads don't race on pyconfind's lazy
    process-wide default. Returns ``None`` on failure — ``generate_document``
    then falls back to pyconfind's own lazy load (slower, but correct); the
    None is cached so a worker whose preload failed doesn't retry per shard.
    """
    try:
        from pyconfind import load_library

        try:
            from pyconfind import cached_rotamer_library
        except ImportError:
            from pyconfind.data import cached_rotamer_library

        return load_library(cached_rotamer_library())
    except Exception as exc:  # noqa: BLE001 — optional speedup, never fatal
        warnings.warn(f"rotamer-library preload failed ({exc}); per-call load",
                      stacklevel=2)
        return None


def structure_from_cif(data: str | bytes, *, entry_id: str) -> gemmi.Structure:
    """Parse mmCIF text/bytes into a ``gemmi.Structure`` (no temp file)."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    structure = gemmi.read_structure_string(data)
    if not structure.name:
        structure.name = str(entry_id)
    return structure


def build_output_row(row: dict, result, *, structure_name: str = NAME) -> dict[str, Any]:
    """Assemble one output row: format metadata + manifest provenance.

    ``result`` is a ``GenerationResult``. ``metadata_row()`` already carries
    ``entry_id`` (the manifest seed), the document text, and every metadata
    column; we prepend the structure name and append the manifest passthrough
    columns. Stable key set across all rows (parquet schema).
    """
    out: dict[str, Any] = {"structure": structure_name}
    out.update(result.metadata_row())
    for col in PASSTHROUGH_COLUMNS:
        out[col] = row.get(col)
    return out


def generate_doc_for_row(
    row: dict,
    *,
    cif_uri_column: str = DEFAULT_CIF_URI_COLUMN,
    cif_text_column: str | None = None,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    rotamer_library: Any | None = None,
    structure_name: str = NAME,
) -> dict[str, Any] | None:
    """Fetch + parse + generate one document for one manifest row.

    Returns the output-row dict on success, or ``None`` when the structure is
    unfetchable / unparseable / multi-chain / outside the serializable range
    (too few residues, or too large for the coordinate cube). Never raises on
    a single bad row — a dropped row just shrinks its round slightly (drop,
    don't backfill — same policy as exp53).
    """
    entry_id = row.get("entry_id")
    if cif_text_column is not None:
        cif = row.get(cif_text_column)
        if not cif:
            return None  # lenient: missing inline cif column → skip
    else:
        uri = row.get(cif_uri_column)
        if not uri:
            return None  # lenient: missing URI in row → skip
        cif = read_object_bytes(uri, missing_ok=True)  # lenient: skip on fetch failure
        if cif is None:
            return None
    try:
        structure = structure_from_cif(cif, entry_id=entry_id)
        result = generate_document(
            structure,
            entry_id=str(entry_id) if entry_id is not None else None,
            context_length=context_length,
            config=config,
            rotamer_library=rotamer_library,
        )
    except (ValueError, RuntimeError) as exc:
        # lenient: parse/generate failures (multi-chain, invalid cif,
        # out-of-range) are warned + dropped.
        warnings.warn(f"generate failed for {entry_id}: {exc}", stacklevel=2)
        return None
    if result is None:
        return None
    return build_output_row(row, result, structure_name=structure_name)


def generate_shard(
    items: Iterable[dict],
    shard_info: Any = None,
    *,
    cif_uri_column: str = DEFAULT_CIF_URI_COLUMN,
    cif_text_column: str | None = None,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    fetch_concurrency: int = 32,
    structure_name: str = NAME,
) -> Iterator[dict[str, Any]]:
    """Generate documents for one shard's rows (the ``map_shard`` body).

    Signature ``(items, shard_info, *, ...)`` matches zephyr's ``map_shard``
    contract, but the function is plain Python and unit-tested directly. The
    rotamer library is loaded once for the whole shard. For the URI path,
    delegates to :func:`marinfold.document_structures.io.thread_per_row_in_shard`
    so the per-row GCS GETs overlap pyconfind's contact computation (pyconfind
    releases the GIL in its compiled backend). For the inline-cif path there is
    no I/O to overlap, so rows run sequentially.
    """
    rotamer_library = _load_rotamer_library()
    worker = partial(
        generate_doc_for_row,
        cif_uri_column=cif_uri_column,
        cif_text_column=cif_text_column,
        context_length=context_length,
        config=config,
        rotamer_library=rotamer_library,
        structure_name=structure_name,
    )
    if cif_text_column is not None:
        for row in items:
            out = worker(row)
            if out is not None:
                yield out
        return
    yield from thread_per_row_in_shard(
        items, worker=worker,
        fetch_concurrency=fetch_concurrency,
        thread_name_prefix="exp105-fetch",
    )
