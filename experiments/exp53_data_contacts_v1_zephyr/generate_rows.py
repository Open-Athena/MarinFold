# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B per-row generation — fetch one structure, emit one document.

This module is the *worker* half of Stage B. It deliberately has **no
zephyr import** so it can be unit-tested locally (against inline
``cif_content``) without the marin distributed runtime. ``cli.py`` next
door wraps these functions in a ``zephyr`` ``map_shard`` to fan out on
Iris.

It calls **straight into** marinfold's contacts-v1 generator
(``generate_document`` → ``metadata_row``) — exp53 does not re-implement
any document logic (issue #53). Each input row is one selected
(entry, round) record from the Stage-A manifest; the structure is fetched
from the row's ``gcs_uri`` (the at-scale path) or read from an inline
``cif_content`` column (the local-test path), parsed with gemmi, and handed
to ``generate_document`` with the manifest ``entry_id`` as the deterministic
seed. The output row is contacts-v1's ``metadata_row()`` plus the manifest
provenance columns (``round``, ``struct_cluster_id``, …).
"""

import warnings
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import gemmi

from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    generate_document,
)
from marinfold.document_structures.contacts_v1.vocab import CONTEXT_LENGTH, NAME

# Manifest columns copied verbatim onto every output row (provenance: lets a
# generated doc be traced back to its cluster / round / split / source).
# ``entry_id`` comes from contacts-v1's ``metadata_row`` (same value — it is
# the generation seed), so it is not repeated here.
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


# Process-wide cache of the parsed rotamer library (or None on failure). A
# Zephyr worker processes many shards; parsing pyconfind's (large) EBL.out is
# ~tens of seconds, so we do it once per worker process, not once per shard.
_ROTAMER_UNSET: Any = object()
_ROTAMER_LIBRARY: Any = _ROTAMER_UNSET


def _load_rotamer_library() -> Any | None:
    """Parse pyconfind's Dunbrack rotamer library once per process (best effort).

    Memoized process-wide and passed explicitly to every ``generate_document``
    call so pyconfind does not re-parse the (large) ``EBL.out`` for each
    structure or each shard, and so the shard's fetch threads don't race on
    pyconfind's lazy process-wide default. ``cached_rotamer_library()``
    downloads + caches the library per user and returns its path;
    ``load_library`` parses it. Returns ``None`` on failure —
    ``generate_document(rotamer_library=None)`` then falls back to pyconfind's
    own lazy load (slower, but correct).
    """
    global _ROTAMER_LIBRARY
    if _ROTAMER_LIBRARY is not _ROTAMER_UNSET:
        return _ROTAMER_LIBRARY
    try:
        from pyconfind import load_library

        try:
            from pyconfind import cached_rotamer_library
        except ImportError:
            from pyconfind.data import cached_rotamer_library

        _ROTAMER_LIBRARY = load_library(cached_rotamer_library())
    except Exception as exc:  # noqa: BLE001 — optional speedup, never fatal
        warnings.warn(f"rotamer-library preload failed ({exc}); per-call load",
                      stacklevel=2)
        _ROTAMER_LIBRARY = None
    return _ROTAMER_LIBRARY


def structure_from_cif(data: str | bytes, *, entry_id: str) -> gemmi.Structure:
    """Parse mmCIF text/bytes into a ``gemmi.Structure`` (no temp file)."""
    if isinstance(data, bytes):
        data = data.decode("utf-8", "replace")
    structure = gemmi.read_structure_string(data)
    if not structure.name:
        structure.name = str(entry_id)
    return structure


def _fetch_cif_bytes(uri: str) -> bytes | None:
    """Fetch a cif from a URI, reading the *whole* object. ``None`` on error.

    Uses the filesystem's ``cat_file`` (a single full GET) rather than a
    seekable ``open().read()``. The AFDB GCS objects are gzip-transcoded
    (``Content-Encoding: gzip``) and report their *compressed* size, so a
    size/range-based read returns only that many bytes of the decompressed
    stream and silently truncates larger structures mid-``_atom_site`` loop
    (gemmi then raises "Wrong number of values in loop _atom_site"). A plain
    GET lets GCS decompressively transcode and we read to EOF -- the full mmCIF.
    """
    import fsspec

    try:
        fs, path = fsspec.core.url_to_fs(uri)
        return fs.cat_file(path)
    except (OSError, ValueError) as exc:
        warnings.warn(f"fetch failed for {uri}: {exc}", stacklevel=2)
        return None


def build_output_row(row: dict, result, *, structure_name: str = NAME) -> dict[str, Any]:
    """Assemble one output row: contacts-v1 metadata + manifest provenance.

    ``result`` is a ``GenerationResult``. ``metadata_row()`` already carries
    ``entry_id`` (the manifest seed), the document text, and every contacts-v1
    metadata column; we prepend the structure name and append the manifest
    passthrough columns. Stable key set across all rows (parquet schema).
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

    Returns the output-row dict on success, or ``None`` when the structure
    is unfetchable / unparseable / multi-chain / outside contacts-v1's
    serializable range. Never raises on a single bad row — a dropped row
    just shrinks its round slightly (issue #53: drop, don't backfill).
    """
    entry_id = row.get("entry_id")
    if cif_text_column is not None:
        cif = row.get(cif_text_column)
        if not cif:
            return None
    else:
        uri = row.get(cif_uri_column)
        if not uri:
            return None
        cif = _fetch_cif_bytes(uri)
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
    fetches run on a ``ThreadPoolExecutor`` so the per-row GCS GETs overlap
    pyconfind's contact computation (pyconfind releases the GIL in its
    compiled backend). For the inline-cif path there is no I/O to overlap,
    so rows run sequentially. Output order mirrors input order
    (``pool.map``), keeping per-shard output deterministic.
    """
    rows = list(items)
    if not rows:
        return
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
        for row in rows:
            out = worker(row)
            if out is not None:
                yield out
        return
    workers = min(fetch_concurrency, len(rows))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exp53-fetch") as pool:
        for out in pool.map(worker, rows):
            if out is not None:
                yield out
