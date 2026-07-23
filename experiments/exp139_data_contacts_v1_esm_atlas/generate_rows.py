# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-row generation — analyze one ESM-Atlas structure ONCE, emit one combined row.

The *worker* half of the pipeline. It deliberately has **no zephyr import**
so it can be unit-tested locally (against inline mmCIF text) without the marin
distributed runtime. ``cli.py`` next door wraps these functions in a zephyr
``map_shard`` to fan out on Iris.

Unlike exp53 / exp105 (AFDB, per-row ``gcs_uri`` fetch), the ESM-Atlas
distillation parts carry the mmCIF **inline** in a ``cif_content`` column, so
there is no per-row network fetch — pyconfind is the only real work.

The one structural novelty of this experiment: we run pyconfind **once** per
structure (:func:`analyze_structure`) and emit a single row that carries
**both**

- the reusable raw pyconfind contacts (:func:`analyzed_to_row` — residues +
  all ``degree > 0`` contacts, *before* any document-format filtering), and
- the contacts-v1 document itself (:func:`build_document` →
  ``GenerationResult.metadata_row``).

so a future doc type (e.g. contacts-and-crops-v1) can be generated from the
saved contacts without paying for pyconfind again. Publishing splits this one
combined dataset into the two logical views (see ``publish.py``).

Drop policy (lenient, like exp53/exp105): a structure that is unparseable /
multi-chain / outside the serializable residue range (``[2, 2000]``) is
dropped whole — **no** document *and* **no** contacts row — so the documents
and contacts datasets stay 1:1.
"""

import functools
import warnings
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any

import gemmi

from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    analyze_structure,
    analyzed_to_row,
    build_document,
)
from marinfold.document_structures.contacts_v1.vocab import CONTEXT_LENGTH, NAME

# Provenance columns copied verbatim from each ESM-Atlas part row onto the
# output (lets a generated doc / contacts record be traced back to its source
# cluster + confidence). ``entry_id`` comes from ``analyzed_to_row`` /
# ``metadata_row`` (same value — the generation seed), so it is not repeated.
PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "seq_cluster_id",
    "cluster_size",
    "ptm",
    "plddt_std",
    "source",
    "split",
)

# ESM-Atlas parts carry the mmCIF inline here (not a URI pointer like AFDB).
DEFAULT_CIF_TEXT_COLUMN = "cif_content"


@functools.cache
def _load_rotamer_library() -> Any | None:
    """Parse pyconfind's Dunbrack rotamer library once per process (best effort).

    Memoized process-wide and passed explicitly to every ``analyze_structure``
    call so pyconfind does not re-parse the (large) ``EBL.out`` per structure /
    per shard. Returns ``None`` on failure — ``analyze_structure`` then falls
    back to pyconfind's own lazy load (slower, but correct); the None is cached
    so a worker whose preload failed doesn't retry per structure.
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


def build_output_row(
    row: dict,
    analyzed,
    result,
    *,
    structure_name: str = NAME,
) -> dict[str, Any]:
    """Assemble one combined output row: saved contacts + document + provenance.

    - :func:`analyzed_to_row` supplies the reusable pyconfind record (residues
      + raw contacts + ``global_plddt`` + ``seq_len`` + ``num_contacts``).
    - ``result.metadata_row()`` supplies the contacts-v1 ``document`` text and
      its per-doc metadata (``entry_id``, ``seq_len``, ``global_plddt``,
      ``start_index``, contact-selection counts, ``num_tokens``, ``sha1``, …).
      The keys shared with ``analyzed_to_row`` (``entry_id``, ``seq_len``,
      ``global_plddt``) hold the *same* values (both derive from ``analyzed``),
      so the document metadata wins the union with no conflict.
    - ``PASSTHROUGH_COLUMNS`` carry the ESM-Atlas provenance.

    Stable key set across all rows (parquet schema).
    """
    out: dict[str, Any] = {"structure": structure_name}
    out.update(analyzed_to_row(analyzed))
    out.update(result.metadata_row())
    for col in PASSTHROUGH_COLUMNS:
        out[col] = row.get(col)
    return out


def generate_doc_for_row(
    row: dict,
    *,
    cif_text_column: str = DEFAULT_CIF_TEXT_COLUMN,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    rotamer_library: Any | None = None,
    structure_name: str = NAME,
) -> dict[str, Any] | None:
    """Parse + analyze (once) + build one combined row for one ESM-Atlas row.

    Returns the output-row dict on success, or ``None`` when the structure is
    unparseable / multi-chain / outside contacts-v1's serializable residue
    range (``[2, 2000]``). Never raises on a single bad row (drop, don't
    backfill).
    """
    entry_id = row.get("entry_id")
    cif = row.get(cif_text_column)
    if not cif:
        return None  # lenient: missing inline cif → skip
    try:
        structure = structure_from_cif(cif, entry_id=entry_id)
        analyzed = analyze_structure(
            structure,
            entry_id=str(entry_id) if entry_id is not None else None,
            native_only=config.native_only,
            contact_distance=config.contact_distance,
            dcut=config.dcut,
            clash_distance=config.clash_distance,
            assembly=config.assembly,
            rotamer_library=rotamer_library,
        )
        result = build_document(
            analyzed.entry_id,
            analyzed.residues,
            analyzed.contacts,
            context_length=context_length,
            config=config,
            global_plddt=analyzed.global_plddt,
        )
    except (ValueError, RuntimeError) as exc:
        # lenient: parse / analyze failures (multi-chain, invalid cif) →
        # warn + drop.
        warnings.warn(f"generate failed for {entry_id}: {exc}", stacklevel=2)
        return None
    if result is None:
        # <2 or >2000 residues, or the fixed section already overflows the
        # budget: not serializable as contacts-v1 (nor contacts-and-crops-v1,
        # which shares the 2000-position space) — drop the whole row so the
        # documents / contacts datasets stay 1:1.
        return None
    return build_output_row(row, analyzed, result, structure_name=structure_name)


def generate_shard(
    items: Iterable[dict],
    shard_info: Any = None,
    *,
    cif_text_column: str = DEFAULT_CIF_TEXT_COLUMN,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    structure_name: str = NAME,
) -> Iterator[dict[str, Any]]:
    """Generate combined rows for one part's rows (the ``map_shard`` body).

    Signature ``(items, shard_info, *, ...)`` matches zephyr's ``map_shard``
    contract, but the function is plain Python and unit-tested directly. The
    rotamer library is loaded once for the whole shard. The cif is inline, so
    there is no per-row I/O to overlap — rows run **sequentially** on the
    worker's single core (``--worker-cpu 1``; scale via ``--max-workers``).
    """
    rotamer_library = _load_rotamer_library()
    worker = partial(
        generate_doc_for_row,
        cif_text_column=cif_text_column,
        context_length=context_length,
        config=config,
        rotamer_library=rotamer_library,
        structure_name=structure_name,
    )
    for row in items:
        out = worker(row)
        if out is not None:
            yield out
