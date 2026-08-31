# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B per-row generation — one backbone -> 8 redesigned documents.

The *worker* half, with no zephyr import so it unit-tests locally.
``cli.py`` wraps ``generate_shard`` in a ``map_shard``.

**One pass, not two.** ProteinMPNN is a 1.7 M-parameter GNN, and on the
measured numbers a single CPU core runs it at ~0.6 s per 154-residue
protein for all 8 designs — only ~18x slower per sequence than an RTX A5000,
because the GPU path is launch-bound rather than compute-bound. That makes
"redesign on a GPU cluster, then generate documents on the CPU cluster"
a bad trade: it would stage 4 M AFDB cifs cross-cloud from GCS to
CoreWeave's S3, fetch every structure twice, and run two jobs, to save CPU
hours the Iris pool has (exp139 scaled to 512 workers in ~10 minutes).

So one ``map_shard`` does the whole thing per row: fetch the cif once, strip
to backbone once, design 8 sequences, and emit 8 contacts-v1 documents.
``--device cuda`` still works if the pilot says otherwise.

Everything else follows exp53: gzip-safe ``read_object_bytes``, threaded
per-row fetch, heavy init memoized per worker, fail-loud by default.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any

import gemmi

from backbone import (
    backbone_coords,
    prepare_structure,
    relabel_sequence,
    residue_sequence,
    strip_to_backbone,
)
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    generate_document,
)
from marinfold.document_structures.contacts_v1.vocab import CONTEXT_LENGTH, NAME
from marinfold.document_structures.io import read_object_bytes, thread_per_row_in_shard
from redesign import DESIGN_TEMPERATURES, BackboneEntry, design_batch

# Carried verbatim onto every output document so a redesigned row can be
# traced back to the native document it came from. ``entry_id`` comes from
# contacts-v1's own ``metadata_row`` (same value), so it is not repeated.
# ``global_plddt`` is NOT passed through: contacts-v1's ``metadata_row``
# recomputes it from CA B-factors, and stripping to backbone keeps CA, so the
# redesigned row carries the parent AFDB pLDDT unchanged. That is the right
# value — it describes confidence in the *backbone*, which is what we reused —
# but note it says nothing about whether the designed sequence folds there.
PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "struct_cluster_id",
    "seq_cluster_id",
    "split",
    "round",
    "native_contacts_emitted",
    "native_sha1",
)

DEFAULT_CIF_URI_COLUMN = "gcs_uri"


@functools.cache
def _load_rotamer_library() -> Any | None:
    """Parse pyconfind's Dunbrack rotamer library once per process.

    Tens of seconds per parse; doing it per shard instead of per worker is
    the mistake exp53 fixed in daa18e1 and the pipeline skill calls out as a
    41-hour cluster bill.
    """
    try:
        from pyconfind import cached_rotamer_library, load_library

        return load_library(cached_rotamer_library())
    except Exception as exc:  # pragma: no cover - best effort preload
        warnings.warn(f"rotamer preload failed ({exc}); per-call load", stacklevel=2)
        return None


def documents_for_row(
    row: dict,
    *,
    cif_uri_column: str = DEFAULT_CIF_URI_COLUMN,
    cif_text_column: str | None = None,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    rotamer_library: Any = None,
    structure_name: str = NAME,
    device: str = "cpu",
    temperatures: tuple[float, ...] = DESIGN_TEMPERATURES,
) -> list[dict[str, Any]] | None:
    """Fetch one backbone, redesign it, and emit one document per design.

    Returns ``None`` only for *designed-in filter predicates* — a structure
    contacts-v1 or ProteinMPNN refuses to handle. Everything else (I/O
    failure, parse failure, a sequence whose length disagrees with the
    backbone) raises, per the pipeline skill's fail-loud default: a silently
    dropped row corrupts the corpus, and a length disagreement would emit a
    document whose contacts belong to a different protein.
    """
    if cif_text_column is not None:
        data: str | bytes = row[cif_text_column]
    else:
        data = read_object_bytes(row[cif_uri_column])

    entry_id = row["entry_id"]
    backbone = strip_to_backbone(
        prepare_structure(_structure_from_cif(data, entry_id=entry_id))
    )

    native_sequence = residue_sequence(backbone)
    if "X" in native_sequence:
        # Designed-in filter: ProteinMPNN has no rotamer or token for a
        # non-canonical residue, and contacts-v1 would serialize it as <UNK>.
        # The parent corpus tolerates these; a redesign cannot.
        return None
    _chain_ids, coords = backbone_coords(backbone)

    designs = design_batch(
        [BackboneEntry(entry_id, native_sequence, coords)],
        device=device,
        temperatures=temperatures,
    )

    out: list[dict[str, Any]] = []
    for design in designs:
        relabelled = relabel_sequence(backbone, design.sequence)
        # Seed on (entry_id, design_index): each design gets its own
        # statement order and index offset, rather than eight documents
        # differing only in their amino acids.
        result = generate_document(
            relabelled,
            entry_id=f"{entry_id}#{design.design_index}",
            context_length=context_length,
            config=config,
            rotamer_library=rotamer_library,
        )
        if result is None:
            # Designed-in filter: contacts-v1 declined to serialize this
            # chain. The same predicate as the native corpus, so a
            # natively-serializable backbone can still fail here if the
            # redesign changed the contact budget.
            continue
        record = result.metadata_row()
        record["entry_id"] = entry_id          # undo the per-design seed suffix
        record["structure_name"] = structure_name
        record["design_index"] = design.design_index
        record["mpnn_temperature"] = design.mpnn_temperature
        record["mpnn_score"] = design.mpnn_score
        record["identity_to_native"] = design.identity_to_native
        for column in PASSTHROUGH_COLUMNS:
            if column in row:
                record[column] = row[column]
        out.append(record)
    return out or None


def _structure_from_cif(data: str | bytes, *, entry_id: str) -> gemmi.Structure:
    text = data.decode() if isinstance(data, bytes) else data
    structure = gemmi.read_structure_from_string(text, format=gemmi.CoorFormat.Mmcif)
    structure.name = entry_id
    return structure


def generate_shard(
    items: Iterable[dict],
    shard_info: Any = None,
    *,
    cif_uri_column: str = DEFAULT_CIF_URI_COLUMN,
    cif_text_column: str | None = None,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    fetch_concurrency: int = 4,
    structure_name: str = NAME,
    device: str = "cpu",
    temperatures: tuple[float, ...] = DESIGN_TEMPERATURES,
) -> Iterator[dict[str, Any]]:
    """The ``map_shard`` body: one input backbone -> up to 8 output rows.

    ``fetch_concurrency`` defaults far below exp53's 32 on purpose. There the
    per-row GCS GET (~30-80 ms) dwarfed the CPU step, so deep threading was
    the win. Here ProteinMPNN plus 8 pyconfind runs is seconds of compute per
    row, so the fetch is already hidden at low concurrency and extra threads
    only add contention on a ``--worker-cpu 1`` worker.
    """
    rotamer_library = _load_rotamer_library()
    worker = partial(
        documents_for_row,
        cif_uri_column=cif_uri_column,
        cif_text_column=cif_text_column,
        context_length=context_length,
        config=config,
        rotamer_library=rotamer_library,
        structure_name=structure_name,
        device=device,
        temperatures=temperatures,
    )
    if cif_text_column is not None:
        # Inline path: no I/O to overlap, so a thread pool is pure overhead.
        for row in items:
            for record in worker(row) or ():
                yield record
        return
    for records in thread_per_row_in_shard(
        items, worker=worker,
        fetch_concurrency=fetch_concurrency,
        thread_name_prefix="exp266-fetch",
    ):
        yield from records
