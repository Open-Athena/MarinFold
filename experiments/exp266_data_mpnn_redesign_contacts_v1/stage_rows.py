# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage A2 per-row worker — one AFDB mmCIF -> one compact backbone row.

Runs on the **GCP** Iris pool, which is the only place with credentials for
AFDB's requester-pays GCS bucket. CoreWeave task pods carry only CoreWeave S3
credentials (`iris-task-env`: `AWS_*` / `CW_*` / `FSSPEC_S3`, no GCP), so the
backbones have to cross clouds as data rather than as a per-row fetch.

What crosses is small because we only need backbones: N/CA/C/O plus the
sequence and per-residue pLDDT is ~14 KB per protein against ~180 KB of
all-atom mmCIF. For 3.96 M structures that is ~55 GB staged once instead of
~700 GB fetched repeatedly — and the artifact is reusable by any future
backbone-based experiment, the same argument #139 makes for having saved its
raw pyconfind contacts.

This stage is genuinely I/O-bound (a ~30-80 ms GCS GET against a few ms of
parse-and-encode), so it is the textbook case for
``thread_per_row_in_shard`` at the default fetch concurrency — unlike the
document stage next door, where seconds of CPU per row dominate.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import partial
from typing import Any

import gemmi

from backbone import encode_backbone, prepare_structure, strip_to_backbone
from marinfold.document_structures.io import read_object_bytes, thread_per_row_in_shard

# Provenance carried from the Stage-A manifest onto every backbone row.
PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "seq_len",
    "struct_cluster_id",
    "seq_cluster_id",
    "split",
    "round",
    "native_contacts_emitted",
    "native_sha1",
)

DEFAULT_CIF_URI_COLUMN = "gcs_uri"


def _structure_from_cif(data: str | bytes, *, entry_id: str) -> gemmi.Structure:
    # `read_structure_string` takes str or bytes; the name is set explicitly
    # because the parsed entry id is not the AFDB accession we key on.
    structure = gemmi.read_structure_string(data, format=gemmi.CoorFormat.Mmcif)
    structure.name = entry_id
    return structure


def stage_row(
    row: dict,
    *,
    cif_uri_column: str = DEFAULT_CIF_URI_COLUMN,
    cif_text_column: str | None = None,
) -> dict[str, Any] | None:
    """Fetch one structure and return its staged backbone row.

    Returns ``None`` only for designed-in filter predicates — a structure that
    is not the single canonical-residue protein chain the redesign path
    requires. Everything else raises: a silently dropped or mis-encoded row
    would corrupt the corpus, and ``encode_backbone`` deliberately raises on
    non-contiguous residue numbering and on coordinates that are not exact at
    0.001 A rather than encoding them approximately.
    """
    if cif_text_column is not None:
        data: str | bytes = row[cif_text_column]
    else:
        data = read_object_bytes(row[cif_uri_column])

    entry_id = row["entry_id"]
    structure = strip_to_backbone(
        prepare_structure(_structure_from_cif(data, entry_id=entry_id))
    )

    try:
        staged = encode_backbone(structure)
    except ValueError as exc:
        message = str(exc)
        # The two designed-in filters. Anything else — a missing mainchain
        # atom, an inexact coordinate, non-contiguous numbering — is a real
        # surprise about the input and must surface as a crash.
        if "non-canonical residues" in message or "expected 1 chain" in message:
            return None
        raise

    staged["entry_id"] = entry_id
    for column in PASSTHROUGH_COLUMNS:
        if column in row:
            staged[column] = row[column]
    return staged


def stage_shard(
    items: Iterable[dict],
    shard_info: Any = None,
    *,
    cif_uri_column: str = DEFAULT_CIF_URI_COLUMN,
    cif_text_column: str | None = None,
    fetch_concurrency: int = 32,
) -> Iterator[dict[str, Any]]:
    """The ``map_shard`` body: one manifest row -> one backbone row."""
    worker = partial(
        stage_row,
        cif_uri_column=cif_uri_column,
        cif_text_column=cif_text_column,
    )
    if cif_text_column is not None:
        # Inline path: no I/O to overlap, so a thread pool is pure overhead.
        for row in items:
            staged = worker(row)
            if staged is not None:
                yield staged
        return
    yield from thread_per_row_in_shard(
        items, worker=worker,
        fetch_concurrency=fetch_concurrency,
        thread_name_prefix="exp266-stage",
    )
