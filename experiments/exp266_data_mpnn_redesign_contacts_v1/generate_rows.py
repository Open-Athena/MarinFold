# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage B per-row generation — one staged backbone -> 8 redesigned documents.

The *worker* half, import-light (no zephyr, no iris) so it unit-tests locally
and runs unchanged inside the CoreWeave task image.

Input is a **staged backbone row** (`stage_rows.py` output), not an mmCIF URI:
CoreWeave task pods have no GCP credentials, so the structures arrive as data
in CoreWeave object storage. `backbone.decode_backbone` rebuilds exactly the
gemmi structure pyconfind would have seen — pinned byte-identical by
`tests/test_backbone.py::test_staged_backbone_round_trip_is_byte_identical`.

There is no per-row I/O left here at all: the caller hands over a shard that
is already in memory, and the row carries its own coordinates. So the cost is
pure CPU/GPU and the right concurrency knob is the worker's core count, not a
fetch-latency ratio.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Iterable, Iterator
from typing import Any

from backbone import backbone_coords_from_row, decode_backbone, relabel_sequence
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    generate_document,
)
from marinfold.document_structures.contacts_v1.vocab import CONTEXT_LENGTH, NAME
from redesign import DESIGN_TEMPERATURES, BackboneEntry, design_batch

# Carried verbatim onto every output document so a redesigned row can be traced
# back to the native document it came from. `entry_id` comes from contacts-v1's
# own `metadata_row` (same value), so it is not repeated.
#
# `global_plddt` is not passed through either: `metadata_row` recomputes it from
# CA B-factors, which `decode_backbone` restores from the staged `ca_plddt`, so
# the redesigned row carries the parent AFDB pLDDT unchanged. That is the right
# value — it describes confidence in the *backbone* we reused — but note it says
# nothing about whether the designed sequence folds there.
PASSTHROUGH_COLUMNS: tuple[str, ...] = (
    "struct_cluster_id",
    "seq_cluster_id",
    "split",
    "round",
    "native_contacts_emitted",
    "native_sha1",
)


@functools.cache
def _load_rotamer_library() -> Any | None:
    """Parse pyconfind's Dunbrack rotamer library once per process.

    Tens of seconds per parse; doing it per shard instead of per worker is the
    mistake exp53 fixed in daa18e1 and the pipeline skill calls out as a
    41-hour cluster bill.
    """
    try:
        from pyconfind import cached_rotamer_library, load_library

        return load_library(cached_rotamer_library())
    except Exception as exc:  # pragma: no cover - best effort preload
        warnings.warn(f"rotamer preload failed ({exc}); per-call load", stacklevel=2)
        return None


def documents_for_designs(
    row: dict,
    designs: list,
    *,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    rotamer_library: Any = None,
    structure_name: str = NAME,
) -> list[dict[str, Any]]:
    """Turn one staged row plus its designs into contacts-v1 document rows.

    Split out from the design step so the GPU path can batch designs across
    many backbones and still generate documents one backbone at a time.
    """
    backbone = decode_backbone(row)
    entry_id = row["entry_id"]

    out: list[dict[str, Any]] = []
    for design in designs:
        relabelled = relabel_sequence(backbone, design.sequence)
        # Seed on (entry_id, design_index): each design gets its own statement
        # order and index offset, rather than eight documents differing only in
        # their amino acids.
        result = generate_document(
            relabelled,
            entry_id=f"{entry_id}#{design.design_index}",
            context_length=context_length,
            config=config,
            rotamer_library=rotamer_library,
        )
        if result is None:
            # Designed-in filter: contacts-v1 declined to serialize this chain.
            # The same predicate as the native corpus, so a natively
            # serializable backbone can still fail here if the redesign changed
            # the contact budget.
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
    return out


def documents_for_row(
    row: dict,
    *,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    rotamer_library: Any = None,
    structure_name: str = NAME,
    device: str = "cpu",
    temperatures: tuple[float, ...] = DESIGN_TEMPERATURES,
) -> list[dict[str, Any]] | None:
    """Redesign one staged backbone and emit one document per design.

    The single-row path — used by the CPU fallback and by the tests. The
    CoreWeave GPU worker batches `design_batch` across many rows instead and
    calls `documents_for_designs` directly.
    """
    designs = design_batch(
        [BackboneEntry(row["entry_id"], row["sequence"], backbone_coords_from_row(row))],
        device=device,
        temperatures=temperatures,
    )
    out = documents_for_designs(
        row, designs,
        context_length=context_length, config=config,
        rotamer_library=rotamer_library, structure_name=structure_name,
    )
    return out or None


def generate_shard(
    items: Iterable[dict],
    shard_info: Any = None,
    *,
    context_length: int = CONTEXT_LENGTH,
    config: GenerationConfig = GenerationConfig(),
    structure_name: str = NAME,
    device: str = "cpu",
    temperatures: tuple[float, ...] = DESIGN_TEMPERATURES,
) -> Iterator[dict[str, Any]]:
    """The ``map_shard`` body for the CPU fallback: one row -> up to 8 rows.

    No thread pool: there is no per-row I/O to overlap once the backbones are
    staged, and a `--worker-cpu 1` worker has nothing to gain from threads that
    only contend for its single core.
    """
    rotamer_library = _load_rotamer_library()
    for row in items:
        records = documents_for_row(
            row,
            context_length=context_length, config=config,
            rotamer_library=rotamer_library, structure_name=structure_name,
            device=device, temperatures=temperatures,
        )
        if records:
            yield from records
