# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-coordinates-v1 document structure.

Extends contacts-v1 (see ``SPEC.md`` in this directory): the identical
sequence section and ``<contact>`` statements, plus a **coordinate
section** — a stream of ``<pX> <ATOM> <xyz-...>+`` mention events that
reveal 3D atom positions coarse-to-fine (hundreds → tenths of an Å) with
calibrated per-mention noise, expressed in a random rotated + translated
frame inside a 1000 Å cube.

Public surface:

- :data:`NAME`, :data:`CONTEXT_LENGTH`, :data:`NUM_POSITION_INDICES`,
  :func:`all_domain_tokens`, :func:`xyz_token` — from :mod:`.vocab`.
- :func:`analyze_coordinates`, :class:`AnalyzedCoordStructure`,
  :func:`iter_coordinate_structures` — from :mod:`.parse` (residues +
  pyconfind contacts + per-atom coordinates).
- :func:`generate_document`, :func:`generate_documents`,
  :func:`build_document`, :class:`GenerationConfig`,
  :class:`GenerationResult` — from :mod:`.generate`. ``generate_document``
  is the single-structure entry point a zephyr data job calls per input.
"""

from .generate import (
    GenerationConfig,
    GenerationResult,
    build_document,
    generate_document,
    generate_documents,
)
from .parse import (
    DEFAULT_CIF_COLUMN,
    DEFAULT_ID_COLUMN,
    AnalyzedCoordStructure,
    AtomCoord,
    analyze_coordinates,
    iter_coordinate_structures,
)
from .vocab import (
    CONTEXT_LENGTH,
    DOC_TYPE_TOKEN,
    NAME,
    NUM_POSITION_INDICES,
    NUM_XYZ_TOKENS,
    all_domain_tokens,
    atom_token,
    position_token,
    xyz_token,
    xyz_token_for_digits,
)

__all__ = [
    "CONTEXT_LENGTH",
    "DEFAULT_CIF_COLUMN",
    "DEFAULT_ID_COLUMN",
    "DOC_TYPE_TOKEN",
    "NAME",
    "NUM_POSITION_INDICES",
    "NUM_XYZ_TOKENS",
    "AnalyzedCoordStructure",
    "AtomCoord",
    "GenerationConfig",
    "GenerationResult",
    "all_domain_tokens",
    "analyze_coordinates",
    "atom_token",
    "build_document",
    "generate_document",
    "generate_documents",
    "iter_coordinate_structures",
    "position_token",
    "xyz_token",
    "xyz_token_for_digits",
]
