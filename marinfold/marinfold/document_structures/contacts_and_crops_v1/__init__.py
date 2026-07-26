# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-and-crops-v1 document structure.

A coordinate-bearing format that fits the **8192-token** context of the
contacts formats (vs contacts-and-coordinates-v1's 32768). It emits
contacts-v1's sequence section and ``<contact>`` statements verbatim, then
a bounded **two-pass coordinate section** (see ``SPEC.md``):

- **Pass 1 (coarse boxes):** every atom, sampled with replacement, gets a
  cheap ``<pX> <ATOM> <xyz-HHH> <xyz-TTT>`` mention placing it in its 10 Å
  box (budget-truncated for large chains).
- **Pass 2 (crops):** a small reserved budget reveals 0.1 Å detail inside a
  handful of selected boxes, each opened by a ``<crop> <xyz-HHH> <xyz-TTT>``
  header, with progressive refinement of re-shown boxes.

Positions live in a random rotated + translated frame inside a 1000 Å cube
(free data augmentation — no physical distance changes). Only two new tokens
(``<contacts-and-crops-v1>``, ``<crop>``); the ``<xyz-DDD>`` vocab and frame
are shared with ccoord, so the format warm-starts from a contacts-v1/ccoord
checkpoint.

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
    CROP_TOKEN,
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
    "CROP_TOKEN",
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
