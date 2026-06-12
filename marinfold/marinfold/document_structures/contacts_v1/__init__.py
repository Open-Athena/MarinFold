# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""contacts-v1 document structure.

Serialize each protein as a *sequence section* (one ``<pX> <AA>``
statement per residue, plus ``<n-term>`` / ``<c-term>`` markers, in random
order) followed by a *structure section* of ``<contact> <pX> <pY>``
statements for the strongest pyconfind side-chain contacts (as many as
fill the context-length budget), listed in random order. Residues are
numbered from a random n-terminal index that wraps around 2000 position
tokens. See ``SPEC.md`` in this directory.

Public surface:

- :data:`NAME`, :data:`CONTEXT_LENGTH`, :data:`NUM_POSITION_INDICES`,
  :func:`all_domain_tokens` — from :mod:`.vocab`.
- :func:`analyze_structure`, :class:`AnalyzedStructure`,
  :class:`ResidueInfo`, :class:`RawContact`, :func:`residues_from_sequence`
  — from :mod:`.parse` (the pyconfind layer; ``residues_from_sequence`` is
  the structure-free residue builder for the sequence-only path).
- :func:`generate_document`, :func:`generate_documents`,
  :func:`generate_sequence_only_document`, :func:`build_document`,
  :class:`GenerationConfig`, :class:`GenerationResult`,
  :class:`EmittedContact` — from :mod:`.generate`. ``generate_document`` is
  the single-structure entry point a zephyr data job calls per input;
  ``generate_sequence_only_document`` is its structure-free analogue for
  sequence databases (e.g. UniRef50; see exp64).
"""

from .generate import (
    EmittedContact,
    GenerationConfig,
    GenerationResult,
    build_document,
    generate_document,
    generate_documents,
    generate_sequence_only_document,
)
from .parse import (
    DEFAULT_CIF_COLUMN,
    DEFAULT_ID_COLUMN,
    AnalyzedStructure,
    RawContact,
    ResidueInfo,
    analyze_structure,
    iter_analyzed_structures,
    iter_parquet_analyzed_structures,
    residues_from_sequence,
)
from .vocab import (
    CONTEXT_LENGTH,
    NAME,
    NUM_POSITION_INDICES,
    SEQUENCE_ONLY_DOC_TYPE_TOKEN,
    all_domain_tokens,
)

__all__ = [
    "CONTEXT_LENGTH",
    "DEFAULT_CIF_COLUMN",
    "DEFAULT_ID_COLUMN",
    "NAME",
    "NUM_POSITION_INDICES",
    "SEQUENCE_ONLY_DOC_TYPE_TOKEN",
    "AnalyzedStructure",
    "EmittedContact",
    "GenerationConfig",
    "GenerationResult",
    "RawContact",
    "ResidueInfo",
    "all_domain_tokens",
    "analyze_structure",
    "build_document",
    "generate_document",
    "generate_documents",
    "generate_sequence_only_document",
    "iter_analyzed_structures",
    "iter_parquet_analyzed_structures",
    "residues_from_sequence",
]
