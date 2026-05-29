# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema vocabularies for the CSR protein-data substrate.

Two re-exports from marinfold's v1 vocab — the project's canonical source
of truth for these chemistry constants:

* ``AMINO_ACIDS`` — the 20 canonical residue 3-letter codes. Anything
  else parses to ``"UNK"`` in :mod:`parse`.
* ``ATOM_NAMES`` — the ~37 in-vocab heavy-atom names. Indexes the CSR
  ``atom_name_id: uint8`` column, so changes here would break stored
  CSR parquet files.

Intentionally *not* re-exporting anything doc-format-specific (no
contact/distance/think tokens, no per-format control vocab). The CSR
substrate is doc-format-agnostic — those tokens belong with the
document-format experiment that emits them, not with the substrate.
"""

from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    AMINO_ACIDS,
    ATOM_NAMES,
)

__all__ = ["AMINO_ACIDS", "ATOM_NAMES"]
