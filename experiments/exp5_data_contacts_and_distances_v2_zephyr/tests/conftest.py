# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures shared across exp5's test suite.

Two path adjustments:

* Add the experiment dir itself to ``sys.path`` so ``import parse`` /
  ``import generate`` / ``import vocab`` work — experiments aren't
  importable packages.
* Add exp34's directory so the byte-identity test can import its
  parse + generate as the reference oracle.

Provides a ``synthetic_cif`` fixture: a realistic-ish polymer mmCIF
text (~50 residues) built by converting a synthetic PDB via gemmi —
identical generation logic to exp34's ``_make_long_pdb`` test helper,
so cross-experiment byte-identity checks are reproducible.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXP_DIR = Path(__file__).resolve().parents[1]
_EXP34_DIR = _EXP_DIR.parent / "exp34_document_structures_contacts_and_distances_v2"

# Order matters: exp5's own modules first so ``import parse`` resolves to
# exp5's parse, not exp34's. The exp34 oracle imports under the
# ``exp34_*`` aliases in test_byte_identity.py via a sys.path swap.
for p in (_EXP_DIR, _EXP34_DIR):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)


def _make_long_pdb(num_residues: int) -> str:
    """Build a synthetic poly-ALA PDB string with realistic backbone+CB atoms.

    Helical-ish coordinates with monotonically increasing z so CB-CB
    distances span the eligibility cutoff and we exercise all three
    contact modes (short / medium / long range).
    """
    lines = [f"HEADER    SYNTHETIC                              01-JAN-26   SYN"]
    serial = 1
    for i in range(1, num_residues + 1):
        z = i * 3.0
        atoms = [
            ("N",  6.0, 0.0, z + 0.0),
            ("CA", 6.0, 1.5, z + 0.5),
            ("C",  6.0, 3.0, z + 0.0),
            ("O",  6.0, 3.0, z - 1.2),
            ("CB", 4.5, 1.5, z + 0.5),
        ]
        for name, x, y, zc in atoms:
            element = name[0]
            lines.append(
                f"ATOM  {serial:5d}  {name:<3s} ALA A{i:4d}    "
                f"{x:8.3f}{y:8.3f}{zc:8.3f}  1.00 90.00          {element:>2s}"
            )
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="session")
def synthetic_cif() -> str:
    """Realistic-ish ~50-residue polymer mmCIF text (built once per session)."""
    import gemmi

    pdb = _make_long_pdb(50)
    st = gemmi.read_pdb_string(pdb)
    st.setup_entities()
    return st.make_mmcif_document().as_string()


@pytest.fixture
def small_cif() -> str:
    """A 5-residue cif for parse-shape assertions."""
    import gemmi

    pdb = _make_long_pdb(5)
    st = gemmi.read_pdb_string(pdb)
    st.setup_entities()
    return st.make_mmcif_document().as_string()
