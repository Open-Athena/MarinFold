# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures shared across exp42's test suite.

* Add the experiment dir itself to ``sys.path`` so ``import parse`` /
  ``import csr_store`` / ``import dataset`` / ``import vocab`` work
  (experiments aren't importable packages).
* Provide a ``synthetic_cif`` fixture: a realistic-ish polymer mmCIF
  text (~50 residues) built by converting a synthetic PDB via gemmi.
  Same generation logic as the exp5 / exp34 fixtures, so the CSR
  roundtrip tests use the exact same structures those tests do.

Intentionally NO cross-experiment imports — exp42 is doc-format-agnostic
and self-contained. Specific doc formats validate themselves in their
own experiments (e.g. exp5's byte-identity-vs-exp34 test for v2).
"""

import sys
from pathlib import Path

import pytest

_EXP_DIR = Path(__file__).resolve().parents[1]

if str(_EXP_DIR) not in sys.path:
    sys.path.insert(0, str(_EXP_DIR))


def _make_long_pdb(num_residues: int) -> str:
    """Build a synthetic poly-ALA PDB string with realistic backbone+CB atoms.

    Helical-ish coordinates with monotonically increasing z so CB-CB
    distances span the eligibility cutoff — exercises every residue +
    atom field the CSR schema carries.
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
