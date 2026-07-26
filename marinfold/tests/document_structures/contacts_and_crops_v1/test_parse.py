# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse-layer tests that need only gemmi (no pyconfind).

The heavy pyconfind-backed path is covered in ``test_integration.py``; here
we exercise the pure-gemmi heavy-atom walk and its polymer filtering. The
parse layer is reused verbatim from ccoord.
"""

from pathlib import Path

import gemmi

from marinfold.document_structures.contacts_and_crops_v1.parse import (
    _atoms_by_residue_key,
    _vocab_safe_atoms,
)

_1QYS = Path(__file__).parents[2] / "data" / "1QYS.cif"


def test_atoms_by_residue_key_excludes_waters_and_ligands():
    # 1QYS carries waters (HOH) on the same author chain (A) as the protein.
    # A water's oxygen is named "O", which is in the atom vocab — the polymer
    # filter must drop them so they don't shadow a same-numbered residue.
    structure = gemmi.read_structure(str(_1QYS))

    naive_keys = {
        (chain.name, res.seqid.num)
        for chain in structure[0]
        for res in chain
        if _vocab_safe_atoms(res)
    }
    water_keys = {
        (chain.name, res.seqid.num)
        for chain in structure[0]
        for res in chain
        if res.name in ("HOH", "WAT") and _vocab_safe_atoms(res)
    }
    assert water_keys, "expected 1QYS to contain vocab-eligible waters"

    polymer_keys = set(_atoms_by_residue_key(structure))
    assert polymer_keys == naive_keys - water_keys
    assert not (polymer_keys & water_keys)


def test_atoms_by_residue_key_keeps_modified_residues():
    # 1QYS models MSE (selenomethionine) as HETATM; it is a polymer residue,
    # so get_polymer() keeps it and it carries a CA.
    structure = gemmi.read_structure(str(_1QYS))
    atoms_by_key = _atoms_by_residue_key(structure)
    mse_keys = [
        (chain.name, res.seqid.num)
        for chain in structure[0]
        for res in chain
        if res.name == "MSE"
    ]
    assert mse_keys, "expected 1QYS to contain MSE residues"
    for key in mse_keys:
        assert key in atoms_by_key
        assert "CA" in {name for name, *_ in atoms_by_key[key]}
