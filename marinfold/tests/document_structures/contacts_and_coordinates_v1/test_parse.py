# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Parse-layer tests that need only gemmi (no pyconfind).

The heavy pyconfind-backed path is covered in ``test_integration.py``; here
we exercise the pure-gemmi heavy-atom walk and its polymer filtering.
"""

from pathlib import Path

import gemmi

from marinfold.document_structures.contacts_and_coordinates_v1.parse import (
    _atoms_by_residue_key,
    _vocab_safe_atoms,
)

_1QYS = Path(__file__).parents[2] / "data" / "1QYS.cif"


def test_atoms_by_residue_key_excludes_waters_and_ligands():
    # 1QYS carries 7 waters (HOH) on the *same* author chain (A) as the
    # protein, numbered 107-113. A water's oxygen is named "O", which is in
    # the atom vocab — so a naive walk over every residue would add those
    # waters to the coordinate map and could shadow a same-numbered protein
    # residue. The polymer filter must drop them.
    structure = gemmi.read_structure(str(_1QYS))

    # What a naive all-residue walk would have keyed (protein + waters).
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
    # The polymer walk keeps every protein residue but none of the waters.
    assert polymer_keys == naive_keys - water_keys
    assert not (polymer_keys & water_keys)


def test_atoms_by_residue_key_keeps_modified_residues():
    # 1QYS models MSE (selenomethionine) as HETATM; it is a polymer residue,
    # so get_polymer() keeps it and it carries a CA. (A HETATM filter would
    # wrongly drop it — hence get_polymer(), not het_flag.)
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
