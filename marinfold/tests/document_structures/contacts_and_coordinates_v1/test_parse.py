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
    _chain_key,
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


def test_chain_key_normalizes_the_blank_author_chain():
    # gemmi renders a blank author chain id as "", pyconfind as "_". Keying
    # the coordinate walk on one and looking it up with the other joined
    # nothing and silently produced a residue list with no coordinates.
    assert _chain_key("") == "_"
    assert _chain_key(" ") == "_"
    assert _chain_key("_") == "_"
    # A named chain is untouched.
    assert _chain_key("A") == "A"
    assert _chain_key("AAA") == "AAA"


def test_atoms_by_residue_key_finds_atoms_on_a_blank_chain(tmp_path):
    # CASP target files (and other hand-built PDBs) leave the chain id blank.
    structure = gemmi.read_structure(str(_1QYS))
    structure.setup_entities()
    for chain in structure[0]:
        chain.name = ""
    blank_pdb = tmp_path / "blank_chain.pdb"
    structure.write_pdb(str(blank_pdb))

    reloaded = gemmi.read_structure(str(blank_pdb))
    keyed = _atoms_by_residue_key(reloaded)
    assert keyed, "blank-chain structure produced no atoms"
    # Every key uses the normalized blank spelling, which is what the
    # pyconfind-derived residue list will look the atoms up with.
    assert {chain for chain, _ in keyed} == {"_"}
