# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The canonical structure-file contract: round-trip and its refusals.

The contract is the whole interface between a predictor and the scorer, so the
things it promises to reject matter as much as the things it round-trips: a
silently dropped out-of-vocabulary atom or a silently merged duplicate would
show up downstream as a coverage number that is quietly wrong.
"""

import numpy as np
import pytest

from canonical_pdb import (
    atom_keys,
    build_atom_array,
    element_of,
    read_structure,
    write_structure,
)

# (res_id, res_name, atom_name, x, y, z, uncertainty)
_ATOMS = [
    (1, "ALA", "CB", 3.5, 2.0, 3.0, 0.0),
    (1, "ALA", "N", 1.0, 2.0, 3.0, 0.0),
    (1, "ALA", "CA", 2.0, 2.0, 3.0, 0.5),
    (1, "ALA", "C", 3.0, 2.0, 3.0, 0.0),
    (7, "GLY", "CA", 5.0, 2.0, 3.0, 2.887),
]


def test_round_trip_preserves_identity_coordinates_and_uncertainty(tmp_path):
    array = build_atom_array(_ATOMS)
    path = tmp_path / "pred.pdb"
    write_structure(array, path)
    back = read_structure(path)

    assert atom_keys(back) == atom_keys(array)
    assert np.allclose(back.coord, array.coord, atol=1e-3)
    assert np.allclose(back.b_factor, array.b_factor, atol=1e-2)
    assert set(np.unique(back.chain_id)) == {"A"}


def test_atoms_are_emitted_backbone_first_within_a_residue():
    array = build_atom_array(_ATOMS)
    assert atom_keys(array) == [
        (1, "N"),
        (1, "CA"),
        (1, "C"),
        (1, "CB"),
        (7, "CA"),
    ]


def test_duplicate_atom_is_rejected():
    with pytest.raises(ValueError, match="duplicate atom"):
        build_atom_array(_ATOMS + [(1, "ALA", "CA", 9.0, 9.0, 9.0, 0.0)])


def test_out_of_vocabulary_atom_is_rejected():
    with pytest.raises(ValueError, match="outside the vocabulary"):
        build_atom_array([(1, "ALA", "HB1", 0.0, 0.0, 0.0, 0.0)])


def test_non_canonical_residue_is_rejected():
    with pytest.raises(ValueError, match="not canonical"):
        build_atom_array([(1, "MSE", "CA", 0.0, 0.0, 0.0, 0.0)])


def test_reader_rejects_a_second_chain(tmp_path):
    array = build_atom_array(_ATOMS)
    path = tmp_path / "two_chains.pdb"
    write_structure(array, path)
    text = path.read_text().splitlines(keepends=True)
    # Duplicate the first ATOM record onto chain B.
    text.insert(1, text[0][:21] + "B" + text[0][22:])
    path.write_text("".join(text))
    with pytest.raises(ValueError, match="single chain"):
        read_structure(path)


def test_element_is_derived_from_the_atom_name():
    assert element_of("CA") == "C"
    assert element_of("ND1") == "N"
    assert element_of("OXT") == "O"
    assert element_of("SD") == "S"
    with pytest.raises(ValueError):
        element_of("XX")
