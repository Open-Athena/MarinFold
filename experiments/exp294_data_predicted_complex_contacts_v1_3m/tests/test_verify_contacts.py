# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The independent distance measure must be independent, and ignore hydrogens."""

import sys
from pathlib import Path

import gemmi
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from verify_contacts import (  # noqa: E402
    LONG_SIDE_CHAINS,
    MIN_SEPARATION_FACTOR,
    ROTAMER_REACH_CEILING,
    _min_heavy_atom_distance,
    _residue_index,
)


def _residue(name: str, atoms: list[tuple[str, str, tuple[float, float, float]]]):
    residue = gemmi.Residue()
    residue.name = name
    residue.seqid = gemmi.SeqId(1, " ")
    for atom_name, element, (x, y, z) in atoms:
        atom = gemmi.Atom()
        atom.name = atom_name
        atom.element = gemmi.Element(element)
        atom.pos = gemmi.Position(x, y, z)
        residue.add_atom(atom)
    return residue


def test_distance_is_the_closest_heavy_atom_pair_not_the_first() -> None:
    a = _residue("ALA", [("CA", "C", (0, 0, 0)), ("CB", "C", (5, 0, 0))])
    b = _residue("SER", [("CA", "C", (20, 0, 0)), ("OG", "O", (9, 0, 0))])
    assert _min_heavy_atom_distance(a, b) == pytest.approx(4.0)


def test_hydrogens_are_ignored() -> None:
    """Deposits vary on whether hydrogens are modelled at all.

    Counting them would make the same interface measure differently depending on
    the deposit, which is exactly the kind of inconsistency this check exists to
    rule out.
    """
    a = _residue("ALA", [("CA", "C", (0, 0, 0))])
    near_h = _residue("ALA", [("CA", "C", (10, 0, 0)), ("HB1", "H", (1, 0, 0))])
    assert _min_heavy_atom_distance(a, near_h) == pytest.approx(10.0)


def test_residue_index_keys_on_chain_and_seqid() -> None:
    structure = gemmi.Structure()
    model = gemmi.Model("1")
    for chain_name in ("R", "L"):
        chain = gemmi.Chain(chain_name)
        chain.add_residue(_residue("GLY", [("CA", "C", (0, 0, 0))]))
        model.add_chain(chain)
    structure.add_model(model)
    index = _residue_index(structure)
    # Same seqid on two chains must not collide -- that would silently measure
    # the wrong pair for every inter-chain contact.
    assert set(index) == {("R", 1), ("L", 1)}


def test_thresholds_encode_the_rotamer_argument() -> None:
    """The bound comes from side-chain reach, not from what the data happened to do."""
    assert ROTAMER_REACH_CEILING == pytest.approx(14.0)
    assert MIN_SEPARATION_FACTOR >= 3.0
    assert {"ARG", "LYS", "GLU"} <= LONG_SIDE_CHAINS
