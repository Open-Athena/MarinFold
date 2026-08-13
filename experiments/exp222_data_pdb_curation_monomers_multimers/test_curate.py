# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pieces of exp222's curation that are pure logic.

The geometry filters need real coordinates and are exercised end to end by
``validate.py`` against the generated corpora; what is worth pinning here is
the fiddly bookkeeping around them, where a silent mistake would mis-key
every cluster id or mis-size every assembly without anything failing.

Run with ``uv run pytest test_curate.py``.
"""

import gemmi
import pytest

from curate import (
    MAX_ADJACENT_CA_DISTANCE,
    _has_adjacent_ca_break,
    assembly_subchain_entities,
    load_clusters,
)
from scan_metadata import _count_operators


def _residue(num: int, ca_x: float | None) -> gemmi.Residue:
    residue = gemmi.Residue()
    residue.name = "ALA"
    residue.seqid = gemmi.SeqId(num, " ")
    if ca_x is not None:
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(ca_x, 0.0, 0.0)
        residue.add_atom(atom)
    return residue


class TestAdjacentCaBreak:
    def test_contiguous_chain_is_fine(self):
        residues = [_residue(i + 1, 3.8 * i) for i in range(10)]
        assert not _has_adjacent_ca_break(residues)

    def test_break_between_consecutive_numbers_is_caught(self):
        residues = [_residue(1, 0.0), _residue(2, MAX_ADJACENT_CA_DISTANCE + 1.0)]
        assert _has_adjacent_ca_break(residues)

    def test_numbering_gap_is_not_a_break(self):
        """An unresolved stretch leaves a jump in space that is expected.

        AF3's filter is about *consecutively numbered* residues; residues 1
        and 40 being far apart says nothing about model quality, and treating
        it as a break would throw away most crystal structures.
        """
        residues = [_residue(1, 0.0), _residue(40, 500.0)]
        assert not _has_adjacent_ca_break(residues)

    def test_residue_without_ca_resets_the_comparison(self):
        residues = [_residue(1, 0.0), _residue(2, None), _residue(3, 500.0)]
        assert not _has_adjacent_ca_break(residues)


class TestOperatorExpressions:
    @pytest.mark.parametrize("expression,expected", [
        ("1", 1),
        ("1,2,3", 3),
        ("1-60", 60),
        ("(1-60)", 60),
        ("(1-60)(61-88)", 60 * 28),
        ("1-3,5", 4),
        ("P", 1),
        ("", 1),
    ])
    def test_count(self, expression, expected):
        assert _count_operators(expression) == expected


class TestAssemblySubchainEntities:
    def _assembly(self, subchains: list[str]) -> gemmi.Structure:
        structure = gemmi.Structure()
        model = gemmi.Model("1")
        for name in subchains:
            chain = gemmi.Chain(name)
            residue = _residue(1, 0.0)
            residue.subchain = name
            chain.add_residue(residue)
            model.add_chain(chain)
        structure.add_model(model)
        return structure

    def test_inverts_add_number_naming(self):
        asu = {"A": "1", "B": "2"}
        assembly = self._assembly(["A1", "A2", "B1"])
        assert assembly_subchain_entities(asu, assembly) == {
            "A1": "1", "A2": "1", "B1": "2",
        }

    def test_longest_prefix_wins(self):
        """``A`` and ``A1`` can both be real asym ids.

        Stripping trailing digits would map the copy ``A11`` to ``A``, i.e.
        the wrong entity and the wrong RCSB cluster. The longest matching
        prefix resolves it.
        """
        asu = {"A": "1", "A1": "2"}
        assembly = self._assembly(["A11"])
        assert assembly_subchain_entities(asu, assembly) == {"A11": "2"}

    def test_non_protein_copies_are_left_out(self):
        asu = {"A": "1"}
        assembly = self._assembly(["A1", "C1"])
        assert assembly_subchain_entities(asu, assembly) == {"A1": "1"}


def test_load_clusters(tmp_path):
    path = tmp_path / "clusters.txt"
    path.write_text("1ABC_1 1ABC_2 2XYZ_1\n3DEF_1\n")
    clusters = load_clusters(str(path))
    assert clusters == {"1ABC_1": 0, "1ABC_2": 0, "2XYZ_1": 0, "3DEF_1": 1}
    # Same cluster for the two entities that share a line, different for the
    # entry on its own -- the id is the line number, stable across lookups.
    assert clusters["1ABC_1"] == clusters["2XYZ_1"] != clusters["3DEF_1"]
