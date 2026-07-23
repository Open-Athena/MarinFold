# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests that actually run pyconfind + gemmi on 1QYS.

Need ``pyconfind`` installed (``uv sync --extra contacts-v1``) and, on a
cold cache, network to download the Dunbrack rotamer library once — hence
the ``network`` marker. Skip with ``pytest -m 'not network'``.
"""

from pathlib import Path

import pytest

pytest.importorskip("pyconfind")

from marinfold.document_structures.contacts_and_crops_v1 import (  # noqa: E402
    analyze_coordinates,
    generate_document,
)

_1QYS = Path(__file__).parents[2] / "data" / "1QYS.cif"


@pytest.mark.network
def test_analyze_coordinates_aligns_atoms_to_residues():
    analyzed = analyze_coordinates(_1QYS)
    assert analyzed.entry_id == "1QYS"
    assert len(analyzed.residues) == 92
    assert set(analyzed.atoms_by_seq_index) == {r.seq_index for r in analyzed.residues}
    ca_count = 0
    for r in analyzed.residues:
        atoms = analyzed.atoms_by_seq_index[r.seq_index]
        names = [name for name, *_ in atoms]
        assert len(names) == len(set(names)), "duplicate atom name in a residue"
        if "CA" in names:
            ca_count += 1
        if r.resname == "GLY":
            assert "CB" not in names
    assert ca_count == 92


@pytest.mark.network
def test_generate_document_1qys_fits_8k_and_is_wellformed():
    result = generate_document(_1QYS)
    assert result is not None
    assert result.seq_len == 92
    # The whole point of the format: fits the 8k contacts-format context.
    assert result.num_tokens <= 8192
    assert result.num_tokens == len(result.document.split())
    toks = result.document.split()
    assert toks[0] == "<contacts-and-crops-v1>"
    assert toks[1] == "<begin_sequence>"
    assert toks[-1] == "<end>"
    assert "<begin_statements>" in toks
    # Both passes produced content.
    assert result.num_pass1_mentions > 0
    assert result.num_crops > 0
    assert "<crop>" in toks
    assert any(t.startswith("<xyz-") for t in toks)


@pytest.mark.network
def test_generate_document_deterministic():
    a = generate_document(_1QYS)
    b = generate_document(_1QYS)
    assert a is not None and b is not None
    assert a.document == b.document
