"""Guard against false acceptance caused by alignment and geometry mistakes."""

from pathlib import Path

import gemmi
import numpy as np
import pytest

from quality import aligned_rmsd, backbone_geometry, ca_geometry, confidence_percent


def test_alignment_removes_translation_and_rotation() -> None:
    points = np.random.default_rng(278).normal(size=(30, 3))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert aligned_rmsd(points, points @ rotation + [2, 6, -3]) < 1e-12


def test_cis_proline_is_not_a_chain_break() -> None:
    structure = gemmi.read_structure(
        str(Path(__file__).parent / "data/cis-proline-fixture.pdb")
    )
    coordinates = np.asarray(
        [list(residue["CA"][0].pos) for residue in structure[0][0]]
    )
    assert ca_geometry(coordinates)["ca_chain_breaks"] == 1
    result = backbone_geometry(structure)
    assert result["ca_geometry_pass"]
    assert result["cis_proline_bonds"] == 1


def test_compressed_nonproline_remains_rejected() -> None:
    structure = gemmi.read_structure(
        str(Path(__file__).parent / "data/cis-proline-fixture.pdb")
    )
    structure[0][0][2].name = "ALA"
    assert not backbone_geometry(structure)["ca_geometry_pass"]


def test_broken_peptide_link_is_not_rescued_by_proline() -> None:
    structure = gemmi.read_structure(
        str(Path(__file__).parent / "data/cis-proline-fixture.pdb")
    )
    structure[0][0][1]["C"][0].pos.x += 4
    assert not backbone_geometry(structure)["ca_geometry_pass"]


def test_alignment_does_not_accept_mirror_image() -> None:
    points = np.random.default_rng(278).normal(size=(30, 3)) * 10
    assert aligned_rmsd(points, points * [-1, 1, 1]) > 2


def test_alignment_does_not_drop_missing_residues() -> None:
    with pytest.raises(ValueError, match="matching"):
        aligned_rmsd(np.ones((12, 3)), np.ones((11, 3)))


def test_chain_break_fails_geometry() -> None:
    points = np.column_stack([np.arange(10) * 3.8, np.zeros((10, 2))])
    assert ca_geometry(points)["ca_geometry_pass"]
    points[5:] += [10, 0, 0]
    assert not ca_geometry(points)["ca_geometry_pass"]


def test_nonlocal_overlap_is_not_ignored() -> None:
    points = np.array([[0, 0, 0], [3.8, 0, 0], [3.8, 3.8, 0], [0, 3.8, 0], [0, 0, 0]])
    result = ca_geometry(points)
    assert result["ca_chain_breaks"] == 0
    assert result["ca_clashes"] == 1
    assert not result["ca_geometry_pass"]


def test_esm_confidence_units_at_the_acceptance_boundary() -> None:
    assert confidence_percent(0.69) < 70
    assert confidence_percent(0.71) >= 70
    with pytest.raises(ValueError, match="normalized"):
        confidence_percent(90)
