"""Guard against false acceptance caused by alignment and geometry mistakes."""

import numpy as np
import pytest

from quality import aligned_rmsd, ca_geometry, confidence_percent


def test_alignment_removes_translation_and_rotation() -> None:
    points = np.random.default_rng(278).normal(size=(30, 3))
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    assert aligned_rmsd(points, points @ rotation + [2, 6, -3]) < 1e-12


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
