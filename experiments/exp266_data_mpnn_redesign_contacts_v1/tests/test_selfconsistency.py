# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Metrics for the refold check. A wrong RMSD here would silently pass or fail
every design, so the properties are pinned rather than eyeballed."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from selfconsistency import kabsch_rmsd, tm_score  # noqa: E402


def _helix(n: int = 60) -> np.ndarray:
    """A CA trace with realistic 1.5 A rise / 100 degree twist."""
    t = np.arange(n)
    return np.stack([2.3 * np.cos(np.deg2rad(100) * t),
                     2.3 * np.sin(np.deg2rad(100) * t),
                     1.5 * t], axis=1)


def test_identical_structures_score_perfectly() -> None:
    p = _helix()
    assert kabsch_rmsd(p, p.copy())[0] == pytest.approx(0.0, abs=1e-9)
    assert tm_score(p, p.copy()) == pytest.approx(1.0, abs=1e-9)


def test_rigid_motion_is_removed() -> None:
    """Translation + rotation must not register as structural difference."""
    p = _helix()
    theta = 0.7
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = p @ rot.T + np.array([13.0, -4.0, 7.5])
    assert kabsch_rmsd(moved, p)[0] == pytest.approx(0.0, abs=1e-9)
    assert tm_score(moved, p) == pytest.approx(1.0, abs=1e-9)


def test_mirror_image_is_not_a_match() -> None:
    """The reflection guard: a mirrored fold must not superpose onto the original.

    Without the det() correction the SVD can return an improper rotation, which
    would score a mirror image as a perfect match and pass every left-handed
    decoy as designable.
    """
    p = _helix()
    mirrored = p * np.array([1.0, 1.0, -1.0])
    assert kabsch_rmsd(mirrored, p)[0] > 2.0
    assert tm_score(mirrored, p) < 0.5


def test_noise_degrades_scores_monotonically() -> None:
    rng = np.random.default_rng(0)
    p = _helix()
    prev_rmsd, prev_tm = -1.0, 2.0
    for sigma in (0.1, 0.5, 1.5, 4.0):
        q = p + rng.normal(0, sigma, p.shape)
        r = kabsch_rmsd(q, p)[0]
        t = tm_score(q, p)
        assert r > prev_rmsd and t < prev_tm
        prev_rmsd, prev_tm = r, t


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        kabsch_rmsd(_helix(60), _helix(50))
