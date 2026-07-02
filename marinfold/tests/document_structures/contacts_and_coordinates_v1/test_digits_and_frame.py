# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Digit extraction + coordinate-frame tests (pure, no pyconfind)."""

import math
import random

import numpy as np
import pytest

from marinfold.document_structures.contacts_and_coordinates_v1.generate import (
    GenerationConfig,
    _apply_frame,
    _coordinate_digits,
    _random_rotation_matrix,
    _xyz_tokens,
)


def test_spec_worked_example_digits():
    # SPEC → Digit extraction, position (205.3, 180.2, 5.7).
    assert _coordinate_digits(205.3) == (2, 0, 5, 3)
    assert _coordinate_digits(180.2) == (1, 8, 0, 2)
    assert _coordinate_digits(5.7) == (0, 0, 5, 7)
    assert _xyz_tokens(205.3, 180.2, 5.7, 4) == [
        "<xyz-210>", "<xyz-080>", "<xyz-505>", "<xyz-327>",
    ]


def test_digit_extraction_float_bug_value():
    # 180.2 / 0.1 == 1801.9999999999998 in IEEE-754; the integer-space
    # extraction must still read the tenths digit as 2, not 1.
    assert _coordinate_digits(180.2)[3] == 2
    # A few more values whose *10 lands just under an integer.
    assert _coordinate_digits(0.3)[3] == 3
    assert _coordinate_digits(70.7)[3] == 7


def test_digit_extraction_clamps():
    assert _coordinate_digits(-5.0) == (0, 0, 0, 0)
    assert _coordinate_digits(10_000.0) == (9, 9, 9, 9)  # clamped to 999.9
    assert _coordinate_digits(999.9) == (9, 9, 9, 9)
    assert _coordinate_digits(0.0) == (0, 0, 0, 0)


def test_xyz_tokens_depth_prefix():
    toks = _xyz_tokens(205.3, 180.2, 5.7, 2)
    assert toks == ["<xyz-210>", "<xyz-080>"]  # hundreds + tens only
    assert _xyz_tokens(205.3, 180.2, 5.7, 1) == ["<xyz-210>"]


def test_random_rotation_is_orthonormal():
    rng = random.Random(1234)
    for _ in range(20):
        matrix, quat = _random_rotation_matrix(rng)
        # Orthonormal + proper rotation (det +1).
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
        assert math.isclose(float(np.linalg.det(matrix)), 1.0, abs_tol=1e-9)
        assert math.isclose(sum(c * c for c in quat), 1.0, rel_tol=1e-9)


def test_frame_preserves_pairwise_distances():
    rng = random.Random(7)
    coords = np.array([
        [12.0, -4.0, 30.0], [15.0, -1.0, 33.0], [-8.0, 20.0, 5.0],
        [40.0, 40.0, 40.0], [0.0, 0.0, 0.0],
    ])
    framed = _apply_frame(rng, coords, GenerationConfig())
    assert framed is not None
    transformed, _quat, _translation = framed
    # Rotation + translation is rigid: every pairwise distance is unchanged.
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d0 = float(np.linalg.norm(coords[i] - coords[j]))
            d1 = float(np.linalg.norm(transformed[i] - transformed[j]))
            assert math.isclose(d0, d1, rel_tol=1e-9, abs_tol=1e-9)


def test_frame_places_inside_cube():
    rng = random.Random(3)
    config = GenerationConfig()
    coords = np.random.default_rng(0).normal(scale=20.0, size=(200, 3))
    framed = _apply_frame(rng, coords, config)
    assert framed is not None
    transformed, _quat, _translation = framed
    assert transformed.min() >= config.cube_margin - 1e-6
    assert transformed.max() <= config.cube_size - config.cube_margin + 1e-6


def test_frame_skips_structure_too_large():
    rng = random.Random(0)
    config = GenerationConfig()
    # A dense cloud filling a ~990 Å cube: its bbox span exceeds
    # cube_size - 2*margin = 980 Å on every axis under any rotation
    # (rotation only grows an axis-aligned bbox here), so it is unplaceable.
    coords = np.random.default_rng(0).uniform(-495.0, 495.0, size=(1000, 3))
    assert _apply_frame(rng, coords, config) is None


def test_depth_sigmas_match_spec():
    sigmas = GenerationConfig().depth_sigmas()
    assert sigmas == (25.0, 2.5, 0.25, 0.1)
