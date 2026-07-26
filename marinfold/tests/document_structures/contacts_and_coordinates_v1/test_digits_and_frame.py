# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Digit extraction + coordinate-frame tests (pure, no pyconfind)."""

import math
import random
import re

import numpy as np
import pytest

from marinfold.document_structures.contacts_and_coordinates_v1.generate import (
    GenerationConfig,
    _MAX_SUPPORTED_DEPTH,
    _apply_frame,
    _coordinate_digits,
    _random_rotation_matrix,
    _xyz_tokens,
)


def test_spec_worked_example_digits():
    # SPEC → Digit extraction: (205.3, 180.2, 5.7) rounds to (205, 180, 6)
    # at the default 1 Å resolution (max_depth = 3).
    assert _coordinate_digits(205.3, 3) == (2, 0, 5)
    assert _coordinate_digits(180.2, 3) == (1, 8, 0)
    assert _coordinate_digits(5.7, 3) == (0, 0, 6)
    assert _xyz_tokens(205.3, 180.2, 5.7, 3, 3) == [
        "<xyz-210>", "<xyz-080>", "<xyz-506>",
    ]


def test_digit_extraction_is_integer_round_to_nearest_angstrom():
    # 1 Å resolution rounds to the nearest integer Å before reading digits.
    assert _coordinate_digits(180.2, 3) == (1, 8, 0)
    assert _coordinate_digits(180.7, 3) == (1, 8, 1)
    assert _coordinate_digits(9.9, 3) == (0, 1, 0)  # rounds up across the ones place


def test_max_depth_4_knob_reintroduces_tenths_float_safe():
    # The future max_depth=4 variant quantizes as round(v*10) — never a float
    # /0.1 divide. 180.2 / 0.1 == 1801.9999999999998 would corrupt the tenths
    # digit; the integer path reads it as 2.
    assert _coordinate_digits(205.3, 4) == (2, 0, 5, 3)
    assert _coordinate_digits(180.2, 4) == (1, 8, 0, 2)
    assert _xyz_tokens(205.3, 180.2, 5.7, 4, 4) == [
        "<xyz-210>", "<xyz-080>", "<xyz-505>", "<xyz-327>",
    ]


def test_coarse_max_depth_1_and_2_yield_integer_digits():
    # Regression: GenerationConfig validation accepts max_depth in
    # [1, _MAX_SUPPORTED_DEPTH], but the old extraction used a float divisor
    # (100 * 10**(max_depth - 3), a float for max_depth < 3), so every "digit"
    # came out a float and token formatting crashed with
    # "Unknown format code 'd' for object of type 'float'". Digits must be ints
    # at every advertised depth, and compose into valid <xyz-NNN> tokens.
    for value in (0.0, 5.7, 123.4, 999.0, 1e4):
        for max_depth in (1, 2):
            digits = _coordinate_digits(value, max_depth)
            assert len(digits) == max_depth
            assert all(isinstance(d, int) for d in digits), (value, max_depth, digits)
    for depth in (1, 2):
        toks = _xyz_tokens(205.3, 180.2, 5.7, depth, depth)
        assert len(toks) == depth
        assert all(re.fullmatch(r"<xyz-\d{3}>", t) for t in toks), toks
    # The whole advertised range constructs without raising.
    for d in range(1, _MAX_SUPPORTED_DEPTH + 1):
        GenerationConfig(max_depth=d)


def test_digit_extraction_clamps():
    assert _coordinate_digits(-5.0, 3) == (0, 0, 0)
    assert _coordinate_digits(10_000.0, 3) == (9, 9, 9)  # clamped to 999
    assert _coordinate_digits(999.0, 3) == (9, 9, 9)
    assert _coordinate_digits(0.0, 3) == (0, 0, 0)
    # max_depth 4 clamps to 999.9.
    assert _coordinate_digits(10_000.0, 4) == (9, 9, 9, 9)


def test_xyz_tokens_depth_prefix():
    toks = _xyz_tokens(205.3, 180.2, 5.7, 2, 3)
    assert toks == ["<xyz-210>", "<xyz-080>"]  # hundreds + tens only
    assert _xyz_tokens(205.3, 180.2, 5.7, 1, 3) == ["<xyz-210>"]


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
    # Default 1 Å resolution: one sigma per depth, w_d / 4, no floor.
    assert GenerationConfig().depth_sigmas() == (25.0, 2.5, 0.25)
    # max_depth 4 would add the (physically over-precise) tenths sigma.
    assert GenerationConfig(max_depth=4).depth_sigmas() == (25.0, 2.5, 0.25, 0.025)
