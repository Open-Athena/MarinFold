# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Digit / box extraction + coordinate-frame tests (pure, no pyconfind)."""

import math
import random

import numpy as np
import pytest

from marinfold.document_structures.contacts_and_crops_v1.generate import (
    GenerationConfig,
    _apply_frame,
    _box_header_tokens,
    _cell,
    _digits,
    _neighbors,
    _ones_tenths_tokens,
    _random_rotation_matrix,
)


def test_spec_worked_example_digits():
    # SPEC worked example: (205.6, 72.3, 6.1) → quantize round(v*10) →
    # 2056 / 723 / 61. ones token <xyz-526>, tenths token <xyz-631>.
    assert _digits(205.6) == (2, 0, 5, 6)
    assert _digits(72.3) == (0, 7, 2, 3)
    assert _digits(6.1) == (0, 0, 6, 1)
    assert _ones_tenths_tokens(205.6, 72.3, 6.1) == ("<xyz-526>", "<xyz-631>")


def test_box_header_names_true_cell():
    # True (205.3, 71.8, 6.4): box x∈[200,210), y∈[70,80), z∈[0,10) →
    # header <xyz-200> <xyz-070>.
    cell = (_cell(205.3), _cell(71.8), _cell(6.4))
    assert cell == (20, 7, 0)
    assert _box_header_tokens(cell) == ("<xyz-200>", "<xyz-070>")


def test_cell_is_consistent_with_digits():
    # The cell index equals hundreds*10 + tens for the same quantization.
    for v in (0.0, 5.7, 99.94, 205.3, 254.95, 999.9):
        h, t, _o, _p = _digits(v)
        assert _cell(v) == h * 10 + t


def test_digit_extraction_float_safe_and_clamped():
    # round(v*10) never divides by a float 0.1 (180.2/0.1 == 1801.999… bug).
    assert _digits(180.2) == (1, 8, 0, 2)
    assert _digits(-5.0) == (0, 0, 0, 0)
    assert _digits(10_000.0) == (9, 9, 9, 9)  # clamped to 999.9
    assert _cell(-1.0) == 0
    assert _cell(10_000.0) == 99


def test_neighbors_26_and_bounds():
    # Interior cell has all 26 neighbors.
    assert len(_neighbors((10, 10, 10))) == 26
    assert (11, 11, 11) in _neighbors((10, 10, 10))
    assert (10, 10, 10) not in _neighbors((10, 10, 10))  # excludes self
    # A corner cell is clipped to the in-bounds subset.
    corner = _neighbors((0, 0, 0))
    assert len(corner) == 7
    assert all(0 <= c < 100 for nb in corner for c in nb)
    assert _neighbors((99, 99, 99)) and all(
        c < 100 for nb in _neighbors((99, 99, 99)) for c in nb
    )


def test_random_rotation_is_orthonormal():
    rng = random.Random(1234)
    for _ in range(20):
        matrix, quat = _random_rotation_matrix(rng)
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
    coords = np.random.default_rng(0).uniform(-495.0, 495.0, size=(1000, 3))
    assert _apply_frame(rng, coords, config) is None


def test_refine_sigma_schedule():
    cfg = GenerationConfig()
    assert cfg.refine_sigma(0) == 1.0
    assert cfg.refine_sigma(1) == 0.25
    assert math.isclose(cfg.refine_sigma(2), 1.0 / 9.0)
    assert cfg.refine_sigma(3) == 0.0625


def test_select_probs_validate():
    assert math.isclose(GenerationConfig().pass2_select_reshow, 0.10)
    with pytest.raises(ValueError):
        GenerationConfig(pass2_select_random=0.6, pass2_select_frontier=0.6)
