# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Noise-calibration checks (SPEC → box noise / refinement).

Statistical checks that the sigma choices deliver the documented behavior:
Pass 1's σ=2 keeps an at-center atom in its true 10 Å box ~98.8% of the time
(so a small fraction land in the wrong box on purpose), and Pass 2's
progressive σ = 1/(i+1)^2 sharpens a box's tenths digit on repeated reads.
"""

import random

from marinfold.document_structures.contacts_and_crops_v1.generate import (
    GenerationConfig,
    _cell,
    _digits,
)


def _box_reliability(true_value: float, sigma: float, n: int) -> float:
    """Fraction of σ-noised draws whose 10 Å cell equals the truth's."""
    rng = random.Random(20260717)
    true_cell = _cell(true_value)
    hits = sum(1 for _ in range(n) if _cell(true_value + rng.gauss(0.0, sigma)) == true_cell)
    return hits / n


def _tenths_reliability(true_value: float, sigma: float, n: int) -> float:
    """Fraction of σ-noised draws whose tenths digit equals the truth's."""
    rng = random.Random(20260717)
    true_tenths = _digits(true_value)[3]
    hits = sum(1 for _ in range(n) if _digits(true_value + rng.gauss(0.0, sigma))[3] == true_tenths)
    return hits / n


def test_pass1_box_reliability_about_98_8pct_at_center():
    # Cell 25 spans round(v*10) in [2500, 2599] → center ≈ 254.95; nearest
    # boundary 5 Å away, so at σ=2 P(stay) = 2Φ(2.5)-1 ≈ 0.9876.
    sigma = GenerationConfig().pass1_box_noise_sigma
    assert sigma == 2.0
    reliability = _box_reliability(254.95, sigma, 40_000)
    assert 0.975 <= reliability <= 0.995, reliability
    # A small but real fraction lands in the wrong box (intended).
    assert reliability < 1.0


def test_pass2_refinement_sharpens_tenths_digit():
    cfg = GenerationConfig()
    # A tenths-bin center (n=2555 → tenths digit 5, boundaries 0.05 Å away).
    true_value = 255.5
    # First read (σ=1.0, 20× the tenths half-width) is near-noise on the
    # tenths; σ shrinks as 1/(i+1)^2, so a deeply re-shown box nails it. The
    # tenths only becomes crisp after several reads — exactly the SPEC's
    # "usable everywhere, sharp only where budget allows".
    r0 = _tenths_reliability(true_value, cfg.refine_sigma(0), 40_000)
    r1 = _tenths_reliability(true_value, cfg.refine_sigma(1), 40_000)
    r3 = _tenths_reliability(true_value, cfg.refine_sigma(3), 40_000)
    r10 = _tenths_reliability(true_value, cfg.refine_sigma(10), 40_000)
    assert r0 < r1 < r3 < r10
    assert r0 < 0.2        # first read: tenths is basically noise
    assert r10 > 0.99      # crisp once a box is revisited deeply
