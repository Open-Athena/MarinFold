# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Noise-calibration tests (SPEC → Noise model).

Verifies that the SPEC's sigma choices actually deliver the target
bin-center reliability: for a true value sitting at the exact center of its
bin, the finest revealed digit is correct ~95.45% of the time at depths
1-3 (sigma = bin_width/4) and ~38.3% at depth 4 (sigma floored at 0.1 Å).
These are statistical checks on the calibration, not restatements of the
generator's arithmetic.
"""

import random

import pytest

from marinfold.document_structures.contacts_and_coordinates_v1.generate import (
    GenerationConfig,
    _coordinate_digits,
)


_SIGMAS = GenerationConfig().depth_sigmas()
_N = 40_000


def _reliability(true_value: float, sigma: float, place: int, n: int) -> float:
    """Fraction of noisy draws whose digit at ``place`` equals the truth's."""
    rng = random.Random(20240607)
    true_digit = _coordinate_digits(true_value)[place]
    hits = 0
    for _ in range(n):
        noisy = true_value + rng.gauss(0.0, sigma)
        if _coordinate_digits(noisy)[place] == true_digit:
            hits += 1
    return hits / n


@pytest.mark.parametrize(
    ("depth", "true_value", "place"),
    [
        (1, 250.0, 0),   # hundreds bin [200,300), center 250, sigma 25
        (2, 255.0, 1),   # tens bin     [250,260), center 255, sigma 2.5
        (3, 255.5, 2),   # ones bin     [255,256), center 255.5, sigma 0.25
    ],
)
def test_bin_center_reliability_is_about_95pct(depth, true_value, place):
    sigma = _SIGMAS[depth - 1]
    reliability = _reliability(true_value, sigma, place, _N)
    # Target 2*Phi(2)-1 = 0.9545; allow sampling slop.
    assert 0.94 <= reliability <= 0.97, reliability


def test_tenths_digit_is_deliberately_soft():
    # Depth 4: sigma floored at 0.1 -> bin-center reliability ~2*Phi(0.5)-1
    # = 0.383, intentionally the softest of the four digits.
    sigma = _SIGMAS[3]
    assert sigma == 0.1
    # A tenths-digit bin is centered on x.x0 (digit extraction rounds), so
    # 250.30 is the center of the digit-3 bin [250.25, 250.35).
    reliability = _reliability(250.30, sigma, 3, _N)
    assert 0.35 <= reliability <= 0.42, reliability
