# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Noise-calibration tests (SPEC → Noise model).

Verifies that the SPEC's sigma choices actually deliver the target
bin-center reliability: for a true value sitting at the exact center of its
bin, the finest revealed digit is correct ~95.45% of the time at every depth
(sigma = bin_width/4, uniformly). This is a statistical check on the
calibration, not a restatement of the generator's arithmetic.

Digit extraction rounds to the nearest Å, so a digit's "bin" in true-value
space is offset by half an Å from the integer boundaries: e.g. the hundreds
digit is 2 for round(v) in [200, 299] i.e. v in [199.5, 299.5) — center
249.5. The trues below sit at those (offset) bin centers.
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
    true_digit = _coordinate_digits(true_value, 3)[place]
    hits = 0
    for _ in range(n):
        noisy = true_value + rng.gauss(0.0, sigma)
        if _coordinate_digits(noisy, 3)[place] == true_digit:
            hits += 1
    return hits / n


@pytest.mark.parametrize(
    ("depth", "true_value", "place"),
    [
        (1, 249.5, 0),   # hundreds digit 2: round(v) in [200,299], center 249.5, sigma 25
        (2, 254.5, 1),   # tens digit 5:    round(v) in [250,259], center 254.5, sigma 2.5
        (3, 255.0, 2),   # ones digit 5:    round(v) == 255,        center 255.0, sigma 0.25
    ],
)
def test_bin_center_reliability_is_about_95pct(depth, true_value, place):
    sigma = _SIGMAS[depth - 1]
    reliability = _reliability(true_value, sigma, place, _N)
    # Target 2*Phi(2)-1 = 0.9545 at every depth now (no tenths floor); allow
    # sampling slop.
    assert 0.94 <= reliability <= 0.97, reliability
