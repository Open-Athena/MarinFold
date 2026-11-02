"""Independent small examples for diversity, count control and shuffle margins."""

from collections import Counter

import numpy as np

from analysis_stats import curveball, diversity, thin_matched


def test_pairwise_distance_with_empty_samples() -> None:
    result = diversity([set(), {1}, {1, 2}], 5)
    assert np.isclose(result["jaccard"], (1 + 1 + 0.5) / 3)
    assert diversity([set(), set()], 5)["jaccard"] == 0


def test_frequency_preserving_shuffle() -> None:
    original = [{1, 2, 3}, {1, 2}, {3, 4}, {5}, set()]
    shuffled = curveball(original, 373, 1000)
    assert [len(s) for s in original] == [len(s) for s in shuffled]
    assert Counter(x for s in original for x in s) == Counter(
        x for s in shuffled for x in s
    )
    assert original == [{1, 2, 3}, {1, 2}, {3, 4}, {5}, set()]


def test_count_control_retains_subsets() -> None:
    original = [{1, 2, 3}, set()]
    other = [{4, 5}, {6}]
    a, b = thin_matched(original, other, 373)
    assert list(map(len, a)) == list(map(len, b)) == [2, 0]
    assert all(x <= y for x, y in zip(a, original, strict=True))
    assert all(x <= y for x, y in zip(b, other, strict=True))
