"""Exact-budget ranking must not silently lose contacts during mapping."""

import pytest

from prepare_marinfold_l2 import select_pairs


def test_filter_before_cut_retains_order_and_reaches_exact_budget() -> None:
    ranked = [[0, 5], [0, 10], [2, 7], [1, 10], [0, 20]]
    assert select_pairs(ranked, 2) == [(2, 0, 10), (4, 1, 10)]
    with pytest.raises(ValueError, match="Only 3 eligible pairs"):
        select_pairs(ranked, 4)


def test_duplicate_pairs_cannot_fill_the_budget() -> None:
    with pytest.raises(ValueError, match="Duplicate ranked pair"):
        select_pairs([[0, 10], [10, 0], [1, 10]], 2)
