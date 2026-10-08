"""Validate the scientific conditioning semantics before predictor dispatch."""

import numpy as np
import pytest

from oracle_budgets import BUDGETS, build_maps, requested_count


def test_sparse_maps_reveal_only_exact_true_contacts() -> None:
    oracle = np.ones((30, 30), dtype=np.int64)
    np.fill_diagonal(oracle, 0)
    for i in range(20):
        oracle[i, i + 7] = oracle[i + 7, i] = 2
    maps = build_maps(oracle, "protein-a", 30)
    for arm in BUDGETS:
        for replicate in range(2):
            state = maps[f"{arm}-{replicate}"]
            assert np.triu(state == 2, 1).sum() == requested_count(arm, 30)
            assert not np.any(state == 1)
            assert not np.any((state == 2) & (oracle != 2))
            assert np.array_equal(state, state.T)
    for replicate in range(2):
        ordered = sorted(BUDGETS, key=lambda arm: requested_count(arm, 30))
        for small, large in zip(ordered[:-1], ordered[1:]):
            assert np.all(maps[f"{large}-{replicate}"][maps[f"{small}-{replicate}"] == 2] == 2)
    assert not np.array_equal(maps["random_10-0"], maps["random_10-1"])
    assert not np.any(maps["top_0-0"])
    assert np.array_equal(maps["positive_all-0"] == 2, oracle == 2)
    assert not np.any(maps["positive_all-0"] == 1)
    assert np.array_equal(maps["oracle-0"], oracle)
    repeat = build_maps(oracle, "protein-a", 30)
    assert all(np.array_equal(state, repeat[key]) for key, state in maps.items())


def test_fixed_budget_fails_but_relative_budget_caps() -> None:
    oracle = np.zeros((20, 20), dtype=np.int64)
    oracle[0, 1:12] = oracle[1:12, 0] = 2
    maps = build_maps(oracle, "few-contacts", 100)
    assert np.triu(maps["random_L2-0"] == 2, 1).sum() == 11
    assert np.triu(maps["random_10-0"] == 2, 1).sum() == 10
    oracle[0, 9:12] = oracle[9:12, 0] = 0
    with pytest.raises(ValueError, match="requests 10 contacts"):
        build_maps(oracle, "too-few", 100)
