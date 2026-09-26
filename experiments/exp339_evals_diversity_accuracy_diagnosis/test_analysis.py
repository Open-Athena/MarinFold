"""Small counterexamples for the diagnosis's metric distinctions."""

import numpy as np
import pandas as pd

from analyze import analyze_pool


def frame(maps: list[list[tuple[int, int]]], valid: list[bool]) -> pd.DataFrame:
    """Construct complete maps with controlled validity."""
    return pd.DataFrame({"rollout": range(len(maps)), "contacts": maps,
                         "finished": valid, "malformed_contacts": 0,
                         "native_nll": np.arange(len(maps), dtype=float)})


def test_emission_order_is_not_an_unordered_map_oracle() -> None:
    """Two true contacts exist, but an early false contact displaces one."""
    truth = {"stem": "synthetic", "L": 30, "resolved": list(range(30)),
             "contacts": [(0, 10, 1.0), (0, 20, 1.0)]}
    samples = frame([[(1, 10), (0, 10), (0, 20)]] * 100, [True] * 100)
    discovery = frame([[(0, 10), (0, 20)]] * 100, [True] * 100)
    result = analyze_pool(samples, discovery, truth, "all")
    assert result["best_emission_fixed_r"] == 0.5
    assert result["best_within_map_truth_ranking"] == 1.0
    assert result["best_frequency_fixed_r"] == 1.0
    assert np.isclose(result["mean_random_order_fixed_r"], 2 / 3)


def test_union_truth_is_not_a_realizable_single_sample() -> None:
    """Perfect union coverage can coexist with short, incomplete samples."""
    truth = {"stem": "synthetic", "L": 30, "resolved": list(range(30)),
             "contacts": [(0, 10, 1.0), (0, 20, 1.0)]}
    samples = frame([[(0, 10)], [(0, 20)]] * 50, [True] * 100)
    result = analyze_pool(samples, samples, truth, "all")
    assert result["union_recall"] == 1.0
    assert result["best_within_map_truth_ranking"] == 0.5
    assert result["best_cardinality_ceiling"] == 0.5


def test_invalid_map_cannot_supply_primary_oracle_or_union() -> None:
    """An invalid perfect map must not rescue an otherwise wrong pool."""
    truth = {"stem": "synthetic", "L": 30, "resolved": list(range(30)),
             "contacts": [(0, 10, 1.0), (0, 20, 1.0)]}
    maps = [[(0, 10), (0, 20)]] + [[(1, 10)]] * 99
    samples = frame(maps, [False] + [True] * 99)
    before = samples.native_nll.copy()
    result = analyze_pool(samples, samples, truth, "all")
    assert result["best_within_map_truth_ranking"] == 0.0
    assert result["best_emission_fixed_r"] == 0.0
    assert result["union_recall"] == 0.0
    pd.testing.assert_series_equal(samples.native_nll, before)
