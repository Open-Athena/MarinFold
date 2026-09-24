"""Unit tests for oracle scoring and pool construction."""

import pandas as pd

from analyze_results import alternating_pool, rollout_r_precision, top_two_pool


def frame(prefix: str, count: int) -> pd.DataFrame:
    """Build a tiny ordered frame carrying distinguishable source rows."""
    return pd.DataFrame(
        {
            "rollout": range(count),
            "source": [f"{prefix}{index}" for index in range(count)],
        }
    )


def test_rollout_precision_uses_emission_order_and_fixed_denominator() -> None:
    truth = {(0, 10), (1, 11), (2, 12)}
    contacts = [(9, 19), (0, 10), (0, 10), (1, 11), (2, 12)]
    assert rollout_r_precision(contacts, truth) == 2 / 3


def test_budget_matched_pool_construction() -> None:
    native = frame("n", 100)
    first = frame("a", 100)
    second = frame("b", 100)
    mixed = alternating_pool(native, first)
    assert list(mixed.source.iloc[:4]) == ["n0", "a0", "n1", "a1"]
    assert len(mixed) == 100
    heterogeneous = top_two_pool(native, first, second)
    assert len(heterogeneous) == 100
    assert sum(value.startswith("n") for value in heterogeneous.source) == 50
    assert sum(value.startswith("a") for value in heterogeneous.source) == 25
    assert sum(value.startswith("b") for value in heterogeneous.source) == 25
