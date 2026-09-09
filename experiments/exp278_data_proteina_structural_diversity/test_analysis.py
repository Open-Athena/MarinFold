"""Protect the scientific interpretation of structural clustering and matching."""

import numpy as np

from analyze_screen import components, effective_clusters, matched_comparison


def test_excluded_bridge_cannot_join_retained_clusters() -> None:
    edges = [("alpha", "bridge"), ("bridge", "beta"), ("beta", "beta2")]
    result = components(["alpha", "beta", "beta2"], edges)
    assert {frozenset(group) for group in result} == {
        frozenset(["alpha"]),
        frozenset(["beta", "beta2"]),
    }


def test_entropy_penalizes_repeated_structures() -> None:
    stems = list("abcdef")
    equal_sizes = [("a", "b"), ("c", "d"), ("e", "f")]
    uneven_sizes = [("a", "b"), ("b", "c"), ("c", "d")]
    assert np.isclose(effective_clusters(stems, equal_sizes), 3)
    assert effective_clusters(stems, uneven_sizes) < 3


def test_matched_comparison_does_not_reward_larger_conditioned_pool() -> None:
    quality = []
    for condition, count in [
        ("unconditional", 6),
        ("1.x.x.x", 20),
        ("2.x.x.x", 20),
        ("3.x.x.x", 20),
    ]:
        quality.extend(
            {
                "length": "100",
                "condition": condition,
                "stem": f"{condition}-{i}",
                "quality_pass": "True",
            }
            for i in range(count)
        )
    result = matched_comparison(quality, [])
    assert result["matched_n_per_arm"] == 6
    assert result["ratio_median"] == 1


def test_unavailable_class_is_not_silently_removed_from_balanced_comparison() -> None:
    rows = [
        {
            "length": "100",
            "condition": "unconditional",
            "stem": str(i),
            "quality_pass": "True",
        }
        for i in range(6)
    ]
    result = matched_comparison(rows, [])
    assert result["matched_n_per_arm"] == 0
    assert result["ratio_median"] is None
