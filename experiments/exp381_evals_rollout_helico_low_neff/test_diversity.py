"""Test contact geometry and representative selection on known cluster structure."""

import numpy as np
import pytest

from select_diversity import jaccard, medoids
from select_maxmin import farthest_points


def test_jaccard_handles_empty_sets_repeats_and_large_intersections() -> None:
    large = [(i, i + 10) for i in range(300)]
    distance = jaccard([[], [], large, large + [large[0]], large[:150]])
    assert distance[0, 1] == 0
    assert distance[0, 2] == 1
    assert distance[2, 3] == 0
    assert distance[2, 4] == 0.5
    np.testing.assert_array_equal(np.diag(distance), 0)
    np.testing.assert_array_equal(distance, distance.T)


def test_medoids_recover_separated_clusters_with_sampling_multiplicity() -> None:
    contacts = [
        [(cluster * 100 + i, cluster * 100 + i + 10) for i in range(20)]
        for cluster in range(10)
        for _ in range(cluster + 1)
    ]
    distance = jaccard(contacts)
    chosen, fit = medoids(distance, 10, 381)
    assert fit["objective"] == 0
    assert len({tuple(contacts[i]) for i in chosen}) == 10
    assert np.all(np.diff(chosen) > 0)
    np.testing.assert_array_equal(chosen, medoids(distance, 10, 381)[0])


def test_insufficient_unique_sets_fails_instead_of_repeating_medoids() -> None:
    with pytest.raises(ValueError, match="distinct"):
        medoids(jaccard([[], [], [(1, 10)]]), 3, 381)


def test_sampling_frequency_changes_the_representative() -> None:
    a, b, c = [(1, 10)], [(1, 10), (2, 11)], [(2, 11)]
    unique = medoids(jaccard([a, b, c]), 1, 381)[0]
    weighted = medoids(jaccard([a] * 8 + [b, c]), 1, 381)[0]
    assert unique.tolist() == [1]
    assert weighted.tolist() == [0]


def test_farthest_points_chooses_a_distant_set_without_duplicates() -> None:
    contacts = [[(1, 10)]] * 8 + [[(1, 10), (2, 11)], [(2, 11)]]
    chosen = farthest_points(jaccard(contacts), 2)
    assert chosen.tolist() == [0, 9]
    with pytest.raises(ValueError, match="distinct"):
        farthest_points(jaccard(contacts), 4)
