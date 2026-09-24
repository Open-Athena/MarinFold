"""Tests for geometry-aware whole-map clustering and seed construction."""

import numpy as np

from map_policy import (
    choose_clustering,
    geometry_f1,
    make_seed_plans,
    pairwise_map_distances,
)


def shifted_map(offset: int, n: int = 20) -> list[tuple[int, int]]:
    """Make a diagonal-like synthetic map."""
    return [(index + offset, index + offset + 20) for index in range(n)]


def test_geometry_f1_tolerates_local_shifts_but_rejects_other_topology() -> None:
    base = shifted_map(0)
    local = [(left + 1, right + 1) for left, right in base]
    other = [(left, right + 30) for left, right in base]
    assert geometry_f1(base, local, radius=2) == 1.0
    assert geometry_f1(base, other, radius=8) == 0.0


def test_kmedoids_recovers_two_synthetic_map_basins() -> None:
    maps = [shifted_map(offset) for offset in (0, 1, 2)] + [
        [(left, right + 35) for left, right in shifted_map(offset)]
        for offset in (0, 1, 2)
    ]
    distances = pairwise_map_distances(maps)
    medoids, labels, silhouette, fallback = choose_clustering(distances)
    assert len(medoids) == 2
    assert len(set(labels[:3])) == 1
    assert len(set(labels[3:])) == 1
    assert labels[0] != labels[3]
    assert silhouette > 0.5
    assert not fallback


def test_seed_plans_are_deterministic_medoid_subsets_and_size_matched() -> None:
    maps = []
    for rollout in range(50):
        basin = 0 if rollout < 25 else 50
        maps.append([(index, index + 25 + basin + rollout % 2) for index in range(24)])
    medoid, random_control, diagnostics = make_seed_plans(
        maps, bundle_size=8, length=120, n_branches=50, seed=7
    )
    repeated, repeated_random, _ = make_seed_plans(
        maps, bundle_size=8, length=120, n_branches=50, seed=7
    )
    assert medoid == repeated
    assert random_control == repeated_random
    assert len(medoid) == len(random_control) == 50
    assert diagnostics["n_clusters"] == 2
    for plan in medoid:
        assert len(plan.bundle) == 8
        assert set(plan.bundle) <= set(maps[plan.medoid_rollout])
    assert all(len(plan.bundle) == 8 for plan in random_control)
    counts = np.bincount([plan.cluster_id for plan in medoid])
    assert int(counts.max() - counts.min()) <= 1
