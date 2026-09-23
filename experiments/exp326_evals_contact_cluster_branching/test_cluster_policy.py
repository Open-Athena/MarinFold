"""Tests for truth-free coherent contact-cluster branch planning."""

from cluster_policy import (
    canonical_maps,
    contact_neighborhoods,
    make_bundle_plans,
)


def synthetic_maps() -> list[list[tuple[int, int]]]:
    """Two stable modes plus shared contacts and deterministic light noise."""
    shared = [(2, 30), (4, 34), (6, 38), (8, 42), (10, 46)]
    left = [(20, 70), (22, 74), (24, 78), (26, 82), (28, 86), (30, 90)]
    right = [(40, 100), (42, 104), (44, 108), (46, 112), (48, 116), (50, 120)]
    maps = []
    for index in range(50):
        mode = left if index % 2 == 0 else right
        delta = (index // 2) % 2
        mode = [
            (left_index + delta, right_index + delta)
            for left_index, right_index in mode
        ]
        contacts = shared + mode
        contacts += [(60 + index % 5, 140 + index % 7)]
        maps.append(contacts)
    return maps


def test_neighborhoods_collapse_local_substitutions_without_chaining() -> None:
    maps = canonical_maps(
        [
            [(10, 50), (11, 51)],
            [(10, 50), (12, 52)],
            [(13, 53)],
        ]
    )
    assignment, _ = contact_neighborhoods(maps, radius=2)
    assert assignment[(10, 50)] == assignment[(11, 51)]
    assert assignment[(10, 50)] == assignment[(12, 52)]
    assert assignment[(13, 53)] != assignment[(10, 50)]


def test_cluster_bundles_are_coherent_and_deterministic() -> None:
    maps = synthetic_maps()
    cluster, random_control, diagnostics = make_bundle_plans(
        maps, bundle_size=3, seed=17
    )
    replay, replay_random, replay_diagnostics = make_bundle_plans(
        maps, bundle_size=3, seed=17
    )
    assert cluster == replay
    assert random_control == replay_random
    assert diagnostics == replay_diagnostics
    assert len(cluster) == len(random_control) == 50
    assert diagnostics["n_stable_clusters"] >= 2
    assert diagnostics["fallback_fraction"] == 0
    for plan in cluster:
        source = set(maps[plan.source_rollout])
        assert len(plan.bundle) == 3
        assert set(plan.bundle) <= source
        assert not plan.used_fallback
    for left, right in zip(cluster, random_control, strict=True):
        assert left.source_rollout == right.source_rollout


def test_fallback_still_uses_one_observed_rollout() -> None:
    maps = [
        [(index, index + 10), (index + 20, index + 40), (index + 50, index + 80)]
        for index in range(50)
    ]
    cluster, _, diagnostics = make_bundle_plans(maps, bundle_size=3, seed=9)
    assert diagnostics["fallback_fraction"] == 1
    for plan in cluster:
        assert set(plan.bundle) <= set(maps[plan.source_rollout])
        assert plan.used_fallback
