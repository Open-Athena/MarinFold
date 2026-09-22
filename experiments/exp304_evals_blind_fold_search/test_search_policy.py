"""Behavioral checks for reference-free diversity and branch construction."""

from search_policy import branch_bundles, diverse_indices


def test_minority_contact_bundle_survives_dominant_consensus() -> None:
    common = {(i, i + 20) for i in range(10)}
    dominant = {(30 + i, 60 + i) for i in range(10)}
    minority = {(40 + i, 70 + i) for i in range(10)}
    maps = [frozenset(common | dominant | {(100 + i, 120 + i)}) for i in range(8)]
    maps += [frozenset(common | minority | {(200 + i, 220 + i)}) for i in range(2)]

    selected = diverse_indices(maps, 2)
    bundles = branch_bundles(maps, 2, 5)

    assert any(index >= 8 for index in selected)
    assert len(bundles) == 2
    assert any(bundle <= minority for bundle in bundles)
    assert any(bundle <= dominant for bundle in bundles)


def test_empty_and_duplicate_maps_do_not_fill_shortlist() -> None:
    map_a = frozenset((i, i + 20) for i in range(10))
    map_b = frozenset((i + 30, i + 50) for i in range(10))
    assert diverse_indices([frozenset(), map_a, map_a, map_b], 16) == [1, 3]
