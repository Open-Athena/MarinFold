"""Reference-free diverse shortlist selection, matched to exp304."""

from collections import Counter
from collections.abc import Iterable
from math import log1p, sqrt
from statistics import median

Contact = tuple[int, int]
ContactMap = frozenset[Contact]


def canonical(pairs: Iterable[Iterable[int]]) -> ContactMap:
    """Canonicalize a contact map as unique ordered residue pairs."""
    return frozenset(
        (min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs if i != j
    )


def contact_weights(maps: list[ContactMap]) -> dict[Contact, float]:
    """Downweight ubiquitous contacts while retaining rare pairs."""
    frequency = Counter(pair for contacts in maps for pair in contacts)
    return {pair: 1.0 / sqrt(count) for pair, count in frequency.items()}


def similarity(a: ContactMap, b: ContactMap, weights: dict[Contact, float]) -> float:
    """Weighted Jaccard similarity between two complete maps."""
    union = a | b
    if not union:
        return 1.0
    return sum(weights[pair] for pair in a & b) / sum(weights[pair] for pair in union)


def eligible_maps(maps: list[ContactMap]) -> list[int]:
    """Exclude empty, severely sparse, and exact-duplicate maps."""
    lengths = [len(contacts) for contacts in maps if contacts]
    if not lengths:
        return []
    typical = median(lengths)
    lower, upper = max(5, typical * 0.5), typical * 1.5
    seen: set[ContactMap] = set()
    kept = []
    for index, contacts in enumerate(maps):
        if lower <= len(contacts) <= upper and contacts not in seen:
            kept.append(index)
            seen.add(contacts)
    return kept


def diverse_indices(maps: list[ContactMap], n: int, novelty: float = 1.0) -> list[int]:
    """Greedily select the same support/diversity shortlist as exp304."""
    candidates = eligible_maps(maps)
    if not candidates:
        return []
    weights = contact_weights([maps[index] for index in candidates])
    support = Counter(pair for index in candidates for pair in maps[index])
    scale = log1p(len(candidates))
    quality = {
        index: sum(log1p(support[pair]) / scale for pair in maps[index]) / len(maps[index])
        for index in candidates
    }
    selected: list[int] = []
    remaining = set(candidates)
    nearest_distance = {index: 1.0 for index in candidates}
    while remaining and len(selected) < n:
        if not selected:
            choice = max(remaining, key=lambda index: (quality[index], -index))
        else:
            choice = max(
                remaining,
                key=lambda index: (
                    quality[index] + novelty * nearest_distance[index],
                    -index,
                ),
            )
        selected.append(choice)
        remaining.remove(choice)
        for index in remaining:
            distance = 1.0 - similarity(maps[index], maps[choice], weights)
            nearest_distance[index] = min(nearest_distance[index], distance)
    return selected
