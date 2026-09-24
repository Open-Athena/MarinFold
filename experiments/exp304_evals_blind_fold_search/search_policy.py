"""Reference-free contact-map selection and self-conditioning policy.

The GPU worker imports only this module and receives sequence-only targets.
Every decision here depends on generated maps and their metadata, never on
Fold1/Fold2 contacts or switching-region annotations.
"""

from collections import Counter
from collections.abc import Iterable
from math import log1p, sqrt
import random
from statistics import median

Contact = tuple[int, int]
ContactMap = frozenset[Contact]


def canonical(pairs: Iterable[Iterable[int]]) -> ContactMap:
    """Canonicalize a contact map as unique ordered residue pairs."""
    return frozenset((min(int(i), int(j)), max(int(i), int(j))) for i, j in pairs if i != j)


def contact_weights(maps: list[ContactMap]) -> dict[Contact, float]:
    """Downweight ubiquitous contacts while retaining rare co-occurring pairs."""
    freq = Counter(pair for contacts in maps for pair in contacts)
    return {pair: 1.0 / sqrt(count) for pair, count in freq.items()}


def similarity(a: ContactMap, b: ContactMap, weights: dict[Contact, float]) -> float:
    """Weighted Jaccard similarity between complete generated maps."""
    union = a | b
    if not union:
        return 1.0
    return sum(weights[pair] for pair in a & b) / sum(weights[pair] for pair in union)


def eligible_maps(maps: list[ContactMap]) -> list[int]:
    """Exclude empty, severely sparse, and exact-duplicate maps without truth."""
    if not maps:
        return []
    lengths = [len(contacts) for contacts in maps if contacts]
    if not lengths:
        return []
    typical = median(lengths)
    lower = max(5, typical * 0.5)
    upper = typical * 1.5
    seen: set[ContactMap] = set()
    kept = []
    for idx, contacts in enumerate(maps):
        if lower <= len(contacts) <= upper and contacts not in seen:
            kept.append(idx)
            seen.add(contacts)
    return kept


def diverse_indices(maps: list[ContactMap], n: int, novelty: float = 1.0) -> list[int]:
    """Greedy quality/diversity selection of generated maps.

    Quality is contact support across the candidate pool. Diversity uses a
    weighted whole-map distance, so common consensus contacts have less say.
    The same selector is used for every inference arm.
    """
    candidates = eligible_maps(maps)
    if not candidates:
        return []
    weights = contact_weights([maps[idx] for idx in candidates])
    support = Counter(pair for idx in candidates for pair in maps[idx])
    scale = log1p(len(candidates))
    quality = {
        idx: sum(log1p(support[pair]) / scale for pair in maps[idx]) / len(maps[idx])
        for idx in candidates
    }
    selected: list[int] = []
    remaining = set(candidates)
    nearest_distance = {idx: 1.0 for idx in candidates}
    while remaining and len(selected) < n:
        if not selected:
            choice = max(remaining, key=lambda idx: (quality[idx], -idx))
        else:
            choice = max(
                remaining,
                key=lambda idx: (
                    quality[idx] + novelty * nearest_distance[idx],
                    -idx,
                ),
            )
        selected.append(choice)
        remaining.remove(choice)
        for idx in remaining:
            distance = 1.0 - similarity(maps[idx], maps[choice], weights)
            nearest_distance[idx] = min(nearest_distance[idx], distance)
    return selected


def branch_bundles(maps: list[ContactMap], n_branches: int, k: int) -> list[ContactMap]:
    """Select diverse roots and contacts supported by each root's neighborhood."""
    medoids = diverse_indices(maps, n_branches)
    if not medoids:
        return []
    weights = contact_weights(maps)
    global_support = Counter(pair for contacts in maps for pair in contacts)
    clusters: dict[int, list[int]] = {idx: [] for idx in medoids}
    for candidate in range(len(maps)):
        nearest = max(
            medoids, key=lambda idx: (similarity(maps[candidate], maps[idx], weights), -idx)
        )
        clusters[nearest].append(candidate)
    bundles = []
    for idx in medoids:
        members = clusters[idx]
        local_support = Counter(pair for other in members for pair in maps[other])
        outside = len(maps) - len(members)

        def branch_score(pair: Contact) -> tuple[float, int, Contact]:
            within = (local_support[pair] + 0.5) / (len(members) + 1)
            elsewhere = (global_support[pair] - local_support[pair] + 0.5) / (outside + 1)
            return within / elsewhere, local_support[pair], pair

        ranked = sorted(
            maps[idx],
            key=branch_score,
            reverse=True,
        )
        bundle = frozenset(ranked[:k])
        if bundle and bundle not in bundles:
            bundles.append(bundle)
    return bundles


def random_bundles(maps: list[ContactMap], n_branches: int, k: int, seed: int) -> list[ContactMap]:
    """Control: choose a random generated map before picking its seed contacts."""
    rng = random.Random(seed)
    eligible = eligible_maps(maps)
    rng.shuffle(eligible)
    bundles = []
    for idx in eligible[:n_branches]:
        bundle = frozenset(rng.sample(sorted(maps[idx]), min(k, len(maps[idx]))))
        if bundle and bundle not in bundles:
            bundles.append(bundle)
    return bundles
