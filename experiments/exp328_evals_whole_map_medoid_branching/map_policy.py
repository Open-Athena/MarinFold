"""Truth-free whole-map clustering and medoid partial-map seed selection."""

import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import silhouette_score

Contact = tuple[int, int]
ContactMap = list[Contact]


@dataclass(frozen=True)
class SeedPlan:
    """One continuation seed and its truth-free selection provenance."""

    bundle: tuple[Contact, ...]
    source_rollout: int
    cluster_id: int
    cluster_visits: int
    medoid_rollout: int
    n_clusters: int
    silhouette: float
    used_fallback: bool


def canonical(contact: Iterable[int]) -> Contact:
    """Return one ordered integer contact."""
    left, right = (int(value) for value in contact)
    return (min(left, right), max(left, right))


def canonical_maps(maps: Iterable[Iterable[Iterable[int]]]) -> list[ContactMap]:
    """Canonicalize maps while preserving emission order and first occurrence."""
    result = []
    for raw_map in maps:
        seen: set[Contact] = set()
        contacts = []
        for raw_contact in raw_map:
            contact = canonical(raw_contact)
            if contact[0] == contact[1] or contact in seen:
                continue
            seen.add(contact)
            contacts.append(contact)
        result.append(contacts)
    return result


def contact_distance(left: Contact, right: Contact) -> int:
    """Endpoint Chebyshev distance between canonical contacts."""
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))


def _directed_coverage(left: ContactMap, right: ContactMap, radius: int) -> float:
    if not left:
        return 1.0
    if not right:
        return 0.0
    tree = cKDTree(np.asarray(right, dtype=np.float64))
    distances, _ = tree.query(np.asarray(left, dtype=np.float64), k=1, p=np.inf)
    return float(np.mean(distances <= radius))


def geometry_f1(left: ContactMap, right: ContactMap, radius: int) -> float:
    """Symmetric soft map F1 under an endpoint-coordinate tolerance."""
    if not left and not right:
        return 1.0
    precision = _directed_coverage(left, right, radius)
    recall = _directed_coverage(right, left, radius)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def map_distance(left: ContactMap, right: ContactMap) -> float:
    """Combine local (radius 2) and basin-scale (radius 8) geometric overlap."""
    return 1.0 - 0.5 * (
        geometry_f1(left, right, radius=2) + geometry_f1(left, right, radius=8)
    )


def pairwise_map_distances(maps: list[ContactMap]) -> np.ndarray:
    """Return the symmetric precomputed distance matrix used for clustering."""
    distances = np.zeros((len(maps), len(maps)), dtype=np.float64)
    for left in range(len(maps)):
        for right in range(left + 1, len(maps)):
            value = map_distance(maps[left], maps[right])
            distances[left, right] = value
            distances[right, left] = value
    return distances


def _assign(distances: np.ndarray, medoids: tuple[int, ...]) -> np.ndarray:
    ordered = np.asarray(sorted(medoids), dtype=np.int64)
    return ordered[np.argmin(distances[:, ordered], axis=1)]


def _cost(distances: np.ndarray, medoids: tuple[int, ...]) -> float:
    return float(np.min(distances[:, np.asarray(medoids)], axis=1).sum())


def kmedoids(distances: np.ndarray, k: int) -> tuple[tuple[int, ...], np.ndarray]:
    """Fit deterministic PAM-style k-medoids to a precomputed distance matrix."""
    n = len(distances)
    if distances.shape != (n, n) or not 1 <= k <= n:
        raise ValueError("invalid distance matrix or k")
    first = int(np.argmin(distances.sum(axis=1)))
    medoids = [first]
    while len(medoids) < k:
        nearest = np.min(distances[:, medoids], axis=1)
        nearest[medoids] = -1.0
        medoids.append(int(np.argmax(nearest)))
    current = tuple(sorted(medoids))
    current_cost = _cost(distances, current)
    while True:
        best = current
        best_cost = current_cost
        non_medoids = sorted(set(range(n)) - set(current))
        for old in current:
            for new in non_medoids:
                candidate = tuple(sorted((set(current) - {old}) | {new}))
                candidate_cost = _cost(distances, candidate)
                if (candidate_cost, candidate) < (best_cost - 1e-12, best):
                    best, best_cost = candidate, candidate_cost
        if best == current:
            break
        current, current_cost = best, best_cost
    assigned_medoid = _assign(distances, current)
    label_for_medoid = {medoid: label for label, medoid in enumerate(current)}
    labels = np.asarray(
        [label_for_medoid[int(medoid)] for medoid in assigned_medoid], dtype=np.int64
    )
    return current, labels


def choose_clustering(
    distances: np.ndarray, min_cluster_size: int = 3
) -> tuple[tuple[int, ...], np.ndarray, float, bool]:
    """Choose k=2..6 by silhouette, with a documented single-cluster fallback."""
    candidates = []
    for k in range(2, min(6, len(distances) - 1) + 1):
        medoids, labels = kmedoids(distances, k)
        sizes = np.bincount(labels, minlength=k)
        if int(sizes.min()) < min_cluster_size:
            continue
        silhouette = float(silhouette_score(distances, labels, metric="precomputed"))
        candidates.append((silhouette, -k, medoids, labels))
    if candidates:
        silhouette, _, medoids, labels = max(candidates, key=lambda item: item[:2])
        return medoids, labels, silhouette, False
    medoid = int(np.argmin(distances.sum(axis=1)))
    return (medoid,), np.zeros(len(distances), dtype=np.int64), float("nan"), True


def _specificity(
    contact: Contact,
    maps: list[ContactMap],
    labels: np.ndarray,
    cluster_id: int,
    radius: int = 2,
) -> float:
    present = np.asarray(
        [
            any(contact_distance(contact, other) <= radius for other in item)
            for item in maps
        ]
    )
    inside = labels == cluster_id
    outside = ~inside
    within = (float(present[inside].sum()) + 1.0) / (float(inside.sum()) + 2.0)
    beyond = (
        (float(present[outside].sum()) + 1.0) / (float(outside.sum()) + 2.0)
        if outside.any()
        else 0.0
    )
    return within - beyond


def _select_spread(
    contacts: ContactMap,
    bundle_size: int,
    length: int,
    specificity: dict[Contact, float] | None,
    rng: random.Random | None = None,
) -> tuple[Contact, ...]:
    if len(contacts) < bundle_size:
        raise ValueError("source rollout has too few contacts for requested seed")
    emission = {contact: index for index, contact in enumerate(contacts)}
    if specificity is None:
        if rng is None:
            raise ValueError("random spread selection requires an RNG")
        selected = [contacts[rng.randrange(len(contacts))]]
        normalized = {contact: 0.0 for contact in contacts}
    else:
        low = min(specificity.values())
        high = max(specificity.values())
        scale = max(high - low, 1e-12)
        normalized = {
            contact: (specificity[contact] - low) / scale for contact in contacts
        }
        selected = [
            max(
                contacts, key=lambda contact: (specificity[contact], -emission[contact])
            )
        ]
    while len(selected) < bundle_size:
        remaining = [contact for contact in contacts if contact not in selected]

        def score(contact: Contact) -> tuple[float, float, float, int]:
            spread = min(
                contact_distance(contact, chosen) for chosen in selected
            ) / max(length - 1, 1)
            combined = 0.6 * normalized[contact] + 0.4 * min(spread, 1.0)
            return combined, spread, normalized[contact], -emission[contact]

        selected.append(max(remaining, key=score))
    return tuple(sorted(selected, key=emission.__getitem__))


def make_seed_plans(
    maps: list[ContactMap],
    bundle_size: int,
    length: int,
    n_branches: int = 50,
    seed: int = 328,
) -> tuple[list[SeedPlan], list[SeedPlan], dict[str, float]]:
    """Return geometry-aware medoid plans and matched spread-random controls."""
    if len(maps) != 50:
        raise ValueError(f"expected 50 warm-up maps, got {len(maps)}")
    if bundle_size not in (8, 16):
        raise ValueError("bundle_size must be 8 or 16")
    maps = canonical_maps(maps)
    distances = pairwise_map_distances(maps)
    medoids, labels, silhouette, used_fallback = choose_clustering(distances)
    cluster_sizes = Counter(int(label) for label in labels)
    seeds: dict[int, tuple[Contact, ...]] = {}
    seed_medoids = []
    for cluster_id, _unconstrained_medoid in enumerate(medoids):
        members = np.flatnonzero(labels == cluster_id)
        eligible = [
            int(index) for index in members if len(maps[int(index)]) >= bundle_size
        ]
        if not eligible:
            raise ValueError(
                f"cluster {cluster_id} has no map with {bundle_size} contacts"
            )
        medoid = min(
            eligible,
            key=lambda index: (
                float(distances[index, members].sum()),
                index,
            ),
        )
        seed_medoids.append(medoid)
        specificity = {
            contact: _specificity(contact, maps, labels, cluster_id)
            for contact in maps[medoid]
        }
        seeds[cluster_id] = _select_spread(
            maps[medoid], bundle_size, length, specificity
        )

    medoid_plans = []
    counters = Counter()
    for _ in range(n_branches):
        cluster_id = min(
            range(len(medoids)),
            key=lambda label: (counters[label], cluster_sizes[label], label),
        )
        counters[cluster_id] += 1
        medoid = seed_medoids[cluster_id]
        medoid_plans.append(
            SeedPlan(
                bundle=seeds[cluster_id],
                source_rollout=medoid,
                cluster_id=cluster_id,
                cluster_visits=cluster_sizes[cluster_id],
                medoid_rollout=medoid,
                n_clusters=len(medoids),
                silhouette=silhouette,
                used_fallback=used_fallback,
            )
        )

    rng = random.Random(seed)
    random_plans = []
    viable = [
        index for index, contacts in enumerate(maps) if len(contacts) >= bundle_size
    ]
    for plan in medoid_plans:
        source = rng.choice(viable)
        bundle = _select_spread(maps[source], bundle_size, length, None, rng)
        random_plans.append(
            SeedPlan(
                bundle=bundle,
                source_rollout=source,
                cluster_id=-1,
                cluster_visits=0,
                medoid_rollout=-1,
                n_clusters=plan.n_clusters,
                silhouette=plan.silhouette,
                used_fallback=plan.used_fallback,
            )
        )

    within = [
        distances[i, j]
        for i in range(len(maps))
        for j in range(i + 1, len(maps))
        if labels[i] == labels[j]
    ]
    between = [
        distances[i, j]
        for i in range(len(maps))
        for j in range(i + 1, len(maps))
        if labels[i] != labels[j]
    ]
    summary = {
        "n_clusters": float(len(medoids)),
        "silhouette": silhouette,
        "min_cluster_size": float(min(cluster_sizes.values())),
        "max_cluster_size": float(max(cluster_sizes.values())),
        "mean_within_distance": float(np.mean(within)) if within else float("nan"),
        "mean_between_distance": float(np.mean(between)) if between else float("nan"),
        "used_fallback": float(used_fallback),
        "unique_medoid_seeds": float(len(set(seeds.values()))),
    }
    return medoid_plans, random_plans, summary
