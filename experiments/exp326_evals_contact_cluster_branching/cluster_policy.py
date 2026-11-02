"""Truth-free contact co-clustering and coherent branch-bundle selection."""

import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

Contact = tuple[int, int]
ContactMap = list[Contact]


@dataclass(frozen=True)
class BundlePlan:
    """One branch prompt and its truth-free selection provenance."""

    bundle: tuple[Contact, ...]
    source_rollout: int
    cluster_id: int
    cluster_visits: int
    n_stable_clusters: int
    used_fallback: bool


def canonical(contact: Iterable[int]) -> Contact:
    """Return one ordered integer contact."""
    i, j = (int(value) for value in contact)
    return (min(i, j), max(i, j))


def canonical_maps(maps: Iterable[Iterable[Iterable[int]]]) -> list[ContactMap]:
    """Canonicalize maps while preserving emission order and first occurrence."""
    result = []
    for contacts in maps:
        seen: set[Contact] = set()
        ordered = []
        for raw in contacts:
            contact = canonical(raw)
            if contact[0] == contact[1] or contact in seen:
                continue
            seen.add(contact)
            ordered.append(contact)
        result.append(ordered)
    return result


def contact_distance(left: Contact, right: Contact) -> int:
    """Endpoint Chebyshev distance, allowing either contact orientation."""
    direct = max(abs(left[0] - right[0]), abs(left[1] - right[1]))
    reverse = max(abs(left[0] - right[1]), abs(left[1] - right[0]))
    return min(direct, reverse)


def contact_neighborhoods(
    maps: list[ContactMap], radius: int = 2
) -> tuple[dict[Contact, Contact], dict[Contact, int]]:
    """Collapse local coordinate substitutions around frequency-ranked anchors.

    Greedy anchors avoid transitive chains in which a sequence of one-residue
    shifts merges contacts whose endpoints are ultimately far apart. A small
    endpoint-bin index keeps this near-linear in the candidate count.
    """
    frequency = Counter(contact for contacts in maps for contact in set(contacts))
    ordered = sorted(frequency, key=lambda contact: (-frequency[contact], contact))
    width = radius + 1
    bins: dict[tuple[int, int], list[Contact]] = defaultdict(list)
    assignment: dict[Contact, Contact] = {}
    representatives: dict[Contact, int] = {}
    for contact in ordered:
        key = (contact[0] // width, contact[1] // width)
        candidates = []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                candidates.extend(bins.get((key[0] + di, key[1] + dj), ()))
        nearby = [
            anchor
            for anchor in candidates
            if contact_distance(contact, anchor) <= radius
        ]
        if nearby:
            anchor = min(
                nearby,
                key=lambda item: (
                    contact_distance(contact, item),
                    -frequency[item],
                    item,
                ),
            )
        else:
            anchor = contact
            bins[key].append(anchor)
            representatives[anchor] = 0
        assignment[contact] = anchor
        representatives[anchor] += frequency[contact]
    return assignment, representatives


def _incidence(
    maps: list[ContactMap], assignment: dict[Contact, Contact]
) -> tuple[list[Contact], np.ndarray]:
    representatives = sorted(set(assignment.values()))
    column = {contact: index for index, contact in enumerate(representatives)}
    matrix = np.zeros((len(maps), len(representatives)), dtype=np.float64)
    for row, contacts in enumerate(maps):
        for contact in set(contacts):
            matrix[row, column[assignment[contact]]] = 1.0
    return representatives, matrix


def _stable_contact_clusters(
    representatives: list[Contact], matrix: np.ndarray, bundle_size: int
) -> tuple[dict[int, set[Contact]], dict[Contact, int], dict[int, float]]:
    """Find tight clusters whose associations reproduce in both split halves.

    The distance between two contacts is one minus the smaller of their
    first-half and second-half occurrence correlations. Complete linkage at a
    0.2 correlation floor therefore prevents a cluster from being held
    together by an association that appears in only one half. This also avoids
    arbitrary label alignment when several contacts have identical occurrence
    signatures.
    """
    split = len(matrix) // 2
    first, second = matrix[:split], matrix[split:]
    first_support = first.sum(axis=0)
    second_support = second.sum(axis=0)
    total_support = matrix.sum(axis=0)
    keep = (
        (first_support >= 2)
        & (first_support <= len(first) - 2)
        & (second_support >= 2)
        & (second_support <= len(second) - 2)
        & (total_support >= 5)
        & (total_support <= len(matrix) - 5)
    )
    indices = np.flatnonzero(keep)
    if len(indices) < bundle_size:
        return {}, {}, {}

    first_correlation = np.corrcoef(first[:, indices], rowvar=False)
    second_correlation = np.corrcoef(second[:, indices], rowvar=False)
    stable_correlation = np.minimum(first_correlation, second_correlation)
    stable_correlation = np.nan_to_num(stable_correlation, nan=-1.0)
    distance = 1.0 - np.clip(stable_correlation, -1.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    distance = (distance + distance.T) / 2.0
    labels = fcluster(
        linkage(squareform(distance, checks=True), method="complete"),
        t=0.8,
        criterion="distance",
    )

    clusters: dict[int, set[Contact]] = {}
    contact_cluster: dict[Contact, int] = {}
    stability: dict[int, float] = {}
    for label in sorted(set(labels)):
        members = np.flatnonzero(labels == label)
        if len(members) < bundle_size:
            continue
        submatrix = stable_correlation[np.ix_(members, members)]
        off_diagonal = submatrix[~np.eye(len(members), dtype=bool)]
        stability[int(label)] = float(np.mean(off_diagonal))
        contacts = {representatives[int(indices[index])] for index in members}
        clusters[int(label)] = contacts
        contact_cluster.update({contact: int(label) for contact in contacts})
    return clusters, contact_cluster, stability


def _rollout_bundle(
    contacts: ContactMap,
    assignment: dict[Contact, Contact],
    cluster: set[Contact],
    support: dict[Contact, int],
    bundle_size: int,
) -> tuple[Contact, ...] | None:
    """Choose supported, neighborhood-distinct contacts from one real rollout."""
    first_for_representative: dict[Contact, Contact] = {}
    emission_index: dict[Contact, int] = {}
    for index, contact in enumerate(contacts):
        representative = assignment[contact]
        if representative in cluster and representative not in first_for_representative:
            first_for_representative[representative] = contact
            emission_index[contact] = index
    if len(first_for_representative) < bundle_size:
        return None
    selected_representatives = sorted(
        first_for_representative,
        key=lambda item: (-support[item], item),
    )[:bundle_size]
    selected = [first_for_representative[item] for item in selected_representatives]
    return tuple(sorted(selected, key=emission_index.__getitem__))


def _coherent_random_bundle(
    contacts: ContactMap,
    assignment: dict[Contact, Contact],
    bundle_size: int,
    rng: random.Random,
) -> tuple[Contact, ...]:
    """Sample one neighborhood-distinct bundle from a single observed map."""
    first_for_representative: dict[Contact, Contact] = {}
    emission_index: dict[Contact, int] = {}
    for index, contact in enumerate(contacts):
        representative = assignment[contact]
        if representative not in first_for_representative:
            first_for_representative[representative] = contact
            emission_index[contact] = index
    candidates = list(first_for_representative.values())
    if len(candidates) < bundle_size:
        raise ValueError("source rollout has too few distinct contact neighborhoods")
    selected = rng.sample(candidates, bundle_size)
    return tuple(sorted(selected, key=emission_index.__getitem__))


def make_bundle_plans(
    maps: list[ContactMap],
    bundle_size: int,
    n_branches: int = 50,
    seed: int = 326,
) -> tuple[list[BundlePlan], list[BundlePlan], dict[str, float]]:
    """Return paired cluster-aware and coherent-random branch plans.

    Every random control uses the same source rollout as its cluster-aware mate.
    Equal allocation across eligible stable clusters deliberately upsamples
    minority clusters without giving the rarest cluster unbounded priority.
    """
    if len(maps) != 50:
        raise ValueError(f"expected 50 warm-up maps, got {len(maps)}")
    if bundle_size not in (3, 5):
        raise ValueError("bundle_size must be 3 or 5")
    maps = canonical_maps(maps)
    assignment, _ = contact_neighborhoods(maps)
    representatives, matrix = _incidence(maps, assignment)
    support = {
        representative: int(matrix[:, index].sum())
        for index, representative in enumerate(representatives)
    }
    clusters, _, stability = _stable_contact_clusters(
        representatives, matrix, bundle_size
    )
    candidates: dict[int, list[tuple[int, tuple[Contact, ...]]]] = {}
    visits: dict[int, int] = {}
    for cluster_id, cluster in clusters.items():
        bundles = []
        for rollout, contacts in enumerate(maps):
            bundle = _rollout_bundle(
                contacts, assignment, cluster, support, bundle_size
            )
            if bundle is not None:
                bundles.append((rollout, bundle))
        unique = []
        seen: set[tuple[Contact, ...]] = set()
        for item in bundles:
            if item[1] not in seen:
                unique.append(item)
                seen.add(item[1])
        if len(unique) >= 2:
            candidates[cluster_id] = unique
            visits[cluster_id] = len(bundles)

    rng = random.Random(seed)
    eligible = sorted(candidates, key=lambda label: (visits[label], label))
    cluster_plans: list[BundlePlan] = []
    if eligible:
        counters = Counter()
        while len(cluster_plans) < n_branches:
            # Equal branch quotas make clusters with fewer warm-up visits the
            # least-covered states after combining warm-up and branch counts.
            cluster_id = min(
                eligible,
                key=lambda label: (counters[label], visits[label], label),
            )
            choices = candidates[cluster_id]
            source_rollout, bundle = choices[counters[cluster_id] % len(choices)]
            counters[cluster_id] += 1
            cluster_plans.append(
                BundlePlan(
                    bundle=bundle,
                    source_rollout=source_rollout,
                    cluster_id=cluster_id,
                    cluster_visits=visits[cluster_id],
                    n_stable_clusters=len(eligible),
                    used_fallback=False,
                )
            )

    while len(cluster_plans) < n_branches:
        viable = [
            index
            for index, contacts in enumerate(maps)
            if len({assignment[contact] for contact in contacts}) >= bundle_size
        ]
        if not viable:
            raise ValueError("no warm-up rollout can supply a coherent bundle")
        source_rollout = rng.choice(viable)
        bundle = _coherent_random_bundle(
            maps[source_rollout], assignment, bundle_size, rng
        )
        cluster_plans.append(
            BundlePlan(
                bundle=bundle,
                source_rollout=source_rollout,
                cluster_id=-1,
                cluster_visits=0,
                n_stable_clusters=len(eligible),
                used_fallback=True,
            )
        )

    random_plans = []
    for plan in cluster_plans:
        bundle = _coherent_random_bundle(
            maps[plan.source_rollout], assignment, bundle_size, rng
        )
        random_plans.append(
            BundlePlan(
                bundle=bundle,
                source_rollout=plan.source_rollout,
                cluster_id=plan.cluster_id,
                cluster_visits=plan.cluster_visits,
                n_stable_clusters=plan.n_stable_clusters,
                used_fallback=plan.used_fallback,
            )
        )
    diagnostics = {
        "n_contact_neighborhoods": float(len(representatives)),
        "n_stable_clusters": float(len(eligible)),
        "fallback_fraction": float(
            np.mean([plan.used_fallback for plan in cluster_plans])
        ),
        "mean_cluster_visits": (
            float(np.mean(list(visits.values()))) if visits else 0.0
        ),
        "mean_retained_cluster_stability": (
            float(np.mean([stability[label] for label in eligible]))
            if eligible
            else 0.0
        ),
    }
    return cluster_plans, random_plans, diagnostics
