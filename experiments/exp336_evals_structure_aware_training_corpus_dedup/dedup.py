# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Witness-preserving sequence/structure deduplication.

The important contract is stronger than ordinary connected-component
clustering: every removed row names a retained row that directly clears the
active thresholds. A chain ``A ~ B ~ C`` can therefore never collapse ``C``
onto ``A`` when the pair ``A, C`` was not measured as redundant.

Selection is hierarchical. First, a fixed quality order builds star-shaped
sequence clusters: every member directly matches its sequence representative.
Second, each star is split into structural modes. The sequence representative
is always retained; a later member is retained as another mode unless an
already-retained member directly clears both the sequence and structure rules.
This makes the number rescued by the structure condition well defined: it is
the number of extra representatives relative to the frozen sequence-only
partition, not a difference between two unrelated graph traversals.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ProteinRow:
    """One training example in the deduplication universe."""

    row_id: str
    source: str
    source_tokens: int

    def __post_init__(self) -> None:
        if not self.row_id:
            raise ValueError("row_id must be non-empty")
        if not self.source:
            raise ValueError(f"{self.row_id}: source must be non-empty")
        if self.source_tokens <= 0:
            raise ValueError(f"{self.row_id}: source_tokens must be positive")


@dataclass(frozen=True)
class PairEvidence:
    """Symmetric pair evidence, retaining both directional normalizations."""

    left: str
    right: str
    sequence_identity: float
    sequence_coverage_left: float
    sequence_coverage_right: float
    tm_left: float | None = None
    tm_right: float | None = None
    structure_coverage_left: float | None = None
    structure_coverage_right: float | None = None
    contact_similarity: float | None = None

    def __post_init__(self) -> None:
        if not self.left or not self.right or self.left == self.right:
            raise ValueError("pair endpoints must be distinct, non-empty row IDs")
        for name in (
            "sequence_identity",
            "sequence_coverage_left",
            "sequence_coverage_right",
            "tm_left",
            "tm_right",
            "structure_coverage_left",
            "structure_coverage_right",
            "contact_similarity",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def key(self) -> frozenset[str]:
        """Unordered endpoint key."""
        return frozenset((self.left, self.right))

    @property
    def min_sequence_coverage(self) -> float:
        return min(self.sequence_coverage_left, self.sequence_coverage_right)

    @property
    def min_tm(self) -> float | None:
        if self.tm_left is None or self.tm_right is None:
            return None
        return min(self.tm_left, self.tm_right)

    @property
    def min_structure_coverage(self) -> float | None:
        if self.structure_coverage_left is None or self.structure_coverage_right is None:
            return None
        return min(self.structure_coverage_left, self.structure_coverage_right)


@dataclass(frozen=True)
class Thresholds:
    """The operational pairwise redundancy rule."""

    min_sequence_identity: float
    min_sequence_coverage: float = 0.8
    min_tm: float | None = None
    min_structure_coverage: float = 0.8
    min_contact_similarity: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "min_sequence_identity",
            "min_sequence_coverage",
            "min_tm",
            "min_structure_coverage",
            "min_contact_similarity",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

    @property
    def structure_aware(self) -> bool:
        return self.min_tm is not None or self.min_contact_similarity is not None


@dataclass(frozen=True)
class Removal:
    """A removed row and the retained row that directly justifies it."""

    row_id: str
    witness_id: str
    sequence_representative_id: str


@dataclass(frozen=True)
class DedupResult:
    """One threshold point, including the frozen sequence-only baseline."""

    kept: tuple[str, ...]
    removals: tuple[Removal, ...]
    sequence_representatives: tuple[str, ...]
    sequence_assignments: Mapping[str, str]

    @property
    def rescued_by_structure(self) -> tuple[str, ...]:
        """Extra kept rows relative to one representative per sequence star."""
        representatives = set(self.sequence_representatives)
        return tuple(row_id for row_id in self.kept if row_id not in representatives)


def is_sequence_match(evidence: PairEvidence, thresholds: Thresholds) -> bool:
    """Whether a pair clears the identity and bidirectional coverage rule."""
    return (
        evidence.sequence_identity >= thresholds.min_sequence_identity
        and evidence.min_sequence_coverage >= thresholds.min_sequence_coverage
    )


def is_joint_match(evidence: PairEvidence, thresholds: Thresholds) -> bool:
    """Whether a pair is redundant under every active sequence/structure rule."""
    if not is_sequence_match(evidence, thresholds):
        return False
    if thresholds.min_tm is not None:
        if evidence.min_tm is None or evidence.min_tm < thresholds.min_tm:
            return False
        if (
            evidence.min_structure_coverage is None
            or evidence.min_structure_coverage < thresholds.min_structure_coverage
        ):
            return False
    if thresholds.min_contact_similarity is not None:
        if (
            evidence.contact_similarity is None
            or evidence.contact_similarity < thresholds.min_contact_similarity
        ):
            return False
    return True


def _index_evidence(
    rows: Mapping[str, ProteinRow], evidence: Iterable[PairEvidence]
) -> tuple[dict[frozenset[str], PairEvidence], dict[str, list[PairEvidence]]]:
    by_pair: dict[frozenset[str], PairEvidence] = {}
    by_row = {row_id: [] for row_id in rows}
    for pair in evidence:
        unknown = set(pair.key) - rows.keys()
        if unknown:
            raise ValueError(f"pair {pair.left}, {pair.right} names unknown rows {sorted(unknown)}")
        if pair.key in by_pair:
            raise ValueError(f"duplicate pair evidence for {sorted(pair.key)}")
        by_pair[pair.key] = pair
        by_row[pair.left].append(pair)
        by_row[pair.right].append(pair)
    return by_pair, by_row


def _validate_order(rows: Mapping[str, ProteinRow], order: Sequence[str]) -> None:
    if len(order) != len(rows) or len(set(order)) != len(order):
        raise ValueError("order must name every row exactly once")
    if set(order) != set(rows):
        missing = sorted(set(rows) - set(order))
        extra = sorted(set(order) - set(rows))
        raise ValueError(f"order does not match rows: missing={missing}, extra={extra}")


def _best_witness(
    current_row: str,
    candidates: Iterable[str],
    by_pair: Mapping[frozenset[str], PairEvidence],
) -> str:
    """Choose the strongest direct witness with a stable row-ID tie break."""

    def score(row_id: str) -> tuple[float, float, float, float, str]:
        pair = by_pair[frozenset((current_row, row_id))]
        return (
            pair.contact_similarity if pair.contact_similarity is not None else -1.0,
            pair.min_tm if pair.min_tm is not None else -1.0,
            pair.sequence_identity,
            pair.min_sequence_coverage,
            row_id,
        )

    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("cannot choose a witness from an empty collection")
    return max(candidates, key=score)


def deduplicate(
    rows: Iterable[ProteinRow],
    evidence: Iterable[PairEvidence],
    *,
    order: Sequence[str],
    thresholds: Thresholds,
) -> DedupResult:
    """Deduplicate in a fixed quality order while preserving direct witnesses.

    Args:
        rows: The complete analysis universe.
        evidence: At most one measured record per unordered row pair. Missing
            evidence means "not proven redundant" and therefore cannot remove a
            row.
        order: Best-to-worst representative order, naming each row once.
        thresholds: Active sequence, coverage, structure and contact rules.

    Returns:
        Retained rows, direct witness removals, and the frozen sequence-only
        partition used to define structural rescue.
    """
    row_map: dict[str, ProteinRow] = {}
    for row in rows:
        if row.row_id in row_map:
            raise ValueError(f"duplicate row_id {row.row_id!r}")
        row_map[row.row_id] = row
    _validate_order(row_map, order)
    by_pair, by_row = _index_evidence(row_map, evidence)
    rank = {row_id: index for index, row_id in enumerate(order)}

    sequence_representatives: list[str] = []
    sequence_assignments: dict[str, str] = {}
    sequence_clusters: dict[str, list[str]] = {}
    for row_id in order:
        possible = []
        for pair in by_row[row_id]:
            other = pair.right if pair.left == row_id else pair.left
            if other in sequence_clusters and is_sequence_match(pair, thresholds):
                possible.append(other)
        if possible:
            representative = _best_witness(row_id, possible, by_pair)
        else:
            representative = row_id
            sequence_representatives.append(row_id)
            sequence_clusters[row_id] = []
        sequence_assignments[row_id] = representative
        sequence_clusters[representative].append(row_id)

    if not thresholds.structure_aware:
        removals = tuple(
            Removal(row_id, representative, representative)
            for row_id, representative in sequence_assignments.items()
            if row_id != representative
        )
        return DedupResult(
            kept=tuple(sequence_representatives),
            removals=removals,
            sequence_representatives=tuple(sequence_representatives),
            sequence_assignments=sequence_assignments,
        )

    kept: list[str] = []
    removals: list[Removal] = []
    for representative in sequence_representatives:
        modes: list[str] = []
        for row_id in sorted(sequence_clusters[representative], key=rank.__getitem__):
            possible = []
            for mode in modes:
                pair = by_pair.get(frozenset((row_id, mode)))
                if pair is not None and is_joint_match(pair, thresholds):
                    possible.append(mode)
            if possible:
                witness = _best_witness(row_id, possible, by_pair)
                removals.append(Removal(row_id, witness, representative))
            else:
                modes.append(row_id)
                kept.append(row_id)

    result = DedupResult(
        kept=tuple(kept),
        removals=tuple(removals),
        sequence_representatives=tuple(sequence_representatives),
        sequence_assignments=sequence_assignments,
    )
    validate_result(result, row_map, by_pair, thresholds)
    return result


def validate_result(
    result: DedupResult,
    rows: Mapping[str, ProteinRow],
    evidence: Mapping[frozenset[str], PairEvidence],
    thresholds: Thresholds,
) -> None:
    """Prove coverage and direct-witness invariants for a completed selection."""
    kept = set(result.kept)
    removed = {removal.row_id for removal in result.removals}
    if kept & removed or kept | removed != set(rows):
        raise ValueError("kept and removed rows must form a disjoint complete partition")
    for removal in result.removals:
        if removal.witness_id not in kept:
            raise ValueError(f"{removal.row_id}: witness {removal.witness_id} was not retained")
        pair = evidence.get(frozenset((removal.row_id, removal.witness_id)))
        if pair is None or not is_joint_match(pair, thresholds):
            raise ValueError(f"{removal.row_id}: witness does not clear the active thresholds")
