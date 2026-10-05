# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from dedup import PairEvidence, ProteinRow, Thresholds, deduplicate


def row(row_id: str, tokens: int = 100) -> ProteinRow:
    return ProteinRow(row_id=row_id, source="test", source_tokens=tokens)


def pair(
    left: str,
    right: str,
    *,
    identity: float = 0.8,
    sequence_coverage: tuple[float, float] = (0.9, 0.9),
    tm: tuple[float, float] = (0.8, 0.8),
    structure_coverage: tuple[float, float] = (0.9, 0.9),
    contact_similarity: float | None = 0.8,
) -> PairEvidence:
    return PairEvidence(
        left=left,
        right=right,
        sequence_identity=identity,
        sequence_coverage_left=sequence_coverage[0],
        sequence_coverage_right=sequence_coverage[1],
        tm_left=tm[0],
        tm_right=tm[1],
        structure_coverage_left=structure_coverage[0],
        structure_coverage_right=structure_coverage[1],
        contact_similarity=contact_similarity,
    )


DEFAULT = Thresholds(min_sequence_identity=0.5, min_tm=0.7)


def test_requested_50pct_tm07_cell_removes_only_with_direct_witness():
    result = deduplicate(
        [row("a"), row("b")],
        [pair("a", "b", identity=0.5, tm=(0.7, 0.7))],
        order=["a", "b"],
        thresholds=DEFAULT,
    )
    assert result.kept == ("a",)
    assert [(r.row_id, r.witness_id) for r in result.removals] == [("b", "a")]


@pytest.mark.parametrize(
    ("evidence", "rescued"),
    [
        (pair("a", "b", identity=0.499), False),
        (pair("a", "b", sequence_coverage=(0.79, 1.0)), False),
        (pair("a", "b", tm=(0.69, 0.99)), True),
        (pair("a", "b", structure_coverage=(0.79, 1.0)), True),
    ],
)
def test_either_direction_below_a_threshold_preserves_the_row(evidence, rescued):
    result = deduplicate(
        [row("a"), row("b")], [evidence], order=["a", "b"], thresholds=DEFAULT
    )
    assert result.kept == ("a", "b")
    assert result.rescued_by_structure == (("b",) if rescued else ())


def test_sequence_similar_structure_different_is_explicitly_rescued():
    result = deduplicate(
        [row("a"), row("b")],
        [pair("a", "b", identity=0.95, tm=(0.45, 0.46))],
        order=["a", "b"],
        thresholds=DEFAULT,
    )
    assert result.sequence_representatives == ("a",)
    assert result.kept == ("a", "b")
    assert result.rescued_by_structure == ("b",)


def test_transitive_chain_cannot_remove_without_a_retained_direct_witness():
    # A matches B and B matches C, but A and C have no qualifying pair. B is
    # removed against A, so it is forbidden from serving as C's witness.
    result = deduplicate(
        [row("a"), row("b"), row("c")],
        [pair("a", "b"), pair("b", "c")],
        order=["a", "b", "c"],
        thresholds=DEFAULT,
    )
    assert result.kept == ("a", "c")
    assert [(r.row_id, r.witness_id) for r in result.removals] == [("b", "a")]


def test_structural_alternative_can_become_witness_for_its_mode():
    result = deduplicate(
        [row("a"), row("b"), row("c")],
        [
            pair("a", "b", tm=(0.4, 0.4)),
            pair("a", "c", tm=(0.4, 0.4)),
            pair("b", "c", tm=(0.9, 0.9)),
        ],
        order=["a", "b", "c"],
        thresholds=DEFAULT,
    )
    assert result.kept == ("a", "b")
    assert [(r.row_id, r.witness_id) for r in result.removals] == [("c", "b")]


def test_contact_requirement_is_an_additional_label_level_gate():
    result = deduplicate(
        [row("a"), row("b")],
        [pair("a", "b", contact_similarity=0.69)],
        order=["a", "b"],
        thresholds=Thresholds(
            min_sequence_identity=0.5,
            min_tm=0.7,
            min_contact_similarity=0.7,
        ),
    )
    assert result.kept == ("a", "b")


def test_sequence_only_partition_is_frozen_for_structural_rescue():
    evidence = [pair("a", "b", tm=(0.2, 0.2)), pair("b", "c", tm=(0.9, 0.9))]
    result = deduplicate(
        [row("a"), row("b"), row("c")],
        evidence,
        order=["a", "b", "c"],
        thresholds=DEFAULT,
    )
    # A-B form one sequence star. C cannot chain into it through B because B
    # was not a sequence representative when C was assigned.
    assert result.sequence_representatives == ("a", "c")
    assert result.kept == ("a", "b", "c")


def test_sequence_only_mode_keeps_one_direct_representative_per_star():
    result = deduplicate(
        [row("a"), row("b"), row("c")],
        [pair("a", "b"), pair("b", "c")],
        order=["a", "b", "c"],
        thresholds=Thresholds(min_sequence_identity=0.5),
    )
    assert result.kept == ("a", "c")
    assert [(r.row_id, r.witness_id) for r in result.removals] == [("b", "a")]


def test_rejects_duplicate_pair_records():
    with pytest.raises(ValueError, match="duplicate pair evidence"):
        deduplicate(
            [row("a"), row("b")],
            [pair("a", "b"), pair("b", "a")],
            order=["a", "b"],
            thresholds=DEFAULT,
        )


def test_rejects_an_incomplete_order():
    with pytest.raises(ValueError, match="every row exactly once"):
        deduplicate(
            [row("a"), row("b")], [], order=["a"], thresholds=DEFAULT
        )
