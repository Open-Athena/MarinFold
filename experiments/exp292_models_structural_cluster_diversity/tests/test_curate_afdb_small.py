"""Tests for the untrained-cluster arm, where no anchor exists."""

import pytest

from curate_afdb_small import SELECTION_TIER, quality_filter, select_all


def row(
    entry_id: str,
    *,
    cluster: str = "cluster",
    plddt: float = 90.0,
    length: int = 120,
    noncanonical: int = 0,
    anchor: bool = False,
) -> dict:
    """Make one validated member of a never-trained structural cluster."""
    return {
        "entry_id": entry_id,
        "struct_cluster_id": cluster,
        "is_anchor": anchor,
        "sequence": "A" * (length - noncanonical) + "X" * noncanonical,
        "seq_len": length,
        "global_plddt": plddt,
        "noncanonical_residues": noncanonical,
    }


def test_every_member_is_a_candidate_and_no_cluster_is_disqualified() -> None:
    # The anchored arm drops a whole cluster when its anchor fails. Here a bad
    # member must only remove itself, because there is no anchor to invalidate
    # the rest of the cluster's comparisons.
    rows = [row("bad", plddt=79.0), row("good")]
    kept, rejected = quality_filter(rows)
    assert [item["entry_id"] for item in kept] == ["good"]
    assert [item["rejection_reason"] for item in rejected] == ["candidate_plddt"]


def test_candidate_rejection_reasons_are_specific() -> None:
    rows = [
        row("noncanonical", noncanonical=2),
        row("short", length=59),
        row("long", length=1001),
        row("dim", plddt=79.0),
        row("good"),
    ]
    kept, rejected = quality_filter(rows)
    assert [item["entry_id"] for item in kept] == ["good"]
    assert {item["entry_id"]: item["rejection_reason"] for item in rejected} == {
        "noncanonical": "noncanonical_candidate",
        "short": "candidate_length",
        "long": "candidate_length",
        "dim": "candidate_plddt",
    }


def test_an_anchor_in_this_arm_is_an_integrity_failure() -> None:
    with pytest.raises(ValueError, match="hold no anchor"):
        quality_filter([row("anchor", anchor=True)])


def test_source_length_disagreement_raises() -> None:
    broken = row("member")
    broken["seq_len"] = broken["seq_len"] + 1
    with pytest.raises(ValueError, match="source length changed"):
        quality_filter([broken])


def test_every_survivor_is_admitted_and_ranked_by_confidence() -> None:
    rows = [
        row("low", cluster="a", plddt=85.0),
        row("high", cluster="a", plddt=95.0),
        row("solo", cluster="b"),
    ]
    selected = select_all(rows)
    assert len(selected) == 3, "an unanchored cluster admits all of its members"
    by_id = {item["entry_id"]: item for item in selected}
    assert by_id["high"]["selection_rank"] == 1
    assert by_id["low"]["selection_rank"] == 2
    assert by_id["solo"]["selection_rank"] == 1
    assert {item["selection_tier"] for item in selected} == {SELECTION_TIER}


def test_structural_columns_are_null_rather_than_claimed() -> None:
    selected = select_all([row("solo")])
    assert selected[0]["strict_structural_diversity"] is None
    assert selected[0]["max_selected_tm"] is None
    assert selected[0]["structural_novelty"] is None
