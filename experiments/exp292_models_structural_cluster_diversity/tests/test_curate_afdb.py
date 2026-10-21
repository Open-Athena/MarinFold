"""Filtering and structural-selection tests for the AFDB curation stage."""

import numpy as np
import pytest

from curate_afdb import (
    partition_after_screen,
    protein_from_arrays,
    quality_filter,
    select_cluster,
)


def helix(length: int, *, radius: float = 2.3, rise: float = 1.5) -> np.ndarray:
    """Make a regular C-alpha helix so alignments have real geometry."""
    turn = np.arange(length) * 1.75
    return np.stack(
        [radius * np.cos(turn), radius * np.sin(turn), rise * np.arange(length)],
        axis=1,
    )


def wide_helix(length: int) -> np.ndarray:
    """Make a wider helix of the same length as ``helix``.

    It aligns end to end against ``helix`` — full coverage on both chains, so
    the pair is structurally comparable — while superposing poorly enough to
    clear the 0.8 symmetric TM bound. That is the case the strict rule is meant
    to admit, as opposed to a fragment whose novelty is really low coverage.
    """
    return helix(length, radius=5.0)


def row(
    entry_id: str,
    *,
    cluster: str = "cluster",
    anchor: bool = False,
    plddt: float = 90.0,
    length: int = 120,
    noncanonical: int = 0,
    shape: str = "helix",
) -> dict:
    """Make one validated AFDB row with stored coordinates."""
    sequence = "A" * (length - noncanonical) + "X" * noncanonical
    coords = helix(length) if shape == "helix" else wide_helix(length)
    return {
        "entry_id": entry_id,
        "struct_cluster_id": cluster,
        "is_anchor": anchor,
        "sequence": sequence,
        "seq_len": length,
        "global_plddt": plddt,
        "noncanonical_residues": noncanonical,
        "reservoir_rank": 0 if anchor else 1,
        "coords": coords,
    }


def proteins_for(rows: list[dict]) -> dict:
    """Build the per-cluster structure map ``select_cluster`` expects."""
    return {
        item["entry_id"]: protein_from_arrays(
            item["sequence"],
            item["coords"],
            np.full(len(item["sequence"]), 95.0),
        )
        for item in rows
    }


def test_protein_from_arrays_rejects_mismatched_geometry() -> None:
    with pytest.raises(ValueError):
        protein_from_arrays("AAAA", helix(3), np.full(3, 90.0))
    with pytest.raises(ValueError):
        protein_from_arrays("AAAA", helix(4), np.full(3, 90.0))


def test_noncanonical_anchor_removes_whole_cluster() -> None:
    rows = [row("anchor", anchor=True, noncanonical=1), row("candidate")]
    kept, rejected = quality_filter(rows)
    assert kept == []
    assert {item["rejection_reason"] for item in rejected} == {
        "noncanonical_anchor_cluster"
    }


def test_low_confidence_anchor_removes_whole_cluster() -> None:
    rows = [row("anchor", anchor=True, plddt=79.9), row("candidate")]
    kept, rejected = quality_filter(rows)
    assert kept == []
    assert {item["rejection_reason"] for item in rejected} == {"anchor_source_quality"}


def test_candidate_rejection_reasons_are_specific() -> None:
    rows = [
        row("anchor", anchor=True),
        row("noncanonical", noncanonical=2),
        row("short", length=59),
        row("long", length=1001),
        row("dim", plddt=79.0),
        row("good"),
    ]
    kept, rejected = quality_filter(rows)
    assert [item["entry_id"] for item in kept] == ["anchor", "good"]
    assert {item["entry_id"]: item["rejection_reason"] for item in rejected} == {
        "noncanonical": "noncanonical_candidate",
        "short": "candidate_length",
        "long": "candidate_length",
        "dim": "candidate_plddt",
    }


def test_source_length_disagreement_raises() -> None:
    broken = row("anchor", anchor=True)
    broken["seq_len"] = broken["seq_len"] + 1
    with pytest.raises(ValueError, match="source length changed"):
        quality_filter([broken])


def test_cluster_without_anchor_raises() -> None:
    with pytest.raises(ValueError, match="no retained training anchor"):
        quality_filter([row("candidate")])


def test_partition_records_excluded_candidates_and_drops_empty_clusters() -> None:
    rows = [
        row("anchor-a", cluster="a", anchor=True),
        row("candidate-a", cluster="a"),
        row("anchor-b", cluster="b", anchor=True),
        row("candidate-b", cluster="b"),
    ]
    quality, rejected = quality_filter(rows)
    assert not rejected
    clusters, sequence_rejections = partition_after_screen(
        quality, {"candidate-b": {"identity": 0.42, "shorter_coverage": 0.9}}
    )
    assert [item["entry_id"] for item in sequence_rejections] == ["candidate-b"]
    assert [item["rejection_reason"] for item in sequence_rejections] == [
        "heldout_sequence"
    ]
    assert len(clusters) == 1
    assert {item["entry_id"] for item in clusters[0]} == {"anchor-a", "candidate-a"}


def test_small_cluster_fills_without_measuring_any_pair() -> None:
    rows = [row("anchor", anchor=True), row("one", plddt=82), row("two", plddt=91)]
    chosen, measured = select_cluster((rows, {}))
    assert measured == []
    assert [item["entry_id"] for item in chosen] == ["two", "one"]
    assert {item["selection_tier"] for item in chosen} == {"quality_fill"}


def test_structurally_distinct_candidate_is_selected_first() -> None:
    rows = [
        row("anchor", anchor=True),
        row("similar-a", plddt=97),
        row("similar-b", plddt=96),
        row("similar-c", plddt=95),
        row("distinct", plddt=81, shape="wide"),
    ]
    chosen, measured = select_cluster((rows, proteins_for(rows)))
    assert measured, "a cluster with a choice must measure candidate pairs"
    assert chosen[0]["entry_id"] == "distinct"
    assert chosen[0]["selection_tier"] == "structural_diversity"
    assert chosen[0]["max_selected_tm"] <= 0.8
    assert [item["selection_tier"] for item in chosen[1:]] == [
        "quality_fill",
        "quality_fill",
    ]
    assert [item["entry_id"] for item in chosen[1:]] == ["similar-a", "similar-b"]


def test_identical_candidates_never_claim_structural_diversity() -> None:
    rows = [
        row("anchor", anchor=True),
        row("copy-a", plddt=97),
        row("copy-b", plddt=96),
        row("copy-c", plddt=95),
        row("copy-d", plddt=94),
    ]
    chosen, measured = select_cluster((rows, proteins_for(rows)))
    assert len(chosen) == 3
    assert {item["selection_tier"] for item in chosen} == {"quality_fill"}
    assert all(item["max_selected_tm"] > 0.8 for item in chosen)
    assert measured, "the cluster had a choice, so pairs must be recorded"


def test_lazy_selection_skips_candidate_to_candidate_pairs() -> None:
    rows = [
        row("anchor", anchor=True),
        *[row(f"copy-{index}", plddt=90 + index) for index in range(5)],
    ]
    _, measured = select_cluster((rows, proteins_for(rows)))
    # Five candidates against one anchor; the all-pairs matrix would be 15.
    assert len(measured) == 5
    assert all(
        {pair["entry_a"], pair["entry_b"]} & {"anchor"} for pair in measured
    )
