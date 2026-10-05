"""Protect the scientific interpretation of alignment and selection."""

import numpy as np
import pytest

from structure_audit import (
    Protein,
    aligned_indices,
    compare,
    parse_structure,
    select_candidates,
)


def test_rigid_transform_preserves_structure_and_contact_overlap() -> None:
    rng = np.random.default_rng(292)
    coordinates = np.cumsum(rng.normal(size=(70, 3)) * 2, axis=0)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    a = Protein("A" * 70, coordinates, np.full(70, 90.0))
    b = Protein(
        "A" * 70,
        coordinates @ rotation.T + np.array([10.0, 20.0, 30.0]),
        np.full(70, 90.0),
    )
    result = compare(a, b)
    assert result["tm_a"] == pytest.approx(1.0, abs=1e-8)
    assert result["tm_b"] == pytest.approx(1.0, abs=1e-8)
    assert result["ca_contact_jaccard"] == 1.0
    assert result["coverage_a"] == 1.0


def test_gapped_alignment_preserves_original_residue_indices() -> None:
    a, b = aligned_indices("AB-CD-E", "A-BCDFE")
    assert a.tolist() == [0, 2, 3, 4]
    assert b.tolist() == [0, 2, 3, 5]


def test_fragment_is_not_a_diverse_fold_under_both_normalizations() -> None:
    rng = np.random.default_rng(71)
    coords = np.cumsum(rng.normal(size=(100, 3)), axis=0)
    a = Protein("A" * 100, coords, np.full(100, 90.0))
    b = Protein("A" * 40, coords[:40], np.full(40, 90.0))
    result = compare(a, b)
    assert result["tm_b"] > 0.99
    assert result["tm_max"] > 0.99
    assert result["coverage_a"] <= 0.4


@pytest.mark.parametrize("addition_tm,addition_coverage", [(0.98, 1.0), (0.5, 0.6)])
def test_candidate_must_differ_from_every_anchor_and_prior_additions(
    addition_tm: float, addition_coverage: float
) -> None:
    ids = ["rep", "anchor2", "copy_of_anchor2", "alternative", "alternative_copy"]
    rows = [
        {
            "entry_id": i,
            "struct_cluster_id": "cluster",
            "seq_len": "100",
            "is_anchor": str(i in ids[:2]),
        }
        for i in ids
    ]
    similarities = {
        ("rep", "anchor2"): 0.5,
        ("rep", "copy_of_anchor2"): 0.5,
        ("anchor2", "copy_of_anchor2"): 0.99,
        ("alternative", "alternative_copy"): addition_tm,
    }
    pairs = []
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            score = similarities.get((a, b), 0.6)
            pairs.append(
                {
                    "entry_a": a,
                    "entry_b": b,
                    "tm_max": score,
                    "coverage_a": addition_coverage
                    if (a, b) == ("alternative", "alternative_copy")
                    else 1.0,
                    "coverage_b": 1.0,
                    "core_tm_max": score,
                    "ca_contact_jaccard": 0.5,
                }
            )
    selected = select_candidates(rows, pairs)
    by_id = {r["entry_id"]: r for r in selected}
    assert by_id["copy_of_anchor2"]["selection_reason"] == "covered_by_training_anchor"
    assert sum(r["selected_order"] > 0 for r in selected) == 1
    assert by_id["alternative"]["selected_order"] == 1


def test_truncated_structure_fails_instead_of_becoming_shorter_candidate() -> None:
    pdb = b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C\nEND\n"
    with pytest.raises(ValueError, match="Expected 60 residues"):
        parse_structure(pdb, 60)


def test_confident_core_checks_anchor_other_than_whole_chain_nearest() -> None:
    rows = [
        {
            "entry_id": name,
            "struct_cluster_id": "c",
            "seq_len": "100",
            "is_anchor": str(name != "candidate"),
        }
        for name in ["a", "b", "candidate"]
    ]
    pairs = [
        {
            "entry_a": anchor,
            "entry_b": "candidate",
            "tm_max": tm,
            "core_tm_max": core,
            "coverage_a": 1.0,
            "coverage_b": 1.0,
            "ca_contact_jaccard": 0.5,
        }
        for anchor, tm, core in [("a", 0.7, 0.72), ("b", 0.6, 0.92)]
    ]
    candidate = select_candidates(rows, pairs)[0]
    assert candidate["nearest_anchor"] == "a"
    assert candidate["nearest_anchor_core_tm"] == 0.72
    assert candidate["max_anchor_core_tm"] == 0.92
