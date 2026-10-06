"""Tests for leakage decisions whose mistakes would silently certify dirty targets."""

import pyarrow.parquet as pq
import pytest

from analyze_survival import contaminates
from build_candidates import foldbench_members
from build_complex_sequences import chain_sequences
from export_helico_contacts import mapped_rankings
from freeze_foldbench_eval import DATA, qualifying_alignment
from score_complex_rollout_worker import (
    candidate_pair_count,
    generation_token_budget,
    same_chain_too_close,
)
from score_foldbench_contacts import target_r_precision


def test_ring_wrap_and_shuffled_statements() -> None:
    document = (
        "<contacts-v1> <begin_sequence> <p12> <TRP> <p0> <GLY> "
        "<c-term> <p12> <p1999> <ALA> <n-term> <p1999> "
        "<c-term> <p0> <n-term> <p12> <begin_statements> <end>"
    )
    assert sorted(chain_sequences(document, [1, 2])) == ["AG", "W"]


def test_incomplete_document_is_not_a_shorter_training_sequence() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        chain_sequences(
            "<begin_sequence> <n-term> <p0> <p0> <ALA> <c-term> <p2> <begin_statements>",
            [3],
        )


def test_short_training_fragment_excludes_long_query() -> None:
    hit = {
        "nident": "30",
        "alnlen": "100",
        "qstart": "101",
        "qend": "200",
        "qlen": "1000",
        "tstart": "1",
        "tend": "100",
        "tlen": "100",
    }
    assert contaminates(hit)


def test_rounded_identity_does_not_create_a_false_exclusion() -> None:
    hit = {
        "nident": "29",
        "alnlen": "97",
        "qstart": "1",
        "qend": "97",
        "qlen": "100",
        "tstart": "1",
        "tend": "97",
        "tlen": "100",
    }
    assert not contaminates(hit)


def test_high_identity_short_motif_does_not_exclude_whole_protein() -> None:
    hit = {
        "nident": "20",
        "alnlen": "20",
        "qstart": "1",
        "qend": "20",
        "qlen": "200",
        "tstart": "1",
        "tend": "20",
        "tlen": "200",
    }
    assert not contaminates(hit)


def test_foldbench_symmetry_copy_labels_are_not_collapsed() -> None:
    rows = [
        {"interface_chain_id_1": "A", "interface_chain_id_2": "A-2"},
        {"interface_chain_id_1": "A-2", "interface_chain_id_2": "B"},
    ]
    assert foldbench_members(rows) == ["A", "A-2", "B"]


def test_exact_pair_homology_boundary() -> None:
    row = {
        "nident": "30",
        "alnlen": "100",
        "qstart": "1",
        "qend": "50",
        "qlen": "100",
        "tstart": "20",
        "tend": "69",
        "tlen": "200",
    }
    assert qualifying_alignment(row)
    row["nident"] = "29"
    assert not qualifying_alignment(row)


def test_frozen_complex_universe_is_cross_chain_and_group_clean() -> None:
    rows = pq.read_table(DATA / "foldbench_complex_eval_targets.parquet").to_pylist()
    assert len(rows) == 30
    assert sum(row["split"] == "dev" for row in rows) == 8
    splits_by_group: dict[str, set[str]] = {}
    for row in rows:
        splits_by_group.setdefault(row["group_id"], set()).add(row["split"])
        boundary = row["chain_offsets"][1]
        assert row["L"] == sum(row["chain_lengths"])
        assert row["n_gt"] == len(row["gt_contacts"])
        assert row["n_resolved_pairs"] == (
            len(row["resolved_positions_by_chain"][0])
            * len(row["resolved_positions_by_chain"][1])
        )
        for chain_index, (resolved, structure) in enumerate(
            zip(
                row["resolved_positions_by_chain"],
                row["structure_positions_by_chain"],
                strict=True,
            )
        ):
            chain_start = row["chain_offsets"][chain_index]
            chain_end = chain_start + row["chain_lengths"][chain_index]
            assert set(resolved) <= set(structure)
            assert all(chain_start <= position < chain_end for position in structure)
        assert all(i < boundary <= j for i, j in row["gt_contacts"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_contact_eval_excludes_context_failures_before_resplitting() -> None:
    rows = pq.read_table(
        DATA / "foldbench_complex_contact_eval_targets.parquet"
    ).to_pylist()
    assert len(rows) == 23
    assert sum(row["split"] == "dev" for row in rows) == 6
    assert sum(row["split"] == "test" for row in rows) == 17
    assert len({row["group_id"] for row in rows}) == 19
    assert sum(row["complex_type"] == "homodimer" for row in rows) == 6
    splits_by_group: dict[str, set[str]] = {}
    for row in rows:
        splits_by_group.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())


def test_inter_chain_r_precision_uses_only_resolved_interface_pairs() -> None:
    target = {
        "target_id": "example",
        "dataset": "test",
        "stem": "example",
        "split": "test",
        "group_id": "g-example",
        "complex_type": "heterodimer",
        "L": 4,
        "resolved_positions_by_chain": [[0, 1], [2, 3]],
        "n_resolved_pairs": 4,
        "gt_contacts": [[0, 2], [1, 3]],
    }
    scores = {(0, 3): 10, (0, 2): 9, (0, 1): 100}
    row = target_r_precision(target, scores)
    assert row["n_correct"] == 1
    assert row["r_precision"] == 0.5
    assert row["n_invalid_positive_predictions"] == 1


def test_helico_export_uses_chain_local_resolved_indices() -> None:
    target = {
        "chain_ids": ["A-2", "B"],
        "structure_positions_by_chain": [[1, 3, 4, 5, 6, 7, 8], [10, 12]],
    }
    all_contacts, intra_contacts = mapped_rankings(
        target,
        {
            (3, 12): 9,
            (1, 8): 8,
            (1, 3): 100,
            (0, 10): 100,
        },
    )
    assert all_contacts == [["A-2", 1, "B", 1], ["A-2", 0, "A-2", 6]]
    assert intra_contacts == [["A-2", 0, "A-2", 6]]


def test_complex_candidate_universe_applies_separation_within_chains_only() -> None:
    assert candidate_pair_count([10, 3]) == 10 + 0 + 30
    assert same_chain_too_close(8, 9, [10, 3])
    assert not same_chain_too_close(9, 10, [10, 3])
    assert not same_chain_too_close(0, 9, [10, 3])


def test_complex_rollout_budget_avoids_the_monomer_cap_regression() -> None:
    assert (
        generation_token_budget(
            prompt_tokens=1_000,
            length=346,
            contact_mult=12,
        )
        == 4_280
    )
    assert generation_token_budget(prompt_tokens=1_000, length=346) == 7_192
    assert generation_token_budget(prompt_tokens=2_000, length=1_152) == 6_192
