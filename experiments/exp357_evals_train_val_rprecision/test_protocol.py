"""Small adversarial cases guarding the diagnostic's measurement semantics."""

import numpy as np

from protocol import (
    live_contacts,
    oracle_pairs,
    parse_training_document,
    stable_seed,
    vote_matrix,
)
from scoring import score_votes


def test_position_wrap_and_retraction() -> None:
    text = "<contact> <p1998> <p4> <contact> <p4> <p1998> <retract> <p1998> <p4> <contact> <p1999> <p6> <contact> <p1998> <p20>"
    assert live_contacts(text, 1998, 10) == [[1, 8]]


def test_serialized_ground_truth_roundtrip() -> None:
    row = dict(
        document="<contacts-v1> <begin_sequence> <p4> <VAL> <p1998> <ALA> <p1999> <ARG> <p0> <ASN> <p1> <ASP> <p2> <CYS> <p3> <GLN> <n-term> <p1998> <begin_statements> <contact> <p4> <p1998> <end>",
        entry_id="test",
        seq_len=7,
        n_term_index=1998,
        contacts_emitted=1,
        contacts_passing_min_degree=1,
        truncated=False,
    )
    assert parse_training_document(row) == ("ARNDCQV", [[0, 6]])


def test_capped_samples_cannot_vote() -> None:
    samples = [
        dict(finish_reason="length", contacts=[[0, 6]]),
        dict(finish_reason="stop", contacts=[[1, 7]]),
    ]
    matrix = vote_matrix(samples, 8)
    assert matrix[0, 6] == 0
    assert matrix[1, 7] == matrix[7, 1] == 1


def test_oracle_given_contacts_cannot_earn_credit() -> None:
    record = dict(
        dataset="test",
        stem="protein",
        L=10,
        resolved=list(range(10)),
        contacts=[[0, 6, 1], [1, 8, 1], [2, 9, 1], [0, 9, 1]],
    )
    supplied = oracle_pairs(record)
    samples = [dict(finish_reason="stop", contacts=supplied)]
    null = [dict(finish_reason="stop", contacts=[])]
    score = score_votes(record, samples, reduced=True)
    control = score_votes(record, null, reduced=True)
    actual = next(row for row in score if row["range"] == "all" and row["cut"] == "R")
    expected = next(
        row for row in control if row["range"] == "all" and row["cut"] == "R"
    )
    assert actual == expected
    assert actual["n_true"] == 2
    assert actual["n_candidate"] == 8


def test_perfect_remaining_prediction_gets_full_credit() -> None:
    record = dict(
        dataset="test",
        stem="protein",
        L=10,
        resolved=list(range(10)),
        contacts=[[0, 6, 1], [1, 8, 1], [2, 9, 1], [0, 9, 1]],
    )
    supplied = {tuple(pair) for pair in oracle_pairs(record)}
    remaining = [[i, j] for i, j, _ in record["contacts"] if (i, j) not in supplied]
    rows = score_votes(
        record, [dict(finish_reason="stop", contacts=remaining)], reduced=True
    )
    result = next(row for row in rows if row["range"] == "all" and row["cut"] == "R")
    assert result["precision"] == 1.0
    assert result["n_true"] == len(remaining)


def test_nested_budgets_and_seed_are_checkpoint_independent() -> None:
    samples = [
        dict(finish_reason="stop", contacts=[[0, 6]]),
        dict(finish_reason="stop", contacts=[[1, 7]]),
    ]
    assert np.all(vote_matrix(samples[:1], 8) <= vote_matrix(samples, 8))
    assert stable_seed("afdb_train", "p", 0, "sampling") != stable_seed(
        "afdb_train", "p", 1, "sampling"
    )
