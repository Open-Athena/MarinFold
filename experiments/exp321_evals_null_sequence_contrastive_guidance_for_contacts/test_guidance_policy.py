"""Unit tests for the reference-free guidance policy."""

import pytest

from guidance_policy import (
    expects_position_token,
    null_sequence,
    parse_contacts,
    validate_prompt_pair,
)


def test_null_sequences_preserve_length_and_composition() -> None:
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    assert null_sequence(sequence, "polyala", "x") == "A" * len(sequence)
    assert null_sequence(sequence, "polylys", "x") == "K" * len(sequence)
    shuffled = null_sequence(sequence, "shuffle", "x")
    assert sorted(shuffled) == sorted(sequence)
    assert shuffled == null_sequence(sequence, "shuffle", "x")


def test_prompt_pair_may_only_change_amino_acids() -> None:
    validate_prompt_pair([1, 10, 2, 11], [1, 12, 2, 12], {10, 11, 12}, 2)
    with pytest.raises(ValueError, match="outside"):
        validate_prompt_pair([1, 10], [9, 12], {10, 12}, 1)


def test_position_guidance_state_machine() -> None:
    contact, p1, p2 = 5, 101, 102
    positions = {p1, p2}
    assert not expects_position_token([], contact, positions)
    assert expects_position_token([contact], contact, positions)
    assert expects_position_token([contact, p1], contact, positions)
    assert not expects_position_token([contact, p1, p2], contact, positions)


def test_parse_contacts_keeps_first_occurrence_and_statement_ratio() -> None:
    contact, p1, p2, p3 = 5, 101, 107, 120
    tokens = [contact, p1, p2, contact, p2, p1, contact, p1, p3]
    ratios = [0.1, 0.2, 0.3, 9.0, 9.0, 9.0, -0.1, 0.4, 0.7]
    contacts, scores, malformed = parse_contacts(
        tokens, ratios, contact, {p1: 0, p2: 6, p3: 19}
    )
    assert contacts == [[0, 6], [0, 19]]
    assert scores == pytest.approx([0.6, 1.0])
    assert malformed == 0
