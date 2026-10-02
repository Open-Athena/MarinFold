"""Reject invalid model outputs instead of converting them into correct pairs."""

from rollout_validation import parse_contacts


def test_cyclic_output_positions_and_invalid_indices() -> None:
    text = (
        "<contact> <p4> <p1998> <contact> <p1998> <p4> "
        "<contact> <p4> <p3998> <contact> <p2> <p1998>"
    )
    pairs, invalid = parse_contacts(text, "contacts_v1", 7, 1998)
    assert pairs == {(1, 7)}
    assert invalid == 3


def test_plain_output_requires_two_numbers_on_one_line() -> None:
    text = "1 7\n7 1\n0 7\n1\n8\n1 3\nEND\n"
    pairs, invalid = parse_contacts(text, "prompted", 8, 0)
    assert pairs == {(1, 7)}
    assert invalid == 5


def test_malformed_statements_are_counted() -> None:
    pairs, invalid = parse_contacts(
        "<contact> <p0> <p6> <contact> <p-1> <p8> <end><|endoftext|>",
        "contacts_v1",
        10,
        0,
    )
    assert pairs == {(1, 7)}
    assert invalid == 1
