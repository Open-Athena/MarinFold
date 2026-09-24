import pytest

from delta_stream_rollout import (
    CONTACTS_BEGIN_TOKEN_ID,
    DELTA_BASE_TOKEN_ID,
    DOC_END_TOKEN_ID,
    DOC_START_TOKEN_ID,
    STOP_TOKEN_ID,
    UNKNOWN_AA_TOKEN_ID,
    parse_contact_suffix,
    sequence_prefix,
)


def test_sequence_prefix() -> None:
    assert sequence_prefix("ACV") == [DOC_START_TOKEN_ID, 1, 5, 20, CONTACTS_BEGIN_TOKEN_ID]
    assert sequence_prefix("X") == [DOC_START_TOKEN_ID, UNKNOWN_AA_TOKEN_ID, CONTACTS_BEGIN_TOKEN_ID]


def test_parse_suffix_deduplicates_directed_contact() -> None:
    # residue 0 -> 6 and residue 6 -> 0 encode one undirected contact.
    suffix = [
        DELTA_BASE_TOKEN_ID + 1024 + 5,
        STOP_TOKEN_ID,
        *([STOP_TOKEN_ID] * 5),
        DELTA_BASE_TOKEN_ID + 5,
        STOP_TOKEN_ID,
        DOC_END_TOKEN_ID,
    ]
    assert parse_contact_suffix(suffix, 7) == {(0, 6)}


def test_parse_suffix_rejects_incomplete_segments() -> None:
    with pytest.raises(ValueError, match="DOC_END"):
        parse_contact_suffix([STOP_TOKEN_ID, DOC_END_TOKEN_ID], 2)


def test_permissive_parse_ignores_invalid_statements() -> None:
    valid_delta_six = DELTA_BASE_TOKEN_ID + 1024 + 5
    invalid_delta_seven = DELTA_BASE_TOKEN_ID + 1024 + 6
    suffix = [valid_delta_six, invalid_delta_seven, 1, STOP_TOKEN_ID, DOC_END_TOKEN_ID]
    assert parse_contact_suffix(suffix, 7, strict=False) == {(0, 6)}
