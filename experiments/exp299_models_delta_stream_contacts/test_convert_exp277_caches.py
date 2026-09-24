import random

import pytest

from convert_exp277_caches_to_delta_stream import (
    AA_BASE_TOKEN_ID,
    CONTACTS_BEGIN_TOKEN_ID,
    CV1_AA_FIRST,
    CV1_BEGIN_SEQUENCE,
    CV1_BEGIN_STATEMENTS,
    CV1_C_TERM,
    CV1_CONTACT,
    CV1_DOC_TYPE,
    CV1_END,
    CV1_EOS,
    CV1_N_TERM,
    CV1_NUM_POSITIONS,
    CV1_POSITION_FIRST,
    CV1_UNKNOWN_AA,
    DOC_END_TOKEN_ID,
    DOC_START_TOKEN_ID,
    EXTENDED_NEGATIVE_BASE,
    EXTENDED_POSITIVE_BASE,
    LEGACY_MAX_ABS_DELTA,
    MAX_ABS_DELTA,
    UNKNOWN_AA_TOKEN_ID,
    ParsedContactsV1,
    build_delta_stream_ids,
    delta_to_token,
    parse_contacts_v1_ids,
    parse_delta_stream_ids,
    token_to_delta,
)


def _position(index: int) -> int:
    return CV1_POSITION_FIRST + index


def _source_document() -> list[int]:
    # Canonical sequence begins at ring position 1998 and wraps through 0.
    statements = [
        (CV1_C_TERM, _position(1)),
        (_position(0), CV1_AA_FIRST + 2),
        (CV1_N_TERM, _position(1998)),
        (_position(1998), CV1_AA_FIRST),
        (_position(1), CV1_AA_FIRST + 3),
        (_position(1999), CV1_AA_FIRST + 1),
    ]
    random.Random(7).shuffle(statements)
    contacts = [
        (CV1_CONTACT, _position(1), _position(1998)),
        (CV1_CONTACT, _position(1999), _position(1)),
    ]
    return [
        CV1_DOC_TYPE,
        CV1_BEGIN_SEQUENCE,
        *(token for statement in statements for token in statement),
        CV1_BEGIN_STATEMENTS,
        *(token for statement in contacts for token in statement),
        CV1_END,
        CV1_EOS,
    ]


def test_contacts_v1_to_delta_stream_round_trip() -> None:
    parsed = parse_contacts_v1_ids(_source_document())
    assert parsed == ParsedContactsV1(
        amino_acids=(AA_BASE_TOKEN_ID, AA_BASE_TOKEN_ID + 1, AA_BASE_TOKEN_ID + 2, AA_BASE_TOKEN_ID + 3),
        contacts=((0, 3), (1, 3)),
    )
    converted = build_delta_stream_ids(parsed)
    assert converted[0] == DOC_START_TOKEN_ID
    assert converted[5] == CONTACTS_BEGIN_TOKEN_ID
    assert converted[-1] == DOC_END_TOKEN_ID
    assert parse_delta_stream_ids(converted) == parsed


@pytest.mark.parametrize("delta", [-1999, -1025, -1024, -1, 1, 1024, 1025, 1999])
def test_delta_extension_round_trip_and_legacy_stability(delta: int) -> None:
    token_id = delta_to_token(delta)
    assert token_to_delta(token_id) == delta
    if abs(delta) <= LEGACY_MAX_ABS_DELTA:
        expected = 32 + (abs(delta) - 1 if delta < 0 else 1024 + delta - 1)
        assert token_id == expected
    elif delta < 0:
        assert token_id >= EXTENDED_NEGATIVE_BASE
    else:
        assert token_id >= EXTENDED_POSITIVE_BASE


def test_maximum_length_chain_and_delta_are_representable() -> None:
    parsed = ParsedContactsV1(
        amino_acids=tuple([AA_BASE_TOKEN_ID] * CV1_NUM_POSITIONS),
        contacts=((0, MAX_ABS_DELTA),),
    )
    assert parse_delta_stream_ids(build_delta_stream_ids(parsed)) == parsed


def test_unknown_amino_acid_is_preserved() -> None:
    ids = _source_document()
    ids[ids.index(CV1_AA_FIRST + 2)] = CV1_UNKNOWN_AA
    parsed = parse_contacts_v1_ids(ids)
    assert UNKNOWN_AA_TOKEN_ID in parsed.amino_acids
    assert parse_delta_stream_ids(build_delta_stream_ids(parsed)) == parsed


def test_source_rejects_inconsistent_termini() -> None:
    ids = _source_document()
    c_term_index = ids.index(CV1_C_TERM) + 1
    ids[c_term_index] = _position(12)
    with pytest.raises(ValueError, match="termini"):
        parse_contacts_v1_ids(ids)


def test_source_rejects_duplicate_contact() -> None:
    ids = _source_document()
    ids[-2:-2] = [CV1_CONTACT, _position(1998), _position(1)]
    with pytest.raises(ValueError, match="duplicate source contact"):
        parse_contacts_v1_ids(ids)
