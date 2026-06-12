# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary + tokenizer tests for the contacts-v1 document structure.

These are pure (no pyconfind, no network) — they exercise the token list
and the shared ``build_tokenizer``. contacts-v1 mints only 5 tokens and
reuses everything else (positions, section markers, amino acids, ``<UNK>``,
``<end>``) from the contacts-and-distances-v1 vocab.
"""

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    all_domain_tokens as cd_v1_all_domain_tokens,
)
from marinfold.document_structures.contacts_v1.vocab import (
    BEGIN_SEQUENCE_TOKEN,
    BEGIN_STRUCTURE_TOKEN,
    END_TOKEN,
    NAME,
    NATIVE_TOKENS,
    NUM_POSITION_INDICES,
    SEQUENCE_ONLY_DOC_TYPE_TOKEN,
    SEQUENCE_ONLY_TOKENS,
    additional_tokens,
    all_domain_tokens,
    contacts_v1_native_tokens,
    position_token,
    sequence_only_tokens,
)


def test_name_and_index_space():
    assert NAME == "contacts-v1"
    assert NUM_POSITION_INDICES == 2000


def test_native_tokens_are_the_five_unique_ones():
    assert NATIVE_TOKENS == [
        "<contacts-v1>", "<n-term>", "<c-term>", "<contact>", "<think>",
    ]
    assert contacts_v1_native_tokens() == NATIVE_TOKENS


def test_reused_tokens_are_the_cd_v1_spellings():
    # Section markers + positions are reused from contacts-and-distances-v1,
    # NOT minted with contacts-v1's hyphen / <pos-N> spellings.
    assert BEGIN_SEQUENCE_TOKEN == "<begin_sequence>"
    assert BEGIN_STRUCTURE_TOKEN == "<begin_statements>"
    assert END_TOKEN == "<end>"
    assert position_token(0) == "<p0>"
    assert position_token(1999) == "<p1999>"


def test_no_minted_duplicates_of_reused_tokens():
    tokens = set(all_domain_tokens())
    # The old contacts-v1-only spellings must NOT exist anymore.
    for dead in ("<pos-0>", "<pos-1999>", "<begin-sequence>", "<begin-structure>"):
        assert dead not in tokens
    # The reused spellings (from c-and-d-v1) are present exactly once.
    doc = all_domain_tokens()
    for reused in ("<begin_sequence>", "<begin_statements>", "<p0>", "<p1999>",
                   "<ALA>", "<UNK>", "<end>"):
        assert reused in tokens
        assert doc.count(reused) == 1


def test_native_and_additional_are_disjoint():
    native = set(contacts_v1_native_tokens())
    extra = set(additional_tokens())
    assert native.isdisjoint(extra)
    # additional is the full c-and-d-v1 vocab (nothing removed — no overlap).
    assert additional_tokens() == cd_v1_all_domain_tokens()


def test_token_order_invariants():
    tokens = all_domain_tokens()
    # Native tokens lead, then the whole contacts-and-distances-v1 block.
    assert tokens[:5] == NATIVE_TOKENS
    assert tokens[0] == "<contacts-v1>"
    assert tokens[5] == "<contacts-and-distances-v1>"


def test_sequence_only_token_is_appended_last():
    # The sequence-only doc type is minted by contacts-v1 but appended after
    # the contacts-and-distances-v1 block (append-only), so it is NOT one of
    # the 5 native tokens and NOT part of the reused c-and-d-v1 block.
    tokens = all_domain_tokens()
    assert SEQUENCE_ONLY_DOC_TYPE_TOKEN == "<contacts-v1.sequence_only>"
    assert SEQUENCE_ONLY_TOKENS == [SEQUENCE_ONLY_DOC_TYPE_TOKEN]
    assert sequence_only_tokens() == [SEQUENCE_ONLY_DOC_TYPE_TOKEN]
    assert tokens[-1] == SEQUENCE_ONLY_DOC_TYPE_TOKEN
    assert SEQUENCE_ONLY_DOC_TYPE_TOKEN not in NATIVE_TOKENS
    assert SEQUENCE_ONLY_DOC_TYPE_TOKEN not in additional_tokens()
    # Dropping the trailing token recovers exactly the original native +
    # c-and-d-v1 vocabulary, in order — i.e. every pre-existing id is intact.
    assert tokens[:-1] == [*contacts_v1_native_tokens(), *additional_tokens()]


def test_sequence_only_token_takes_the_final_id_only():
    # Adding the token preserved every pre-existing id: the contacts-v1 doc
    # type is still id 2 and the c-and-d-v1 block still starts at id 7; the
    # new token simply occupies the final id.
    tok = build_tokenizer(all_domain_tokens())
    assert tok.convert_tokens_to_ids(SEQUENCE_ONLY_DOC_TYPE_TOKEN) == len(tok) - 1
    assert tok.convert_tokens_to_ids("<contacts-v1>") == 2
    assert tok.convert_tokens_to_ids("<contacts-and-distances-v1>") == 7


def test_tokens_unique():
    tokens = all_domain_tokens()
    assert len(tokens) == len(set(tokens))


def test_domain_token_count():
    # 5 native + the full 2838-token contacts-and-distances-v1 vocab + the 1
    # trailing sequence-only token.
    assert len(all_domain_tokens()) == 5 + len(cd_v1_all_domain_tokens()) + 1 == 2844


def test_build_tokenizer_size_and_specials():
    tok = build_tokenizer(all_domain_tokens())
    assert len(tok) == 2846  # 2844 domain + <pad> + <eos>
    assert tok.convert_tokens_to_ids("<pad>") == 0
    assert tok.convert_tokens_to_ids("<eos>") == 1
    assert tok.convert_tokens_to_ids("<contacts-v1>") == 2


def test_build_tokenizer_roundtrip_sample():
    tok = build_tokenizer(all_domain_tokens())
    sample = (
        "<contacts-v1> <begin_sequence> "
        "<p22> <PHE> <n-term> <p20> <p21> <ALA> "
        "<c-term> <p22> <p20> <ALA> "
        "<begin_statements> "
        "<contact> <p20> <p21> <contact> <p22> <p21> "
        "<end>"
    )
    ids = tok.encode(sample, add_special_tokens=False)
    assert len(ids) == len(sample.split())
    unk_id = tok.convert_tokens_to_ids("<UNK>")
    assert unk_id not in ids


def test_position_tokens_shared_with_cd_v1():
    # A contacts-v1 position token and the same c-and-d-v1 token map to one id.
    tok = build_tokenizer(all_domain_tokens())
    assert tok.convert_tokens_to_ids("<p22>") == tok.convert_tokens_to_ids(
        position_token(22)
    )
    # <p2000>..<p2700> exist (from c-and-d-v1) but are unused by contacts-v1.
    assert tok.convert_tokens_to_ids("<p2700>") != tok.unk_token_id


def test_unk_token_decodes():
    tok = build_tokenizer(all_domain_tokens())
    ids = tok.encode("<UNK>", add_special_tokens=False)
    assert tok.decode(ids).strip() == "<UNK>"
