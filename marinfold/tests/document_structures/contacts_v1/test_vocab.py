# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary + tokenizer tests for the contacts-v1 document structure.

These are pure (no pyconfind, no network) — they exercise the token list
and the shared ``build_tokenizer``.
"""

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_and_distances_v1.vocab import (
    all_domain_tokens as cd_v1_all_domain_tokens,
)
from marinfold.document_structures.contacts_v1.vocab import (
    CONTROL_TOKENS,
    NAME,
    NUM_POSITION_INDICES,
    POSITION_TOKENS,
    additional_tokens,
    all_domain_tokens,
    contacts_v1_native_tokens,
)


def test_name_and_index_space():
    assert NAME == "contacts-v1"
    assert NUM_POSITION_INDICES == 2000
    assert len(POSITION_TOKENS) == 2000
    assert POSITION_TOKENS[0] == "<pos-0>"
    assert POSITION_TOKENS[-1] == "<pos-1999>"


def test_position_tokens_unpadded():
    # SPEC example writes <pos-22>, not <pos-0022>.
    assert "<pos-22>" in POSITION_TOKENS
    assert "<pos-0022>" not in POSITION_TOKENS


def test_token_order_invariants():
    tokens = all_domain_tokens()
    # Group 1 leads: native control, then positions, then <think>.
    assert tokens[0] == "<contacts-v1>"
    assert tokens[: len(CONTROL_TOKENS)] == CONTROL_TOKENS
    assert tokens[len(CONTROL_TOKENS)] == "<pos-0>"
    think_idx = len(CONTROL_TOKENS) + NUM_POSITION_INDICES
    assert tokens[think_idx] == "<think>"
    # Group 2 (the contacts-and-distances-v1 block) follows.
    assert tokens[think_idx + 1] == "<contacts-and-distances-v1>"


def test_tokens_unique_and_end_is_shared():
    tokens = all_domain_tokens()
    assert len(tokens) == len(set(tokens)), "domain tokens must be unique"
    # <end> appears in both groups but is deduped to one shared id.
    assert tokens.count("<end>") == 1


def test_reused_amino_acid_and_unk_tokens_present():
    tokens = set(all_domain_tokens())
    # Uppercase AA tokens are reused from contacts-and-distances-v1.
    for aa in ("<ALA>", "<MET>", "<PHE>", "<VAL>", "<GLY>"):
        assert aa in tokens
    assert "<UNK>" in tokens
    assert "<think>" in tokens
    # The lowercase form from the SPEC example is NOT used.
    assert "<ala>" not in tokens


def test_additional_tokens_are_cd_v1_minus_overlap():
    extra = additional_tokens()
    native = set(contacts_v1_native_tokens())
    # additional_tokens excludes anything already native (only <end> overlaps).
    assert not (set(extra) & native)
    cd_v1 = cd_v1_all_domain_tokens()
    assert set(extra) == set(cd_v1) - {"<end>"}
    assert len(extra) == len(cd_v1) - 1


def test_domain_token_count():
    # 7 control + 2000 positions + 1 <think> + (2838 - 1 deduped <end>).
    expected = (
        len(CONTROL_TOKENS)
        + NUM_POSITION_INDICES
        + 1
        + (len(cd_v1_all_domain_tokens()) - 1)
    )
    assert len(all_domain_tokens()) == expected == 4845


def test_build_tokenizer_size_and_specials():
    tok = build_tokenizer(all_domain_tokens())
    assert len(tok) == 4847  # 4845 domain + <pad> + <eos>
    assert tok.convert_tokens_to_ids("<pad>") == 0
    assert tok.convert_tokens_to_ids("<eos>") == 1
    assert tok.convert_tokens_to_ids("<contacts-v1>") == 2


def test_build_tokenizer_roundtrip_sample():
    tok = build_tokenizer(all_domain_tokens())
    sample = (
        "<contacts-v1> <begin-sequence> "
        "<pos-22> <PHE> <n-term> <pos-20> <pos-21> <ALA> "
        "<c-term> <pos-22> <pos-20> <ALA> "
        "<begin-structure> "
        "<contact> <pos-20> <pos-21> <contact> <pos-22> <pos-21> "
        "<end>"
    )
    ids = tok.encode(sample, add_special_tokens=False)
    assert len(ids) == len(sample.split())
    unk_id = tok.convert_tokens_to_ids("<UNK>")
    assert unk_id not in ids


def test_unk_token_decodes():
    tok = build_tokenizer(all_domain_tokens())
    ids = tok.encode("<UNK>", add_special_tokens=False)
    assert tok.decode(ids).strip() == "<UNK>"
