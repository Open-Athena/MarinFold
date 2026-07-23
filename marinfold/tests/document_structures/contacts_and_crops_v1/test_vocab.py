# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary + tokenizer tests for contacts-and-crops-v1.

Pure (no pyconfind, no network). The load-bearing claims: the inherited
contacts-v1 block is a byte-identical *prefix* of this format's vocab (so
every inherited id is unchanged), and this format is literally ccoord's
native block (doc type, then the 1000 xyz tokens) with one new ``<crop>``
token appended last — so the ``<xyz-DDD>`` ids match ccoord's exactly and a
ccoord checkpoint warm-starts by appending a single row.
"""

import pytest

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_and_distances_v1.vocab import ATOM_NAMES
from marinfold.document_structures.contacts_v1.vocab import (
    all_domain_tokens as contacts_v1_all_domain_tokens,
)
from marinfold.document_structures.contacts_and_coordinates_v1.vocab import (
    all_domain_tokens as ccoord_all_domain_tokens,
)
from marinfold.document_structures.contacts_and_crops_v1.vocab import (
    CONTEXT_LENGTH,
    CROP_TOKEN,
    DOC_TYPE_TOKEN,
    NAME,
    NUM_POSITION_INDICES,
    NUM_XYZ_TOKENS,
    XYZ_TOKENS,
    all_domain_tokens,
    atom_token,
    inherited_tokens,
    native_tokens,
    xyz_token,
    xyz_token_for_digits,
)


def test_name_and_constants():
    assert NAME == "contacts-and-crops-v1"
    assert CONTEXT_LENGTH == 8192
    assert NUM_POSITION_INDICES == 2000
    assert NUM_XYZ_TOKENS == 1000


def test_native_tokens_are_doc_type_then_xyz_then_crop():
    native = native_tokens()
    assert len(native) == 1002
    assert native[0] == DOC_TYPE_TOKEN == "<contacts-and-crops-v1>"
    assert native[1:-1] == XYZ_TOKENS
    assert native[1] == "<xyz-000>"
    assert native[-2] == "<xyz-999>"
    assert native[-1] == CROP_TOKEN == "<crop>"


def test_xyz_token_formatting():
    assert xyz_token(0) == "<xyz-000>"
    assert xyz_token(210) == "<xyz-210>"
    assert xyz_token(999) == "<xyz-999>"
    assert xyz_token_for_digits(2, 1, 0) == "<xyz-210>"
    assert xyz_token_for_digits(0, 8, 0) == "<xyz-080>"


@pytest.mark.parametrize("bad", [-1, 1000, 5000])
def test_xyz_token_out_of_range(bad):
    with pytest.raises(ValueError):
        xyz_token(bad)


def test_inherited_block_is_contacts_v1_unchanged():
    assert inherited_tokens() == contacts_v1_all_domain_tokens()
    assert len(inherited_tokens()) == 2844


def test_all_domain_tokens_order_and_count():
    tokens = all_domain_tokens()
    inherited = inherited_tokens()
    # Inherited block is a byte-identical PREFIX (every inherited id stable).
    assert tokens[: len(inherited)] == inherited
    # Native block appended last.
    assert tokens[len(inherited):] == native_tokens()
    assert len(tokens) == 3846
    # No duplicates anywhere.
    assert len(set(tokens)) == len(tokens)


def test_contacts_v1_ids_are_byte_stable():
    # A contacts-v1 checkpoint warm-starts by appending rows: every one of
    # its domain-token ids is unchanged here.
    ours = all_domain_tokens()
    for i, tok in enumerate(contacts_v1_all_domain_tokens()):
        assert ours[i] == tok


def test_xyz_ids_match_ccoord_and_only_crop_is_new():
    # crops == ccoord's native block + <crop>. Every xyz token therefore
    # keeps the exact id it has in ccoord (both put the doc type at the same
    # position and the xyz block right after it), so a ccoord checkpoint's xyz
    # embeddings transfer at their own ids; <crop> is the single new row.
    ours = all_domain_tokens()
    ccoord = ccoord_all_domain_tokens()
    ours_ids = {tok: i for i, tok in enumerate(ours)}
    ccoord_ids = {tok: i for i, tok in enumerate(ccoord)}
    for tok in XYZ_TOKENS:
        assert ours_ids[tok] == ccoord_ids[tok]
    # crops' doc-type reuses ccoord's doc-type id slot (a different string at
    # the same id — a benign reuse on warm-start), and <crop> is the single
    # genuinely new row, appended after ccoord's whole vocab.
    assert ours_ids[DOC_TYPE_TOKEN] == ccoord_ids["<contacts-and-coordinates-v1>"]
    assert ours_ids[CROP_TOKEN] == len(ccoord)
    assert set(ours) - set(ccoord) == {DOC_TYPE_TOKEN, CROP_TOKEN}


def test_reused_atom_and_position_tokens_present_in_inherited_block():
    inherited = set(inherited_tokens())
    for name in ATOM_NAMES:
        assert atom_token(name) in inherited
    assert "<CA>" in inherited and "<CB>" in inherited


def test_native_block_disjoint_from_inherited():
    assert not (set(native_tokens()) & set(inherited_tokens()))


def test_tokenizer_roundtrips_native_tokens():
    tokenizer = build_tokenizer(all_domain_tokens())
    # 3846 domain tokens + <pad>/<eos>.
    assert len(tokenizer) == 3848
    sample = "<contacts-and-crops-v1> <crop> <xyz-129> <p26> <CA> <xyz-360>"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    assert tokenizer.decode(ids) == sample
