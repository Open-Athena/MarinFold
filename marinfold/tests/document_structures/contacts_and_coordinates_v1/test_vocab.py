# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Vocabulary + tokenizer tests for contacts-and-coordinates-v1.

Pure (no pyconfind, no network). The load-bearing claim: the inherited
contacts-v1 block is a byte-identical *prefix* of this format's vocab (so
every inherited id is unchanged), and the 1001 native tokens (doc type +
1000 xyz) are appended last.
"""

import pytest

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_and_distances_v1.vocab import ATOM_NAMES
from marinfold.document_structures.contacts_v1.vocab import (
    all_domain_tokens as contacts_v1_all_domain_tokens,
)
from marinfold.document_structures.contacts_and_coordinates_v1.vocab import (
    CONTEXT_LENGTH,
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
    assert NAME == "contacts-and-coordinates-v1"
    assert CONTEXT_LENGTH == 32768
    assert NUM_POSITION_INDICES == 2000
    assert NUM_XYZ_TOKENS == 1000


def test_native_tokens_are_doc_type_then_xyz():
    native = native_tokens()
    assert len(native) == 1001
    assert native[0] == DOC_TYPE_TOKEN == "<contacts-and-coordinates-v1>"
    assert native[1:] == XYZ_TOKENS
    assert native[1] == "<xyz-000>"
    assert native[-1] == "<xyz-999>"


def test_xyz_token_formatting():
    assert xyz_token(0) == "<xyz-000>"
    assert xyz_token(7) == "<xyz-007>"
    assert xyz_token(210) == "<xyz-210>"
    assert xyz_token(999) == "<xyz-999>"
    # digit triple: hundreds=x, tens=y, ones=z.
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
    assert len(tokens) == 3845
    # No duplicates anywhere.
    assert len(set(tokens)) == len(tokens)


def test_reused_atom_and_position_tokens_present_in_inherited_block():
    inherited = set(inherited_tokens())
    # Coordinate statements emit atom-name tokens; they must already exist.
    for name in ATOM_NAMES:
        assert atom_token(name) in inherited
    assert "<CA>" in inherited and "<CB>" in inherited


def test_native_block_disjoint_from_inherited():
    assert not (set(native_tokens()) & set(inherited_tokens()))


def test_tokenizer_roundtrips_native_tokens():
    tokenizer = build_tokenizer(all_domain_tokens())
    # 3845 domain tokens + <pad>/<eos>.
    assert len(tokenizer) == 3847
    sample = "<contacts-and-coordinates-v1> <p26> <CA> <xyz-129> <xyz-360>"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    assert tokenizer.decode(ids) == sample
