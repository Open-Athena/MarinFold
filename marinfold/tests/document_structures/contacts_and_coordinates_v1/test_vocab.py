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


def test_inherited_block_is_contacts_v1_minus_retraction():
    # The inherited block is contacts-v1's vocab with its later <retract>
    # extension (issue #158) removed — this format has no retraction, and
    # inheriting that trailing token would shift all 1001 coordinate ids.
    from marinfold.document_structures.contacts_v1.vocab import backtracking_tokens, retract_tokens

    full = contacts_v1_all_domain_tokens()
    retract = set(retract_tokens()) | set(backtracking_tokens())
    assert inherited_tokens() == [t for t in full if t not in retract]
    assert set(full) - set(inherited_tokens()) == retract
    assert len(inherited_tokens()) == 2844


def test_all_domain_tokens_order_and_count():
    from marinfold.document_structures.contacts_v1.vocab import backtracking_tokens, retract_tokens

    tokens = all_domain_tokens()
    inherited = inherited_tokens()
    # Inherited block is a byte-identical PREFIX (every inherited id stable).
    assert tokens[: len(inherited)] == inherited
    # Native (coordinate) block next, then contacts-v1's <retract> appended last.
    assert tokens[len(inherited):-2] == native_tokens()
    assert tokens[-2:] == [*retract_tokens(), *backtracking_tokens()]
    assert len(tokens) == 3847
    # No duplicates anywhere.
    assert len(set(tokens)) == len(tokens)


def test_retract_appended_last_leaves_coordinate_ids_fixed():
    # The superset carries <retract> and the #175 backtracking doc type for
    # retraction-bearing mixtures, both appended AFTER the coordinate block so
    # no coordinate id moves.
    from marinfold.document_structures.contacts_v1.vocab import (
        BACKTRACKING_DOC_TYPE_TOKEN,
        RETRACT_TOKEN,
    )

    tok = build_tokenizer(all_domain_tokens())
    assert tok.convert_tokens_to_ids(BACKTRACKING_DOC_TYPE_TOKEN) == len(tok) - 1
    assert tok.convert_tokens_to_ids(RETRACT_TOKEN) == len(tok) - 2
    # Coordinate native block is unmoved: doc type right after the 2844-token
    # inherited block (ids 0-1 are pad/eos), then the xyz run.
    assert tok.convert_tokens_to_ids(DOC_TYPE_TOKEN) == 2 + 2844
    assert tok.convert_tokens_to_ids("<xyz-000>") == 2 + 2844 + 1
    assert tok.convert_tokens_to_ids("<xyz-999>") == 2 + 2844 + 1000


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
    # 3846 domain tokens (incl. trailing <retract>) + <pad>/<eos>.
    assert len(tokenizer) == 3849
    sample = "<contacts-and-coordinates-v1> <p26> <CA> <xyz-129> <xyz-360> <retract>"
    ids = tokenizer.encode(sample, add_special_tokens=False)
    assert tokenizer.decode(ids) == sample
