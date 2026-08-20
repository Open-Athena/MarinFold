# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The retraction-mode doc type: append-only vocab + a token-0 switch (#175).

#160 trained on a 50:50 mixture whose two halves began with the *identical*
prefix, so the model could not condition on which mode it was generating in.
``<contacts-v1.backtracking>`` is that missing signal. These tests pin the two
properties that make it safe to add to a format with published checkpoints:

1. **It is append-only.** Every pre-existing token id — in contacts-v1 and in
   both coordinate supersets — is unchanged, so an existing checkpoint grows by
   one embedding row rather than needing a remap. The superset case is the one
   that can silently break: those vocabs build their inherited block by
   *filtering* contacts-v1's trailing tokens out and re-appending them, so a new
   trailing token that is not added to that filter lands inside the inherited
   block and shoves the whole xyz/crop block up by one.
2. **It changes nothing but token 0.** With the flag off, generation is
   byte-identical to before, so every existing corpus stays valid.
"""

import pytest

from marinfold.document_structures.contacts_and_coordinates_v1 import vocab as ccoord
from marinfold.document_structures.contacts_and_crops_v1 import vocab as crops
from marinfold.document_structures.contacts_v1 import (
    GenerationConfig,
    build_document,
    residues_from_sequence,
)
from marinfold.document_structures.contacts_v1 import vocab as cv1

TOKEN = "<contacts-v1.backtracking>"
SEQ = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR"


def _ids(tokens):
    return {t: i for i, t in enumerate(["<pad>", "<eos>", *tokens])}


def test_token_is_last_in_every_vocab_that_carries_it():
    for mod in (cv1, crops, ccoord):
        assert mod.all_domain_tokens()[-1] == TOKEN


def test_contacts_v1_ids_are_unchanged():
    """Every token that existed before #175 keeps its id."""
    ids = _ids(cv1.all_domain_tokens())
    assert ids["<contacts-v1>"] == 2
    assert ids["<retract>"] == len(ids) - 2      # still second-to-last
    assert ids[TOKEN] == len(ids) - 1


@pytest.mark.parametrize("mod,name", [(crops, "crops"), (ccoord, "ccoord")])
def test_superset_coordinate_block_does_not_move(mod, name):
    """The failure this test exists for: a trailing token shifting the xyz block.

    Both supersets exclude contacts-v1's trailing groups from what they inherit
    and re-append them at the end. If ``backtracking_tokens`` were left out of
    that filter, ``<xyz-000>`` and ``<crop>`` would each move up by one id and
    every published crops/ccoord checkpoint would be silently wrong.
    """
    ids = _ids(mod.all_domain_tokens())
    assert ids["<xyz-000>"] == 2847, f"{name}: xyz block moved"
    assert ids["<retract>"] == len(ids) - 2, f"{name}: <retract> moved"
    assert ids[TOKEN] == len(ids) - 1
    assert TOKEN not in set(mod.inherited_tokens())


def test_xyz_ids_stay_synced_between_the_two_coordinate_formats():
    crops_ids, ccoord_ids = _ids(crops.all_domain_tokens()), _ids(ccoord.all_domain_tokens())
    for k in (0, 1, 500, 999):
        assert crops_ids[f"<xyz-{k:03d}>"] == ccoord_ids[f"<xyz-{k:03d}>"]


def test_no_duplicates_introduced():
    for mod in (cv1, crops, ccoord):
        toks = mod.all_domain_tokens()
        assert len(set(toks)) == len(toks)


def test_flag_off_is_byte_identical():
    """Existing corpora must stay valid — the default path cannot move."""
    residues = residues_from_sequence(SEQ)
    a = build_document("x", residues, [], config=GenerationConfig())
    b = build_document("x", residues, [], config=GenerationConfig(backtracking=False))
    assert a.document == b.document
    assert a.document.split()[0] == "<contacts-v1>"


def test_flag_on_changes_only_token_zero():
    """The marker is conditioning, not content: nothing else may differ."""
    residues = residues_from_sequence(SEQ)
    plain = build_document("x", residues, [], config=GenerationConfig())
    marked = build_document("x", residues, [], config=GenerationConfig(backtracking=True))
    assert marked.document.split()[0] == TOKEN
    assert marked.document.split()[1:] == plain.document.split()[1:]


def test_backtracking_and_sequence_only_are_mutually_exclusive():
    residues = residues_from_sequence(SEQ)
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_document("x", residues, [],
                       config=GenerationConfig(sequence_only=True, backtracking=True))
