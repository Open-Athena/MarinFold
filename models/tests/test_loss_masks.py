# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contacts-v1 statement-head loss mask (#201 Phase 1b).

The on-device mask is checked against the plain-Python oracle in
``marinfold...contacts_v1.soft_targets.statement_head_slots``, on real generated
documents including packed multi-document windows.

Needs a jax + levanter + marinfold environment. ``models/`` has no venv of its
own (its dep stack is the heavy marin/levanter/jax one, installed per
experiment), so run with an experiment venv that already has both, e.g.::

    PYTHONPATH=models:marinfold <exp-venv>/bin/python -m pytest models/tests -q
"""

import jax.numpy as jnp
import pytest

import haliax as hax

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1 import vocab
from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    build_document,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, ResidueInfo
from marinfold.document_structures.contacts_v1.soft_targets import statement_head_slots
from marinfold_models.loss_masks import (
    BEGIN_SEQUENCE_ID,
    BEGIN_STRUCTURE_ID,
    END_ID,
    contacts_v1_statement_head_mask,
)

_AA_CYCLE = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]
_CFG = GenerationConfig(min_seq_separation=1)
_PAIRS = [(0, 3), (0, 5), (1, 4), (1, 6), (2, 5), (3, 6), (0, 7), (2, 7), (4, 7)]


@pytest.fixture(scope="module")
def token_to_id() -> dict[str, int]:
    """Token -> id at the real contacts-v1 tokenizer (2846 entries)."""
    tokenizer = build_tokenizer(vocab.all_domain_tokens())
    return {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in ["<pad>", "<eos>", *vocab.all_domain_tokens()]
    }


def _document_tokens(entry_id: str, n_res: int = 9, **cfg) -> list[str]:
    residues = [
        ResidueInfo(seq_index=i, resname=_AA_CYCLE[i % len(_AA_CYCLE)],
                    resnum=1 + i, chain="A")
        for i in range(n_res)
    ]
    contacts = [
        RawContact(i, j, 0.9 - 0.001 * k)
        for k, (i, j) in enumerate(_PAIRS)
        if i < n_res and j < n_res
    ]
    config = GenerationConfig(min_seq_separation=1, **cfg) if cfg else _CFG
    return build_document(entry_id, residues, contacts, config=config).document.split()


def _mask_for(ids: list[int]) -> list[float]:
    Pos = hax.Axis("position", len(ids))
    tokens = hax.named(jnp.asarray(ids, dtype=jnp.int32), (Pos,))
    return contacts_v1_statement_head_mask(tokens).array.tolist()


def test_published_token_ids_match_the_module_defaults(token_to_id):
    # The mask defaults are hard-coded ids; if the tokenizer ever moves, every
    # masked run silently masks the wrong slots. Fail here instead.
    assert token_to_id[vocab.BEGIN_SEQUENCE_TOKEN] == BEGIN_SEQUENCE_ID
    assert token_to_id[vocab.BEGIN_STRUCTURE_TOKEN] == BEGIN_STRUCTURE_ID
    assert token_to_id[vocab.END_TOKEN] == END_ID


@pytest.mark.parametrize("entry_id", ["a", "b", "c", "d"])
@pytest.mark.parametrize("n_res", [2, 9, 40])
def test_mask_matches_the_python_oracle(entry_id, n_res, token_to_id):
    tokens = _document_tokens(entry_id, n_res=n_res)
    ids = [token_to_id[t] for t in tokens]
    mask = _mask_for(ids)
    expected_zero = set(statement_head_slots(tokens))
    assert {i for i, m in enumerate(mask) if m == 0.0} == expected_zero


def test_mask_matches_the_oracle_for_sequence_only_documents(token_to_id):
    tokens = _document_tokens("seq", sequence_only=True)
    ids = [token_to_id[t] for t in tokens]
    mask = _mask_for(ids)
    assert {i for i, m in enumerate(mask) if m == 0.0} == set(statement_head_slots(tokens))


@pytest.mark.parametrize("n_res", [2, 9, 40])
def test_exactly_one_slot_is_masked_per_sequence_statement(n_res, token_to_id):
    # The sequence section holds L residue statements plus <n-term> and <c-term>,
    # and each contributes exactly one head slot. (On the real corpus that is
    # ~L+2 out of ~5L tokens; the synthetic documents here have far fewer
    # contacts than residues, so the fraction is not representative -- the count
    # is.)
    tokens = _document_tokens("frac", n_res=n_res)
    mask = _mask_for([token_to_id[t] for t in tokens])
    assert mask.count(0.0) == n_res + 2


def test_the_begin_statements_slot_survives(token_to_id):
    # The last statement body sits at even offset and would be masked by the
    # parity rule alone -- but it predicts <begin_statements>, which is real
    # information. This is the rule's third clause.
    tokens = _document_tokens("closer")
    ids = [token_to_id[t] for t in tokens]
    mask = _mask_for(ids)
    closer_slot = tokens.index(vocab.BEGIN_STRUCTURE_TOKEN) - 1
    assert mask[closer_slot] == 1.0


def test_structure_section_is_never_masked(token_to_id):
    tokens = _document_tokens("struct")
    ids = [token_to_id[t] for t in tokens]
    mask = _mask_for(ids)
    start = tokens.index(vocab.BEGIN_STRUCTURE_TOKEN)
    assert all(m == 1.0 for m in mask[start:])


def test_packed_window_of_several_documents(token_to_id):
    # Documents are packed prefix-only into one 8192 window, so the mask has to
    # reset at each document with no per-document bookkeeping.
    docs = [_document_tokens("p0", n_res=9), _document_tokens("p1", n_res=20),
            _document_tokens("p2", n_res=5, sequence_only=True),
            _document_tokens("p3", n_res=14)]
    ids: list[int] = []
    expected: set[int] = set()
    for tokens in docs:
        offset = len(ids)
        expected |= {offset + i for i in statement_head_slots(tokens)}
        ids += [token_to_id[t] for t in tokens]
    mask = _mask_for(ids)
    assert {i for i, m in enumerate(mask) if m == 0.0} == expected


def test_window_starting_mid_document_masks_nothing(token_to_id):
    # Defensive: pack=True never splits a document, but a window that somehow
    # began after <begin_sequence> must fail safe rather than mask wrongly.
    tokens = _document_tokens("mid")
    ids = [token_to_id[t] for t in tokens]
    start = tokens.index(vocab.BEGIN_SEQUENCE_TOKEN) + 3
    mask = _mask_for(ids[start:])
    closer = tokens.index(vocab.BEGIN_STRUCTURE_TOKEN) - start
    assert all(m == 1.0 for m in mask[:closer])


def test_mask_is_batch_shaped(token_to_id):
    tokens = _document_tokens("batch")
    ids = [token_to_id[t] for t in tokens]
    Batch = hax.Axis("batch", 3)
    Pos = hax.Axis("position", len(ids))
    stacked = hax.named(jnp.tile(jnp.asarray(ids, dtype=jnp.int32), (3, 1)), (Batch, Pos))
    mask = contacts_v1_statement_head_mask(stacked)
    assert mask.axes == (Batch, Pos)
    rows = mask.array.tolist()
    assert rows[0] == rows[1] == rows[2] == _mask_for(ids)
