# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the on-device soft targets (#201 Phase 2).

Strategy: pass an **identity** matrix as the output embedding. The construction
contracts the vocabulary into the embedding before accumulating, so with an
identity table the returned "direction" is literally the unnormalised weight
vector over the vocabulary — which can be compared element-by-element against the
plain-Python oracle in ``marinfold...contacts_v1.soft_targets``.

Run as for the other model tests::

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
from marinfold.document_structures.contacts_v1.soft_targets import (
    FIRST_ENDPOINT,
    SECOND_ENDPOINT,
    STATEMENT_HEAD,
    soft_targets,
)
from marinfold_models.soft_targets import slot_kinds, soft_target_directions

_AA_CYCLE = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]
_PAIRS = [(0, 3), (0, 5), (1, 4), (1, 6), (2, 5), (3, 6), (0, 7), (2, 7), (4, 7)]
_CFG = GenerationConfig(min_seq_separation=1)


@pytest.fixture(scope="module")
def token_to_id() -> dict[str, int]:
    tokenizer = build_tokenizer(vocab.all_domain_tokens())
    return {
        token: tokenizer.convert_tokens_to_ids(token)
        for token in ["<pad>", "<eos>", *vocab.all_domain_tokens()]
    }


@pytest.fixture(scope="module")
def vocab_size() -> int:
    return len(vocab.all_domain_tokens()) + 2


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


def _weights(ids: list[int], vocab_size: int):
    """Run the construction with an identity table -> raw weight vectors."""
    Pos = hax.Axis("position", len(ids))
    Vocab = hax.Axis("vocab", vocab_size)
    Embed = hax.Axis("embed", vocab_size)
    tokens = hax.named(jnp.asarray(ids, dtype=jnp.int32), (Pos,))
    identity = hax.named(jnp.eye(vocab_size, dtype=jnp.float32), (Vocab, Embed))
    direction, normalizer, is_soft = soft_target_directions(
        tokens, identity, Vocab=Vocab, Embed=Embed
    )
    return direction.array, normalizer.array, is_soft.array


def _oracle_distribution(tokens: list[str], token_to_id, vocab_size, kinds):
    """Oracle probabilities as dense vocab vectors, keyed by loss-weight slot."""
    out = {}
    for target in soft_targets(tokens):
        if target.kind not in kinds:
            continue
        dense = [0.0] * vocab_size
        for token, probability in zip(target.support, target.probs):
            dense[token_to_id[token]] = probability
        out[target.target_index - 1] = dense
    return out


@pytest.mark.parametrize("entry_id", ["a", "b", "c"])
@pytest.mark.parametrize("n_res", [2, 9, 30])
def test_soft_slots_match_the_oracle(entry_id, n_res, token_to_id, vocab_size):
    tokens = _document_tokens(entry_id, n_res=n_res)
    ids = [token_to_id[t] for t in tokens]
    direction, normalizer, is_soft = _weights(ids, vocab_size)

    expected = _oracle_distribution(
        tokens, token_to_id, vocab_size, {STATEMENT_HEAD, FIRST_ENDPOINT}
    )
    assert {i for i, flag in enumerate(is_soft.tolist()) if flag} == set(expected)
    for slot, want in expected.items():
        got = (direction[slot] / normalizer[slot]).tolist()
        assert got == pytest.approx(want, abs=1e-6), f"slot {slot}"


def test_second_endpoints_stay_hard(token_to_id, vocab_size):
    # v1 leaves them one-hot deliberately; they must not be flagged soft.
    tokens = _document_tokens("second")
    ids = [token_to_id[t] for t in tokens]
    _, _, is_soft = _weights(ids, vocab_size)
    second = {
        t.target_index - 1 for t in soft_targets(tokens) if t.kind == SECOND_ENDPOINT
    }
    assert second and not any(is_soft[slot] for slot in second)


def test_amino_acid_and_frame_slots_stay_hard(token_to_id, vocab_size):
    tokens = _document_tokens("hard")
    ids = [token_to_id[t] for t in tokens]
    _, _, is_soft = _weights(ids, vocab_size)
    soft_slots = {i for i, flag in enumerate(is_soft.tolist()) if flag}
    hard_slots = {
        t.target_index - 1
        for t in soft_targets(tokens)
        if t.kind not in (STATEMENT_HEAD, FIRST_ENDPOINT)
    }
    assert not (soft_slots & hard_slots)


def test_packed_window_of_several_documents(token_to_id, vocab_size):
    # The accumulations must stop at each document's edge; a leak across the
    # boundary shows up as extra mass in the earlier document's targets.
    docs = [_document_tokens("p0", n_res=9), _document_tokens("p1", n_res=15),
            _document_tokens("p2", n_res=6)]
    ids: list[int] = []
    expected: dict[int, list[float]] = {}
    for tokens in docs:
        offset = len(ids)
        for slot, dense in _oracle_distribution(
            tokens, token_to_id, vocab_size, {STATEMENT_HEAD, FIRST_ENDPOINT}
        ).items():
            expected[offset + slot] = dense
        ids += [token_to_id[t] for t in tokens]

    direction, normalizer, is_soft = _weights(ids, vocab_size)
    assert {i for i, flag in enumerate(is_soft.tolist()) if flag} == set(expected)
    for slot, want in expected.items():
        got = (direction[slot] / normalizer[slot]).tolist()
        assert got == pytest.approx(want, abs=1e-6), f"packed slot {slot}"


def test_sequence_only_documents(token_to_id, vocab_size):
    tokens = _document_tokens("seq", sequence_only=True)
    ids = [token_to_id[t] for t in tokens]
    direction, normalizer, is_soft = _weights(ids, vocab_size)
    expected = _oracle_distribution(
        tokens, token_to_id, vocab_size, {STATEMENT_HEAD, FIRST_ENDPOINT}
    )
    assert {i for i, flag in enumerate(is_soft.tolist()) if flag} == set(expected)
    for slot, want in expected.items():
        assert (direction[slot] / normalizer[slot]).tolist() == pytest.approx(want, abs=1e-6)


def test_slot_kinds_agree_with_the_oracle(token_to_id, vocab_size):
    tokens = _document_tokens("kinds", n_res=12)
    ids = [token_to_id[t] for t in tokens]
    Pos = hax.Axis("position", len(ids))
    kinds = slot_kinds(hax.named(jnp.asarray(ids, dtype=jnp.int32), (Pos,)))
    for name, kind in (("statement_head", STATEMENT_HEAD),
                       ("first_endpoint", FIRST_ENDPOINT),
                       ("second_endpoint", SECOND_ENDPOINT)):
        got = {i for i, flag in enumerate(kinds[name].array.tolist()) if flag}
        want = {t.target_index - 1 for t in soft_targets(tokens) if t.kind == kind}
        assert got == want, name
