# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the statement-head-masked Qwen3 (#201 Phase 1b).

Checks the three things that could silently go wrong in wiring a custom loss
into levanter: that the subclass builds and runs at all, that the masked mean is
exactly the weighted mean over the surviving slots, and that the evaluation path
stays unmasked (so ``eval/.../loss`` remains comparable with #117/#150).

Run as for ``test_loss_masks.py``::

    PYTHONPATH=models:marinfold <exp-venv>/bin/python -m pytest models/tests -q
"""

import jax
import jax.numpy as jnp
import pytest

import haliax as hax
from levanter.models.lm_model import LmExample
from levanter.models.qwen import Qwen3Config

from marinfold import build_tokenizer
from marinfold.document_structures.contacts_v1 import vocab
from marinfold.document_structures.contacts_v1.generate import (
    GenerationConfig,
    build_document,
)
from marinfold.document_structures.contacts_v1.parse import RawContact, ResidueInfo
from marinfold_models.loss_masks import contacts_v1_statement_head_mask
from marinfold_models.masked_loss_model import _next_token_weight
from marinfold_models.masked_loss_model import (
    Qwen3StatementHeadMaskedConfig,
    Qwen3StatementHeadMaskedLMHeadModel,
)

_AA_CYCLE = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]
_PAIRS = [(0, 3), (0, 5), (1, 4), (1, 6), (2, 5), (3, 6), (0, 7), (2, 7), (4, 7)]

# A model small enough to run on CPU in a test, at the real contacts-v1 vocab
# (the mask keys off real token ids, so the vocab has to be the real one).
_SEQ_LEN = 128
_KWARGS = dict(
    max_seq_len=_SEQ_LEN, hidden_dim=32, intermediate_dim=64,
    num_heads=4, num_kv_heads=2, num_layers=2,
)


@pytest.fixture(scope="module")
def example() -> LmExample:
    tokenizer = build_tokenizer(vocab.all_domain_tokens())
    residues = [
        ResidueInfo(seq_index=i, resname=_AA_CYCLE[i % len(_AA_CYCLE)],
                    resnum=1 + i, chain="A")
        for i in range(9)
    ]
    contacts = [RawContact(i, j, 0.9 - 0.001 * k) for k, (i, j) in enumerate(_PAIRS)]
    tokens = build_document(
        "e0", residues, contacts, config=GenerationConfig(min_seq_separation=1)
    ).document.split()
    ids = [tokenizer.convert_tokens_to_ids(t) for t in tokens]
    # Pad to the model length with <pad>; loss_weight zeroes the padding.
    padding = _SEQ_LEN - len(ids)
    assert padding > 0, "test document must fit in the tiny model's context"
    Pos = hax.Axis("position", _SEQ_LEN)
    token_array = hax.named(jnp.asarray(ids + [0] * padding, dtype=jnp.int32), (Pos,))
    weight = hax.named(
        jnp.asarray([1.0] * (len(ids) - 1) + [0.0] * (padding + 1), dtype=jnp.float32),
        (Pos,),
    )
    return LmExample.causal(token_array, loss_weight=weight)


@pytest.fixture(scope="module")
def models():
    """A masked model and a plain Qwen3 sharing identical weights."""
    Vocab = hax.Axis("vocab", 2846)
    key = jax.random.PRNGKey(0)
    masked_config = Qwen3StatementHeadMaskedConfig(**_KWARGS)
    masked = Qwen3StatementHeadMaskedLMHeadModel.init(Vocab, masked_config, key=key)
    plain = Qwen3Config(**_KWARGS).model_type(masked.transformer, masked.embeddings, masked.lm_head)
    return masked, plain


def test_config_registers_and_resolves_its_model_type():
    config = Qwen3StatementHeadMaskedConfig(**_KWARGS)
    assert config.model_type is Qwen3StatementHeadMaskedLMHeadModel
    assert config.section_closer_ids == (config.begin_structure_id, config.end_id)


def test_masked_loss_is_the_weighted_mean_over_surviving_slots(models, example):
    masked_model, plain_model = models
    Pos = example.tokens.axes[-1]

    per_pos = plain_model.compute_next_token_loss(example, reduction=None, reduction_axis=())
    weight = _next_token_weight(Pos, example.loss_weight)
    keep = contacts_v1_statement_head_mask(example.tokens, dtype=weight.dtype)

    expected = hax.sum(per_pos * keep) / hax.sum(weight * keep)
    got = masked_model.compute_next_token_loss(example)
    assert float(got.scalar()) == pytest.approx(float(expected.scalar()), rel=1e-5)


def test_masked_loss_differs_from_the_unmasked_loss(models, example):
    # Guards against the mask silently being all-ones (e.g. wrong token ids),
    # which would make every assertion above pass while changing nothing.
    masked_model, plain_model = models
    masked = float(masked_model.compute_next_token_loss(example).scalar())
    unmasked = float(plain_model.compute_next_token_loss(example).scalar())
    assert abs(masked - unmasked) > 1e-3


def test_eval_path_is_unmasked_and_matches_plain_qwen3(models, example):
    # levanter's evaluator pairs this array with the UNMASKED loss_weight, so it
    # must be the standard per-position loss or eval/loss becomes a hybrid.
    masked_model, plain_model = models
    from_masked = masked_model.compute_next_token_loss(example, reduction=None, reduction_axis=())
    from_plain = plain_model.compute_next_token_loss(example, reduction=None, reduction_axis=())
    assert jnp.allclose(from_masked.array, from_plain.array)


def test_non_mean_reduction_is_rejected(models, example):
    masked_model, _ = models
    with pytest.raises(ValueError, match="reduction"):
        masked_model.compute_next_token_loss(example, reduction=hax.sum, reduction_axis="position")


def test_loss_is_jittable(models, example):
    masked_model, _ = models
    jitted = jax.jit(lambda m, e: m.compute_next_token_loss(e))
    assert jnp.isfinite(jitted(masked_model, example).array)
