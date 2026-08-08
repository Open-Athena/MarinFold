# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the soft-target Qwen3 (#201 Phase 2).

The load-bearing check is ``test_soft_loss_matches_an_explicit_logit_computation``:
the model recovers ``logsumexp(z)`` from levanter's fused hard cross-entropy
rather than materialising logits, so it is compared against an independent
computation that *does* build the logits and apply the oracle's probabilities.

Run as for the other model tests::

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
from marinfold.document_structures.contacts_v1.soft_targets import (
    FIRST_ENDPOINT,
    STATEMENT_HEAD,
    soft_targets,
)
from marinfold_models.soft_loss_model import (
    Qwen3SoftTargetConfig,
    Qwen3SoftTargetLMHeadModel,
    _next_token_weight,
)

_AA_CYCLE = ["MET", "ALA", "GLY", "LYS", "PHE", "SER", "THR", "VAL", "LEU", "ILE"]
_PAIRS = [(0, 3), (0, 5), (1, 4), (1, 6), (2, 5), (3, 6), (0, 7), (2, 7), (4, 7)]
_SEQ_LEN = 128
_VOCAB = 2846
_KWARGS = dict(
    max_seq_len=_SEQ_LEN, hidden_dim=32, intermediate_dim=64,
    num_heads=4, num_kv_heads=2, num_layers=2,
)


@pytest.fixture(scope="module")
def document() -> list[str]:
    residues = [
        ResidueInfo(seq_index=i, resname=_AA_CYCLE[i % len(_AA_CYCLE)],
                    resnum=1 + i, chain="A")
        for i in range(9)
    ]
    contacts = [RawContact(i, j, 0.9 - 0.001 * k) for k, (i, j) in enumerate(_PAIRS)]
    return build_document(
        "e0", residues, contacts, config=GenerationConfig(min_seq_separation=1)
    ).document.split()


@pytest.fixture(scope="module")
def example(document) -> LmExample:
    tokenizer = build_tokenizer(vocab.all_domain_tokens())
    ids = [tokenizer.convert_tokens_to_ids(t) for t in document]
    padding = _SEQ_LEN - len(ids)
    assert padding > 0
    Pos = hax.Axis("position", _SEQ_LEN)
    tokens = hax.named(jnp.asarray(ids + [0] * padding, dtype=jnp.int32), (Pos,))
    weight = hax.named(
        jnp.asarray([1.0] * (len(ids) - 1) + [0.0] * (padding + 1), dtype=jnp.float32),
        (Pos,),
    )
    return LmExample.causal(tokens, loss_weight=weight)


@pytest.fixture(scope="module")
def models():
    """A soft-target model and a plain Qwen3 sharing identical weights."""
    Vocab = hax.Axis("vocab", _VOCAB)
    soft = Qwen3SoftTargetLMHeadModel.init(
        Vocab, Qwen3SoftTargetConfig(**_KWARGS), key=jax.random.PRNGKey(0)
    )
    plain = Qwen3Config(**_KWARGS).model_type(soft.transformer, soft.embeddings, soft.lm_head)
    return soft, plain


def test_config_registers_and_resolves_its_model_type():
    config = Qwen3SoftTargetConfig(**_KWARGS)
    assert config.model_type is Qwen3SoftTargetLMHeadModel


def test_eval_path_is_hard_and_matches_plain_qwen3(models, example):
    soft_model, plain_model = models
    from_soft = soft_model.compute_next_token_loss(example, reduction=None, reduction_axis=())
    from_plain = plain_model.compute_next_token_loss(example, reduction=None, reduction_axis=())
    assert jnp.allclose(from_soft.array, from_plain.array)


def test_soft_loss_matches_an_explicit_logit_computation(models, example, document):
    """Independent check: build the logits, apply the oracle's probabilities.

    The model never materialises logits -- it recovers logsumexp(z) as
    hard_ce + z[y]. This rebuilds the loss the slow, obvious way and requires
    the two to agree.
    """
    soft_model, plain_model = models
    tokenizer = build_tokenizer(vocab.all_domain_tokens())
    Pos = example.tokens.axes[-1]

    logits = plain_model(example.tokens, example.attn_mask)
    log_probs = hax.nn.log_softmax(logits, axis=soft_model.Vocab).array
    weight = _next_token_weight(Pos, example.loss_weight).array
    target_ids = jnp.roll(example.tokens.array, -1)

    # Hard loss everywhere, replaced by the oracle's soft loss where it applies.
    per_position = -log_probs[jnp.arange(_SEQ_LEN), target_ids]
    for target in soft_targets(document):
        if target.kind not in (STATEMENT_HEAD, FIRST_ENDPOINT):
            continue
        slot = target.target_index - 1
        value = 0.0
        for token, probability in zip(target.support, target.probs):
            value -= probability * float(log_probs[slot, tokenizer.convert_tokens_to_ids(token)])
        per_position = per_position.at[slot].set(value)

    expected = float((per_position * weight).sum() / weight.sum())
    got = float(soft_model.compute_next_token_loss(example).scalar())
    assert got == pytest.approx(expected, rel=1e-4)


def test_soft_loss_is_below_the_hard_loss_on_a_random_model(models, example):
    # At init the model is near-uniform, so spreading the target over a set can
    # only help: soft <= hard, and strictly so given the target entropy here.
    soft_model, plain_model = models
    soft = float(soft_model.compute_next_token_loss(example).scalar())
    hard = float(plain_model.compute_next_token_loss(example).scalar())
    assert soft < hard - 1e-3


def test_loss_is_jittable(models, example):
    soft_model, _ = models
    jitted = jax.jit(lambda m, e: m.compute_next_token_loss(e))
    assert jnp.isfinite(jitted(soft_model, example).array)


def test_non_mean_reduction_is_rejected(models, example):
    soft_model, _ = models
    with pytest.raises(ValueError, match="reduction"):
        soft_model.compute_next_token_loss(example, reduction=hax.sum, reduction_axis="position")
