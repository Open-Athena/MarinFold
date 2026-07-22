# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.layers.attention import AttentionBackend
from levanter.models.llama import LlamaConfig

from marinfold.document_structures.core import Vocabulary
from marinfold.document_structures.contacts_v1.training_documents import (
    unordered_contacts_score,
)
from marinfold.document_structures.documents import (
    QUERY,
    AttentionLayout,
    Coordinate,
    Document,
    ScoreContext,
    causal_training_document,
    pack,
)
from marinfold_models.scored_document import (
    levanter_scored_document_batch,
    scored_document_loss,
)


RELATIVE_POSITION = Coordinate("relative_position")


def _tiny_model(*, seq_len: int, vocab_size: int):
    config = LlamaConfig(
        max_seq_len=seq_len,
        hidden_dim=16,
        intermediate_dim=32,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        gradient_checkpointing=False,
        attn_backend=AttentionBackend.VANILLA,
    )
    return config.build(hax.Axis("vocab", vocab_size), key=jax.random.PRNGKey(0))


def test_levanter_loss_applies_document_scorer_after_one_logits_pass() -> None:
    context = Document(
        (3,),
        {RELATIVE_POSITION: (0,)},
        attention=AttentionLayout.FULL,
    ).unscored()
    queries = Document(
        (4, 4, 4, 4),
        {
            RELATIVE_POSITION: (0, 1, 2, 3),
            QUERY: (True, True, True, True),
        },
        attention=AttentionLayout.FULL,
    ).with_targets((5, 6, 7, 2))
    packed = pack((context + queries,), max_seq_len=5)
    batch = levanter_scored_document_batch(
        packed,
        Pos=hax.Axis("position", 5),
        position_coordinate=RELATIVE_POSITION,
    )
    model = _tiny_model(seq_len=5, vocab_size=16)

    actual = scored_document_loss(model, batch)
    logits = model(
        batch.tokens,
        batch.attention_mask,
        pos_ids=batch.position_ids,
    )
    expected = batch.score_ranges[0].scorer(
        logits.array[0, 1:5],
        ScoreContext(
            token_ids=batch.tokens.array[0, 1:5],
            target_ids=jnp.asarray((5, 6, 7, 2)),
        ),
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_levanter_unordered_contact_loss_is_jittable_and_differentiable() -> None:
    context = Document((3,), attention=AttentionLayout.FULL).unscored()
    queries = (
        Document((4,) * 7, attention=AttentionLayout.FULL)
        .with_targets((5, 6, 7, 5, 8, 9, 2))
        .scored_by(unordered_contacts_score)
    )
    batch = levanter_scored_document_batch(
        pack((context + queries,), max_seq_len=8),
        Pos=hax.Axis("position", 8),
    )
    model = _tiny_model(seq_len=8, vocab_size=16)

    loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(scored_document_loss))(
        model, batch
    )
    gradient_leaves = [leaf for leaf in jax.tree.leaves(grads) if eqx.is_array(leaf)]

    assert jnp.isfinite(loss)
    assert gradient_leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)
    assert any(bool(jnp.any(leaf != 0)) for leaf in gradient_leaves)


def test_levanter_loss_rejects_model_smaller_than_document_vocabulary() -> None:
    vocabulary = Vocabulary(
        "too-large", tuple(f"<token-{index}>" for index in range(20))
    )
    document = causal_training_document(
        (vocabulary.token("<token-0>"), vocabulary.token("<token-1>"))
    )
    batch = levanter_scored_document_batch(
        pack((document,), max_seq_len=2),
        Pos=hax.Axis("position", 2),
    )
    model = _tiny_model(seq_len=2, vocab_size=16)

    with pytest.raises(ValueError, match="documents use 'too-large' with 22 tokens"):
        scored_document_loss(model, batch)
