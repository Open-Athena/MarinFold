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
from marinfold.document_structures.documents import (
    ATTENTION_BLOCK,
    AttentionLayout,
    Coordinate,
    Document,
    causal_training_document,
    pack,
)
from marinfold_models.document_loss import (
    document_loss,
    levanter_document_batch,
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


def test_levanter_loss_applies_sparse_weighted_targets_after_one_forward() -> None:
    document = (
        Document(
            (3, 4, 5),
            {RELATIVE_POSITION: (0, 1, 2)},
            attention=AttentionLayout.FULL,
        )
        .unscored()
        .with_target_distribution(
            (6, 7, 8),
            ((4.0, 1.0, 0.0), (0.0, 2.0, 3.0)),
            start=1,
        )
    )
    batch = levanter_document_batch(
        pack((document,), max_seq_len=3),
        Pos=hax.Axis("position", 3),
        position_coordinate=RELATIVE_POSITION,
    )
    model = _tiny_model(seq_len=3, vocab_size=16)

    actual = document_loss(model, batch)
    logits = model(
        batch.tokens,
        batch.attention_mask,
        pos_ids=batch.position_ids,
    )
    log_probs = jax.nn.log_softmax(logits.array, axis=-1)
    expected = (
        -jnp.sum(
            batch.target_weights
            * log_probs[
                batch.target_rows,
                batch.target_positions,
                batch.target_ids,
            ]
        )
        / batch.target_position_count
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert batch.target_position_count == 2
    np.testing.assert_allclose(
        np.asarray(batch.target_weights),
        np.asarray((0.8, 0.2, 0.4, 0.6)),
    )


def test_levanter_weighted_target_loss_is_jittable_and_differentiable() -> None:
    document = (
        Document(
            (3, 4, 5, 6),
            {ATTENTION_BLOCK: (0, 1, 2, 3)},
            attention=AttentionLayout.BLOCK_CAUSAL,
        )
        .unscored()
        .with_target_distribution(
            (7, 8, 9),
            ((4.0, 1.0, 0.0), (0.0, 2.0, 3.0), (1.0, 0.0, 0.0)),
            start=0,
        )
    )
    batch = levanter_document_batch(
        pack((document,), max_seq_len=4),
        Pos=hax.Axis("position", 4),
    )
    model = _tiny_model(seq_len=4, vocab_size=16)

    loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(document_loss))(model, batch)
    gradient_leaves = [leaf for leaf in jax.tree.leaves(grads) if eqx.is_array(leaf)]

    assert jnp.isfinite(loss)
    assert gradient_leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)
    assert any(bool(jnp.any(leaf != 0)) for leaf in gradient_leaves)


def test_levanter_materializes_block_causal_attention_with_segment_isolation() -> None:
    document = Document(
        (3, 4, 5, 6),
        {ATTENTION_BLOCK: (0, 1, 1, 2)},
        attention=AttentionLayout.BLOCK_CAUSAL,
    ).with_targets((7, 8, 9), start=1)
    Pos = hax.Axis("position", 4)
    batch = levanter_document_batch(pack((document,), max_seq_len=4), Pos=Pos)

    materialized = batch.attention_mask.materialize(Pos, hax.Axis("key_position", 4))

    assert materialized is not None
    np.testing.assert_array_equal(
        materialized.rearrange(
            (hax.Axis("batch", 1), Pos, hax.Axis("key_position", 4))
        ).array[0],
        np.asarray(
            (
                (True, False, False, False),
                (True, True, True, False),
                (True, True, True, False),
                (True, True, True, True),
            )
        ),
    )


def test_levanter_lowers_implicit_shifted_targets() -> None:
    document = causal_training_document((3, 4, 5))
    batch = levanter_document_batch(
        pack((document,), max_seq_len=3),
        Pos=hax.Axis("position", 3),
    )

    assert batch.target_position_count == 2
    np.testing.assert_array_equal(np.asarray(batch.target_positions), (0, 1))
    np.testing.assert_array_equal(np.asarray(batch.target_ids), (4, 5))
    np.testing.assert_array_equal(np.asarray(batch.target_weights), (1.0, 1.0))


def test_levanter_loss_rejects_model_smaller_than_document_vocabulary() -> None:
    vocabulary = Vocabulary(
        "too-large", tuple(f"<token-{index}>" for index in range(20))
    )
    document = causal_training_document(
        (vocabulary.token("<token-0>"), vocabulary.token("<token-1>"))
    )
    batch = levanter_document_batch(
        pack((document,), max_seq_len=2),
        Pos=hax.Axis("position", 2),
    )
    model = _tiny_model(seq_len=2, vocab_size=16)

    with pytest.raises(ValueError, match="documents use 'too-large' with 22 tokens"):
        document_loss(model, batch)
