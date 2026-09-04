# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.llama import LlamaConfig

from marinfold.document_structures.contacts_v1.vocab import CONTACT, END
from marinfold_models.document_loss import CompactContactDocumentBatch, compact_contact_document_loss


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


def _compact_batch() -> CompactContactDocumentBatch:
    Batch = hax.Axis("batch", 2)
    Pos = hax.Axis("position", 16)
    tokens = hax.named(
        jnp.asarray(
            [
                [11, 12, int(CONTACT), 5, 6, int(CONTACT), 5, 7, int(END), 0, 0, 0, 0, 0, 0, 0],
                [21, int(CONTACT), 8, 9, int(CONTACT), 8, 10, int(CONTACT), 9, 10, int(END), 0, 0, 0, 0, 0],
            ],
            dtype=jnp.int32,
        ),
        (Batch, Pos),
    )
    position_ids = hax.named(
        jnp.broadcast_to(jnp.arange(Pos.size, dtype=jnp.int32), (Batch.size, Pos.size)),
        (Batch, Pos),
    )
    return CompactContactDocumentBatch(
        tokens=tokens,
        contact_first_ids=jnp.asarray([[5, 5, 0], [8, 8, 9]], dtype=jnp.int32),
        contact_second_ids=jnp.asarray([[6, 7, 0], [9, 10, 10]], dtype=jnp.int32),
        contact_count=jnp.asarray([2, 3], dtype=jnp.int32),
        prediction_start=jnp.asarray([1, 0], dtype=jnp.int32),
        position_ids=position_ids,
        attention_mask=AttentionMask.causal(),
        target_position_count=jnp.asarray([7, 10], dtype=jnp.int32),
        vocabulary=None,
    )


def _reference_compact_loss(log_probs: jax.Array, batch: CompactContactDocumentBatch) -> jnp.ndarray:
    contact_token_id = int(CONTACT)
    end_token_id = int(END)

    def one_example(log_probs_one, first_ids, second_ids, contact_count, prediction_start):
        total = jnp.asarray(0.0, dtype=log_probs_one.dtype)
        for contact_index in range(first_ids.shape[0]):
            valid = contact_index < contact_count
            contact_position = prediction_start + 1 + 3 * contact_index
            first_position = contact_position + 1
            second_position = contact_position + 2
            contact_predict_position = jnp.where(contact_index == 0, prediction_start, second_position - 3)

            contact_loss = -log_probs_one[contact_predict_position, contact_token_id]
            remaining = jnp.arange(first_ids.shape[0], dtype=jnp.int32) >= contact_index
            remaining = remaining & (jnp.arange(first_ids.shape[0], dtype=jnp.int32) < contact_count)
            first_loss_terms = -(
                log_probs_one[contact_position, first_ids]
                + log_probs_one[contact_position, second_ids]
            )
            first_loss = jnp.sum(jnp.where(remaining, first_loss_terms, 0.0)) / jnp.maximum(
                2 * (contact_count - contact_index), 1
            )

            actual_first = first_ids[contact_index]
            incident_first = remaining & (first_ids == actual_first)
            incident_second = remaining & (second_ids == actual_first)
            second_loss_terms = -(
                jnp.where(incident_first, log_probs_one[first_position, second_ids], 0.0)
                + jnp.where(incident_second, log_probs_one[first_position, first_ids], 0.0)
            )
            second_loss = jnp.sum(second_loss_terms) / jnp.maximum(
                jnp.sum(incident_first | incident_second), 1
            )
            total = total + jnp.where(valid, contact_loss + first_loss + second_loss, 0.0)

        end_position = prediction_start + 3 * contact_count
        return total - log_probs_one[end_position, end_token_id]

    losses = jax.vmap(one_example)(
        log_probs,
        batch.contact_first_ids,
        batch.contact_second_ids,
        batch.contact_count,
        batch.prediction_start,
    )
    return jnp.sum(losses) / jnp.sum(batch.target_position_count)


def test_compact_contact_loss_matches_enumerated_soft_targets() -> None:
    batch = _compact_batch()
    model = _tiny_model(seq_len=16, vocab_size=32)

    actual = compact_contact_document_loss(model, batch)
    logits = model(batch.tokens, batch.attention_mask, pos_ids=batch.position_ids)
    expected = _reference_compact_loss(jax.nn.log_softmax(logits.array, axis=-1), batch)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_compact_contact_loss_is_jittable_and_differentiable() -> None:
    batch = _compact_batch()
    model = _tiny_model(seq_len=16, vocab_size=32)

    loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(compact_contact_document_loss))(model, batch)
    gradient_leaves = [leaf for leaf in jax.tree.leaves(grads) if eqx.is_array(leaf)]

    assert jnp.isfinite(loss)
    assert gradient_leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in gradient_leaves)
    assert any(bool(jnp.any(leaf != 0)) for leaf in gradient_leaves)
