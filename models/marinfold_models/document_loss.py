# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter bridge for documents with sparse categorical targets."""

from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmHeadModel, split_activations
from levanter.models.loss import fused_cross_entropy_loss_and_logsumexp_penalty
from levanter.utils.jax_utils import local_cpu_mesh

from marinfold.document_structures.contacts_v1.vocab import CONTACT, END
from marinfold.document_structures.core import VocabularyIdentity
from marinfold.document_structures.documents import (
    ATTENTION_BLOCK,
    POSITION_IDS,
    QUERY,
    AttentionLayout,
    Coordinate,
    PackedBatch,
)


@dataclass(frozen=True)
class _FlatTargets:
    positions: np.ndarray
    token_ids: np.ndarray
    weights: np.ndarray
    position_count: int


class LevanterDocumentBatch(eqx.Module):
    """Named model inputs plus flattened sparse target distributions."""

    tokens: hax.NamedArray
    target_positions: jax.Array
    target_ids: jax.Array
    target_weights: jax.Array
    position_ids: hax.NamedArray
    attention_mask: AttentionMask
    target_position_count: jax.Array
    vocabulary: VocabularyIdentity | None = eqx.field(static=True)


class CompactContactDocumentBatch(eqx.Module):
    """Contacts-v1 block-causal examples with compact oracle targets."""

    tokens: hax.NamedArray
    contact_first_ids: jax.Array
    contact_second_ids: jax.Array
    contact_count: jax.Array
    prediction_start: jax.Array
    position_ids: hax.NamedArray
    segment_ids: hax.NamedArray
    attention_blocks: hax.NamedArray
    target_position_count: jax.Array
    vocabulary: VocabularyIdentity | None = eqx.field(static=True)


class SparseContactDocumentBatch(eqx.Module):
    """Contacts-v1 block-causal examples with sparse second-endpoint targets."""

    tokens: hax.NamedArray
    contact_first_ids: jax.Array
    contact_second_ids: jax.Array
    second_neighbor_ids: jax.Array
    second_neighbor_counts: jax.Array
    second_neighbor_count: jax.Array
    contact_count: jax.Array
    prediction_start: jax.Array
    position_ids: hax.NamedArray
    segment_ids: hax.NamedArray
    attention_blocks: hax.NamedArray
    target_position_count: jax.Array
    vocabulary: VocabularyIdentity | None = eqx.field(static=True)


def _flatten_targets(packed: PackedBatch) -> _FlatTargets:
    positions: list[int] = []
    token_ids: list[int] = []
    weights: list[float] = []
    position_count = 0

    for target_range in packed.score_ranges:
        if not target_range.scored:
            continue
        if target_range.row != 0:
            raise ValueError("Document examples must contain exactly one packed row")
        if target_range.target_ids is None:
            for position in range(target_range.start, target_range.stop - 1):
                positions.append(position)
                token_ids.append(int(packed.token_ids[target_range.row, position + 1]))
                weights.append(1.0)
                position_count += 1
            continue

        if target_range.target_weights is None:
            raise AssertionError("Explicit target range is missing weights")
        for relative_position, weight_row in enumerate(target_range.target_weights):
            nonzero = np.flatnonzero(weight_row)
            if nonzero.size == 0:
                raise AssertionError("Normalized target row unexpectedly has no mass")
            position = target_range.start + relative_position
            for target_index in nonzero:
                positions.append(position)
                token_ids.append(target_range.target_ids[int(target_index)])
                weights.append(float(weight_row[int(target_index)]))
            position_count += 1

    if position_count == 0:
        raise ValueError("Packed document batch has no scored target positions")
    return _FlatTargets(
        positions=np.asarray(positions, dtype=np.int32),
        token_ids=np.asarray(token_ids, dtype=np.int32),
        weights=np.asarray(weights, dtype=np.float32),
        position_count=position_count,
    )


def levanter_document_batch(
    packed: PackedBatch,
    *,
    Pos: hax.Axis,
    position_coordinate: Coordinate = POSITION_IDS,
    batch_axis_name: str = "batch",
    sparse_target_factor: int = 160,
) -> LevanterDocumentBatch:
    """Convert one packed document row and its weighted targets to a Levanter example."""
    del batch_axis_name
    if packed.token_ids.ndim != 2:
        raise ValueError(
            f"Packed document tokens must have rank 2, got {packed.token_ids.shape}"
        )
    if packed.token_ids.shape[0] != 1:
        raise ValueError(f"Document examples must contain exactly one row, got {packed.token_ids.shape[0]}")
    if packed.token_ids.shape[1] != Pos.size:
        raise ValueError(
            f"Packed sequence length {packed.token_ids.shape[1]} does not match "
            f"Levanter Pos axis size {Pos.size}"
        )

    axes = (Pos,)
    with local_cpu_mesh():
        tokens = hax.named(jnp.asarray(packed.token_ids[0]), axes)
        segment_ids = hax.named(jnp.asarray(packed.segment_ids[0]), axes)
        raw_position_ids = np.asarray(packed[position_coordinate])[0]
        position_ids = hax.named(jnp.asarray(np.maximum(raw_position_ids, 0)), axes)

        attention_mask = AttentionMask()
        if packed.attention == AttentionLayout.CAUSAL:
            attention_mask = AttentionMask.causal()
        elif packed.attention == AttentionLayout.BLOCK_CAUSAL:
            attention_blocks = hax.named(jnp.asarray(packed[ATTENTION_BLOCK][0]), axes)
            KPos = hax.Axis("key_position", Pos.size)
            key_blocks = attention_blocks.rename({Pos: KPos})
            explicit_mask = (
                attention_blocks.broadcast_axis(KPos) >= key_blocks.broadcast_axis(Pos)
            ).rearrange((Pos, KPos))
            attention_mask = AttentionMask.explicit(explicit_mask)
        attention_mask = attention_mask.with_segment_ids(segment_ids)

        targets = _flatten_targets(packed)
        max_targets = sparse_target_factor * Pos.size
        if targets.weights.shape[0] > max_targets:
            raise ValueError(
                f"Packed document has {targets.weights.shape[0]} sparse targets, "
                f"exceeding fixed budget {max_targets}"
            )

        padded_positions = np.zeros(max_targets, dtype=np.int32)
        padded_ids = np.zeros(max_targets, dtype=np.int32)
        padded_weights = np.zeros(max_targets, dtype=np.float32)
        target_count = targets.weights.shape[0]
        padded_positions[:target_count] = targets.positions
        padded_ids[:target_count] = targets.token_ids
        padded_weights[:target_count] = targets.weights

        return LevanterDocumentBatch(
            tokens=tokens,
            target_positions=jnp.asarray(padded_positions),
            target_ids=jnp.asarray(padded_ids),
            target_weights=jnp.asarray(padded_weights),
            target_position_count=jnp.asarray(targets.position_count),
            position_ids=position_ids,
            attention_mask=attention_mask,
            vocabulary=packed.vocabulary,
        )


def _model_logits(model: LmHeadModel, batch, *, key=None) -> jax.Array:
    logits = model(
        batch.tokens,
        batch.attention_mask,
        key=key,
        pos_ids=batch.position_ids,
    )
    if batch.vocabulary is not None and logits.array.shape[-1] < batch.vocabulary.size:
        raise ValueError(
            f"Model vocabulary has {logits.array.shape[-1]} logits, but documents use "
            f"{batch.vocabulary.name!r} with {batch.vocabulary.size} tokens"
        )
    return logits.array


def document_loss(
    model: LmHeadModel,
    batch: LevanterDocumentBatch,
    *,
    key=None,
) -> jnp.ndarray:
    """Run one model forward pass and apply weighted categorical cross-entropy."""
    logits = _model_logits(model, batch, key=key)
    batch_indices = jnp.arange(logits.shape[0])[:, None]
    selected_logits = logits[
        batch_indices,
        batch.target_positions,
        batch.target_ids,
    ]
    selected_rows = logits[batch_indices, batch.target_positions]
    selected = selected_logits - jax.nn.logsumexp(selected_rows, axis=-1)
    return -jnp.sum(batch.target_weights * selected) / jnp.sum(batch.target_position_count)


def compact_contact_document_batch(
    packed: PackedBatch,
    *,
    Pos: hax.Axis,
    max_contacts: int | None = None,
    position_coordinate: Coordinate = POSITION_IDS,
) -> CompactContactDocumentBatch:
    """Convert one contacts-v1 block-causal document to compact oracle targets."""
    if packed.token_ids.ndim != 2:
        raise ValueError(
            f"Packed document tokens must have rank 2, got {packed.token_ids.shape}"
        )
    if packed.token_ids.shape[0] != 1:
        raise ValueError(
            f"Compact contacts batches require one packed row, got {packed.token_ids.shape[0]}"
        )
    if packed.token_ids.shape[1] != Pos.size:
        raise ValueError(
            f"Packed sequence length {packed.token_ids.shape[1]} does not match "
            f"Levanter Pos axis size {Pos.size}"
        )
    query = np.asarray(packed[QUERY])[0]
    query_positions = np.flatnonzero(query)
    if query_positions.size == 0:
        raise ValueError("Compact contacts document has no query positions")
    prediction_start = int(query_positions[0])
    token_ids = np.asarray(packed.token_ids[0], dtype=np.int32)
    suffix = token_ids[prediction_start + 1 :]
    end_offsets = np.flatnonzero(suffix == int(END))
    if end_offsets.size == 0:
        raise ValueError("Compact contacts document suffix has no END token")
    suffix = suffix[: int(end_offsets[0])]
    if suffix.size % 3 != 0:
        raise ValueError(f"Contact suffix before END is not triples: {suffix.size} tokens")
    contact_count = suffix.size // 3
    if np.any(suffix[0::3] != int(CONTACT)):
        raise ValueError("Contact suffix triples do not start with CONTACT tokens")
    max_contact_count = max_contacts or ((Pos.size - 2) // 3)
    if contact_count > max_contact_count:
        raise ValueError(f"Document has {contact_count} contacts, exceeding compact budget {max_contact_count}")

    first_ids = np.zeros(max_contact_count, dtype=np.int32)
    second_ids = np.zeros(max_contact_count, dtype=np.int32)
    first_ids[:contact_count] = suffix[1::3]
    second_ids[:contact_count] = suffix[2::3]

    axes = (Pos,)
    with local_cpu_mesh():
        tokens = hax.named(jnp.asarray(token_ids), axes)
        segment_ids = hax.named(jnp.asarray(packed.segment_ids[0]), axes)
        raw_position_ids = np.asarray(packed[position_coordinate])[0]
        position_ids = hax.named(jnp.asarray(np.maximum(raw_position_ids, 0)), axes)

        if packed.attention != AttentionLayout.BLOCK_CAUSAL:
            raise ValueError(f"Compact contacts require block-causal attention, got {packed.attention}")
        attention_blocks = hax.named(jnp.asarray(packed[ATTENTION_BLOCK][0]), axes)

        return CompactContactDocumentBatch(
            tokens=tokens,
            contact_first_ids=jnp.asarray(first_ids),
            contact_second_ids=jnp.asarray(second_ids),
            contact_count=jnp.asarray(contact_count, dtype=jnp.int32),
            prediction_start=jnp.asarray(prediction_start, dtype=jnp.int32),
            position_ids=position_ids,
            segment_ids=segment_ids,
            attention_blocks=attention_blocks,
            target_position_count=jnp.asarray(3 * contact_count + 1, dtype=jnp.int32),
            vocabulary=packed.vocabulary,
        )


def _compact_contact_attention_mask(batch: CompactContactDocumentBatch) -> AttentionMask:
    """Build block-causal attention from compact per-token block ids."""
    Pos = batch.tokens.axes[-1]
    KPos = hax.Axis("key_position", Pos.size)
    query_blocks = batch.attention_blocks
    key_blocks = query_blocks.rename({Pos: KPos})
    explicit_mask = (query_blocks.broadcast_axis(KPos) >= key_blocks.broadcast_axis(Pos)).rearrange(
        (*query_blocks.axes, KPos)
    )
    return AttentionMask.explicit(explicit_mask).with_segment_ids(batch.segment_ids)


def _compact_contact_forward(model: LmHeadModel, batch: CompactContactDocumentBatch | SparseContactDocumentBatch, *, key=None):
    activations, aux_loss = split_activations(
        model.activations(
            batch.tokens,
            _compact_contact_attention_mask(batch),
            key=key,
            pos_ids=batch.position_ids,
        )
    )
    Pos = batch.tokens.axes[-1]
    lm_head = model.get_lm_head()
    target_y = hax.roll(batch.tokens, -1, Pos)
    hard_ce = fused_cross_entropy_loss_and_logsumexp_penalty(
        activations,
        lm_head,
        Contract=model.Embed,
        Label=model.Vocab,
        target_y=target_y,
        reduction=None,
        dtype=jnp.float32,
    )
    target_rows = lm_head.take(model.Vocab, target_y)
    z_target = hax.dot(activations, target_rows, axis=model.Embed)
    log_normalizers = hard_ce + z_target
    return activations, log_normalizers, lm_head, aux_loss, Pos


def compact_contact_document_loss(
    model: LmHeadModel,
    batch: CompactContactDocumentBatch,
    *,
    key=None,
) -> jnp.ndarray:
    """Contacts-v1 soft-target loss from compact endpoint lists.

    This is algebraically the same objective as the dense-logit version, but it
    computes ``CE(q, z) = logsumexp(z) - E_q[z]`` directly. ``logsumexp(z)`` is
    recovered from Levanter's fused linear-CE kernel, and ``E_q[z]`` is computed
    by dotting the position activation with weighted LM-head rows. That keeps the
    custom loss off the memory-heavy ``[batch, position, vocab]`` logits path.
    """
    activations, log_normalizers, lm_head, aux_loss, Pos = _compact_contact_forward(model, batch, key=key)

    activations_array = activations.rearrange((..., Pos, model.Embed)).array
    log_normalizers_array = log_normalizers.rearrange((..., Pos)).array
    lm_head_by_vocab = lm_head.rearrange((model.Vocab, model.Embed)).array

    max_contacts = batch.contact_first_ids.shape[-1]
    contact_token_id = jnp.asarray(int(CONTACT), dtype=jnp.int32)
    end_token_id = jnp.asarray(int(END), dtype=jnp.int32)
    contact_axis = jnp.arange(max_contacts, dtype=jnp.int32)

    def one_example_loss(activations_one, log_z_one, first_ids, second_ids, contact_count, prediction_start):
        def logit(position, token_ids):
            position = jnp.clip(position, 0, activations_one.shape[0] - 1)
            rows = lm_head_by_vocab[token_ids]
            return jnp.sum(activations_one[position] * rows, axis=-1)

        def cross_entropy(position, expected_logit):
            position = jnp.clip(position, 0, log_z_one.shape[0] - 1)
            return log_z_one[position] - expected_logit

        end_position = jnp.clip(prediction_start + 3 * contact_count, 0, log_z_one.shape[0] - 1)
        end_loss = cross_entropy(end_position, logit(end_position, end_token_id))

        def body(c, total):
            valid = c < contact_count
            contact_position = prediction_start + 1 + 3 * c
            first_position = contact_position + 1
            contact_predict_position = jnp.where(c == 0, prediction_start, contact_position - 1)

            remaining = (contact_axis >= c) & (contact_axis < contact_count)
            contact_loss = cross_entropy(contact_predict_position, logit(contact_predict_position, contact_token_id))

            first_endpoint_logits = logit(contact_position, first_ids) + logit(contact_position, second_ids)
            first_expected_logit = (
                jnp.sum(jnp.where(remaining, first_endpoint_logits, 0.0))
                / jnp.maximum(2 * (contact_count - c), 1)
            )
            first_loss = cross_entropy(contact_position, first_expected_logit)

            actual_first = first_ids[c]
            incident_first = remaining & (first_ids == actual_first)
            incident_second = remaining & (second_ids == actual_first)
            second_endpoint_logits = (
                jnp.where(incident_first, logit(first_position, second_ids), 0.0)
                + jnp.where(incident_second, logit(first_position, first_ids), 0.0)
            )
            second_expected_logit = (
                jnp.sum(second_endpoint_logits)
                / jnp.maximum(jnp.sum(incident_first | incident_second), 1)
            )
            second_loss = cross_entropy(first_position, second_expected_logit)

            return total + jnp.where(valid, contact_loss + first_loss + second_loss, 0.0)

        return jax.lax.fori_loop(0, max_contacts, jax.checkpoint(body), end_loss)

    losses = jax.vmap(one_example_loss)(
        activations_array,
        log_normalizers_array,
        batch.contact_first_ids,
        batch.contact_second_ids,
        batch.contact_count,
        batch.prediction_start,
    )
    return jnp.sum(losses) / jnp.sum(batch.target_position_count) + aux_loss


def sparse_contact_document_loss(
    model: LmHeadModel,
    batch: SparseContactDocumentBatch,
    *,
    key=None,
) -> jnp.ndarray:
    """Contacts-v1 soft-target loss using sparse neighbor rows.

    The first endpoint target is represented by a running sum of remaining
    endpoint LM-head embeddings. The second endpoint target is represented as a
    padded sparse row for each teacher contact step, avoiding the old
    per-contact scan over all remaining contacts.
    """
    activations, log_normalizers, lm_head, aux_loss, Pos = _compact_contact_forward(model, batch, key=key)

    activations_array = activations.rearrange((..., Pos, model.Embed)).array
    log_normalizers_array = log_normalizers.rearrange((..., Pos)).array
    lm_head_by_vocab = lm_head.rearrange((model.Vocab, model.Embed)).array

    contact_token_id = jnp.asarray(int(CONTACT), dtype=jnp.int32)
    end_token_id = jnp.asarray(int(END), dtype=jnp.int32)

    def one_example_loss(
        activations_one,
        log_z_one,
        first_ids,
        second_ids,
        second_neighbor_ids,
        second_neighbor_counts,
        second_neighbor_count,
        contact_count,
        prediction_start,
    ):
        contact_axis = jnp.arange(first_ids.shape[0], dtype=jnp.int32)
        valid_contacts = contact_axis < contact_count
        endpoint_rows = lm_head_by_vocab[first_ids] + lm_head_by_vocab[second_ids]
        endpoint_sum0 = jnp.sum(jnp.where(valid_contacts[:, None], endpoint_rows, 0.0), axis=0)

        def logit(position, token_id):
            position = jnp.clip(position, 0, activations_one.shape[0] - 1)
            return jnp.sum(activations_one[position] * lm_head_by_vocab[token_id], axis=-1)

        def cross_entropy(position, expected_logit):
            position = jnp.clip(position, 0, log_z_one.shape[0] - 1)
            return log_z_one[position] - expected_logit

        def body(c, carry):
            endpoint_sum, total = carry
            valid = c < contact_count
            contact_position = prediction_start + 1 + 3 * c
            first_position = contact_position + 1
            contact_predict_position = jnp.where(c == 0, prediction_start, contact_position - 1)

            contact_position = jnp.clip(contact_position, 0, activations_one.shape[0] - 1)
            first_position = jnp.clip(first_position, 0, activations_one.shape[0] - 1)
            contact_loss = cross_entropy(contact_predict_position, logit(contact_predict_position, contact_token_id))
            first_expected_logit = jnp.sum(activations_one[contact_position] * endpoint_sum) / jnp.maximum(
                2 * (contact_count - c), 1
            )
            first_loss = cross_entropy(contact_position, first_expected_logit)

            neighbor_rows = lm_head_by_vocab[second_neighbor_ids[c]]
            neighbor_logits = jnp.sum(activations_one[first_position] * neighbor_rows, axis=-1)
            second_expected_logit = jnp.sum(second_neighbor_counts[c] * neighbor_logits) / jnp.maximum(
                second_neighbor_count[c], 1
            )
            second_loss = cross_entropy(first_position, second_expected_logit)

            current_endpoint_rows = lm_head_by_vocab[first_ids[c]] + lm_head_by_vocab[second_ids[c]]
            next_endpoint_sum = endpoint_sum - jnp.where(valid, current_endpoint_rows, 0.0)
            next_total = total + jnp.where(valid, contact_loss + first_loss + second_loss, 0.0)
            return next_endpoint_sum, next_total

        _, body_loss = jax.lax.fori_loop(0, first_ids.shape[0], body, (endpoint_sum0, jnp.asarray(0.0, jnp.float32)))
        end_position = jnp.clip(prediction_start + 3 * contact_count, 0, log_z_one.shape[0] - 1)
        end_loss = cross_entropy(end_position, logit(end_position, end_token_id))
        return body_loss + end_loss

    losses = jax.vmap(one_example_loss)(
        activations_array,
        log_normalizers_array,
        batch.contact_first_ids,
        batch.contact_second_ids,
        batch.second_neighbor_ids,
        batch.second_neighbor_counts,
        batch.second_neighbor_count,
        batch.contact_count,
        batch.prediction_start,
    )
    return jnp.sum(losses) / jnp.sum(batch.target_position_count) + aux_loss


__all__ = [
    "CompactContactDocumentBatch",
    "LevanterDocumentBatch",
    "SparseContactDocumentBatch",
    "compact_contact_document_batch",
    "compact_contact_document_loss",
    "document_loss",
    "levanter_document_batch",
    "sparse_contact_document_loss",
]
