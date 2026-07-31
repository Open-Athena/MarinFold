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
from levanter.models.lm_model import LmHeadModel
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
    attention_mask: AttentionMask
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


def _model_log_probs(model: LmHeadModel, batch, *, key=None) -> jax.Array:
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
    return jax.nn.log_softmax(logits.array, axis=-1)


def document_loss(
    model: LmHeadModel,
    batch: LevanterDocumentBatch,
    *,
    key=None,
) -> jnp.ndarray:
    """Run one model forward pass and apply weighted categorical cross-entropy."""
    log_probs = _model_log_probs(model, batch, key=key)
    batch_indices = jnp.arange(log_probs.shape[0])[:, None]
    selected = log_probs[
        batch_indices,
        batch.target_positions,
        batch.target_ids,
    ]
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
    if contact_count == 0:
        raise ValueError("Compact contacts document has no contacts")
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

        return CompactContactDocumentBatch(
            tokens=tokens,
            contact_first_ids=jnp.asarray(first_ids),
            contact_second_ids=jnp.asarray(second_ids),
            contact_count=jnp.asarray(contact_count, dtype=jnp.int32),
            prediction_start=jnp.asarray(prediction_start, dtype=jnp.int32),
            position_ids=position_ids,
            attention_mask=attention_mask,
            target_position_count=jnp.asarray(3 * contact_count + 1, dtype=jnp.int32),
            vocabulary=packed.vocabulary,
        )


def compact_contact_document_loss(
    model: LmHeadModel,
    batch: CompactContactDocumentBatch,
    *,
    key=None,
) -> jnp.ndarray:
    """Contacts-v1 soft-target loss from compact endpoint lists."""
    log_probs = _model_log_probs(model, batch, key=key)
    max_contacts = batch.contact_first_ids.shape[-1]
    contact_token_id = jnp.asarray(int(CONTACT), dtype=jnp.int32)
    end_token_id = jnp.asarray(int(END), dtype=jnp.int32)
    contact_axis = jnp.arange(max_contacts, dtype=jnp.int32)

    def one_example_loss(
        log_probs_one, first_ids, second_ids, contact_count, prediction_start
    ):
        valid = contact_axis < contact_count
        contact_positions = prediction_start + 1 + 3 * contact_axis
        first_positions = contact_positions + 1
        second_positions = contact_positions + 2
        contact_predict_positions = jnp.where(
            contact_axis == 0, prediction_start, second_positions - 3
        )
        contact_predict_positions = jnp.clip(contact_predict_positions, 0, log_probs_one.shape[0] - 1)
        contact_loss = -jnp.sum(
            jnp.where(valid, log_probs_one[contact_predict_positions, contact_token_id], 0.0)
        )
        end_position = jnp.clip(prediction_start + 3 * contact_count, 0, log_probs_one.shape[0] - 1)
        end_loss = -log_probs_one[end_position, end_token_id]

        c = contact_axis[:, None]
        r = contact_axis[None, :]
        remaining = (r >= c) & (r < contact_count) & (c < contact_count)
        first_row_positions = jnp.clip(contact_positions[:, None], 0, log_probs_one.shape[0] - 1)
        first_endpoint_loss = -(
            log_probs_one[first_row_positions, first_ids[None, :]]
            + log_probs_one[first_row_positions, second_ids[None, :]]
        )
        first_denominator = jnp.maximum(2 * (contact_count - contact_axis), 1)
        first_loss_by_contact = (
            jnp.sum(jnp.where(remaining, first_endpoint_loss, 0.0), axis=1)
            / first_denominator
        )
        first_loss = jnp.sum(jnp.where(valid, first_loss_by_contact, 0.0))

        actual_first = first_ids[:, None]
        incident_first = remaining & (first_ids[None, :] == actual_first)
        incident_second = remaining & (second_ids[None, :] == actual_first)
        second_row_positions = jnp.clip(first_positions[:, None], 0, log_probs_one.shape[0] - 1)
        second_loss_terms = -(
            jnp.where(
                incident_first,
                log_probs_one[second_row_positions, second_ids[None, :]],
                0.0,
            )
            + jnp.where(
                incident_second,
                log_probs_one[second_row_positions, first_ids[None, :]],
                0.0,
            )
        )
        second_denominator = jnp.maximum(
            jnp.sum(incident_first | incident_second, axis=1), 1
        )
        second_loss_by_contact = jnp.sum(second_loss_terms, axis=1) / second_denominator
        second_loss = jnp.sum(jnp.where(valid, second_loss_by_contact, 0.0))
        return contact_loss + end_loss + first_loss + second_loss

    losses = jax.vmap(one_example_loss)(
        log_probs,
        batch.contact_first_ids,
        batch.contact_second_ids,
        batch.contact_count,
        batch.prediction_start,
    )
    return jnp.sum(losses) / jnp.sum(batch.target_position_count)


__all__ = [
    "CompactContactDocumentBatch",
    "LevanterDocumentBatch",
    "compact_contact_document_batch",
    "compact_contact_document_loss",
    "document_loss",
    "levanter_document_batch",
]
