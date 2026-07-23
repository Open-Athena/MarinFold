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

from marinfold.document_structures.core import VocabularyIdentity
from marinfold.document_structures.documents import (
    ATTENTION_BLOCK,
    POSITION_IDS,
    AttentionLayout,
    Coordinate,
    PackedBatch,
)


@dataclass(frozen=True)
class _FlatTargets:
    rows: np.ndarray
    positions: np.ndarray
    token_ids: np.ndarray
    weights: np.ndarray
    position_count: int


class LevanterDocumentBatch(eqx.Module):
    """Named model inputs plus flattened sparse target distributions."""

    tokens: hax.NamedArray
    target_rows: jax.Array
    target_positions: jax.Array
    target_ids: jax.Array
    target_weights: jax.Array
    position_ids: hax.NamedArray
    attention_mask: AttentionMask
    target_position_count: int = eqx.field(static=True)
    vocabulary: VocabularyIdentity | None = eqx.field(static=True)


def _flatten_targets(packed: PackedBatch) -> _FlatTargets:
    rows: list[int] = []
    positions: list[int] = []
    token_ids: list[int] = []
    weights: list[float] = []
    position_count = 0

    for target_range in packed.score_ranges:
        if not target_range.scored:
            continue
        if target_range.target_ids is None:
            for position in range(target_range.start, target_range.stop - 1):
                rows.append(target_range.row)
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
                rows.append(target_range.row)
                positions.append(position)
                token_ids.append(target_range.target_ids[int(target_index)])
                weights.append(float(weight_row[int(target_index)]))
            position_count += 1

    if position_count == 0:
        raise ValueError("Packed document batch has no scored target positions")
    return _FlatTargets(
        rows=np.asarray(rows, dtype=np.int32),
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
) -> LevanterDocumentBatch:
    """Convert packed documents and their weighted targets to Levanter inputs."""
    if packed.token_ids.ndim != 2:
        raise ValueError(
            f"Packed document tokens must have rank 2, got {packed.token_ids.shape}"
        )
    if packed.token_ids.shape[1] != Pos.size:
        raise ValueError(
            f"Packed sequence length {packed.token_ids.shape[1]} does not match "
            f"Levanter Pos axis size {Pos.size}"
        )

    Batch = hax.Axis(batch_axis_name, packed.token_ids.shape[0])
    axes = (Batch, Pos)
    tokens = hax.named(jnp.asarray(packed.token_ids), axes)
    segment_ids = hax.named(jnp.asarray(packed.segment_ids), axes)
    raw_position_ids = np.asarray(packed[position_coordinate])
    position_ids = hax.named(jnp.asarray(np.maximum(raw_position_ids, 0)), axes)

    attention_mask = AttentionMask()
    if packed.attention == AttentionLayout.CAUSAL:
        attention_mask = AttentionMask.causal()
    elif packed.attention == AttentionLayout.BLOCK_CAUSAL:
        attention_blocks = hax.named(jnp.asarray(packed[ATTENTION_BLOCK]), axes)
        KPos = hax.Axis("key_position", Pos.size)
        key_blocks = attention_blocks.rename({Pos: KPos})
        explicit_mask = (
            attention_blocks.broadcast_axis(KPos) >= key_blocks.broadcast_axis(Pos)
        ).rearrange((Batch, Pos, KPos))
        attention_mask = AttentionMask.explicit(explicit_mask)
    attention_mask = attention_mask.with_segment_ids(segment_ids)

    targets = _flatten_targets(packed)
    return LevanterDocumentBatch(
        tokens=tokens,
        target_rows=jnp.asarray(targets.rows),
        target_positions=jnp.asarray(targets.positions),
        target_ids=jnp.asarray(targets.token_ids),
        target_weights=jnp.asarray(targets.weights),
        target_position_count=targets.position_count,
        position_ids=position_ids,
        attention_mask=attention_mask,
        vocabulary=packed.vocabulary,
    )


def document_loss(
    model: LmHeadModel,
    batch: LevanterDocumentBatch,
    *,
    key=None,
) -> jnp.ndarray:
    """Run one model forward pass and apply weighted categorical cross-entropy."""
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
    log_probs = jax.nn.log_softmax(logits.array, axis=-1)
    selected = log_probs[
        batch.target_rows,
        batch.target_positions,
        batch.target_ids,
    ]
    return -jnp.sum(batch.target_weights * selected) / batch.target_position_count


__all__ = [
    "LevanterDocumentBatch",
    "document_loss",
    "levanter_document_batch",
]
