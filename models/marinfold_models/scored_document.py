# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter bridge for documents with post-forward scoring ranges."""

from dataclasses import dataclass

import equinox as eqx
import haliax as hax
import jax.numpy as jnp
import numpy as np
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmHeadModel

from marinfold.document_structures.documents import (
    POSITION_IDS,
    AttentionLayout,
    Coordinate,
    PackedBatch,
    ScoreContext,
    Scorer,
)
from marinfold.document_structures.core import VocabularyIdentity


@dataclass(frozen=True)
class LevanterScoreRange:
    """Static range routing paired with targets stored in the batch arrays."""

    row: int
    start: int
    stop: int
    scorer: Scorer
    has_explicit_targets: bool

    @property
    def target_count(self) -> int:
        if self.has_explicit_targets:
            return self.stop - self.start
        return self.stop - self.start - 1


class LevanterScoredDocumentBatch(eqx.Module):
    """Named model inputs plus static scorer routing for one packed batch.

    Token and target values are ordinary JAX leaves. Range bounds and callback
    identities are static because Python callables cannot be dynamic JAX
    values. Consequently, callers should bucket batches by packing/range shape
    to reuse compilations; changing only token or target values does not
    recompile.
    """

    tokens: hax.NamedArray
    target_ids: hax.NamedArray
    position_ids: hax.NamedArray
    attention_mask: AttentionMask
    score_ranges: tuple[LevanterScoreRange, ...] = eqx.field(static=True)
    vocabulary: VocabularyIdentity | None = eqx.field(static=True)


def levanter_scored_document_batch(
    packed: PackedBatch,
    *,
    Pos: hax.Axis,
    position_coordinate: Coordinate = POSITION_IDS,
    batch_axis_name: str = "batch",
) -> LevanterScoredDocumentBatch:
    """Convert a packed document batch into Levanter/Haliax model inputs."""
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
    attention_mask = attention_mask.with_segment_ids(segment_ids)

    target_ids = np.full(packed.token_ids.shape, -1, dtype=np.int32)
    compiled_ranges: list[LevanterScoreRange] = []
    for score_range in packed.score_ranges:
        if score_range.scorer is None:
            continue
        has_explicit_targets = score_range.target_ids is not None
        if has_explicit_targets:
            target_ids[score_range.row, score_range.start : score_range.stop] = (
                score_range.target_ids
            )
        compiled = LevanterScoreRange(
            row=score_range.row,
            start=score_range.start,
            stop=score_range.stop,
            scorer=score_range.scorer,
            has_explicit_targets=has_explicit_targets,
        )
        if compiled.target_count > 0:
            compiled_ranges.append(compiled)
    if not compiled_ranges:
        raise ValueError("Packed document batch has no scored target positions")

    return LevanterScoredDocumentBatch(
        tokens=tokens,
        target_ids=hax.named(jnp.asarray(target_ids), axes),
        position_ids=position_ids,
        attention_mask=attention_mask,
        score_ranges=tuple(compiled_ranges),
        vocabulary=packed.vocabulary,
    )


def scored_document_loss(
    model: LmHeadModel,
    batch: LevanterScoredDocumentBatch,
    *,
    key=None,
) -> jnp.ndarray:
    """Run one Levanter model forward pass, then apply document scorers."""
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
    weighted_loss = jnp.asarray(0.0, dtype=jnp.float32)
    target_count = 0
    for score_range in batch.score_ranges:
        row = score_range.row
        start = score_range.start
        stop = score_range.stop
        explicit_targets = None
        if score_range.has_explicit_targets:
            explicit_targets = batch.target_ids.array[row, start:stop]
        range_loss = score_range.scorer(
            logits.array[row, start:stop],
            ScoreContext(
                token_ids=batch.tokens.array[row, start:stop],
                target_ids=explicit_targets,
            ),
        )
        weighted_loss = weighted_loss + range_loss * score_range.target_count
        target_count += score_range.target_count
    return weighted_loss / target_count
