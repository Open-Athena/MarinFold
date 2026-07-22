# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest

from marinfold.document_structures.core import Vocabulary
from marinfold.document_structures.documents import (
    QUERY,
    AttentionLayout,
    Document,
    ScoreContext,
    causal_training_document,
    next_token_score,
    pack,
)


def _custom_score(logits: Any, context: ScoreContext) -> Any:
    del context
    return np.sum(logits)


def test_causal_training_document_uses_default_shifted_token_scorer() -> None:
    document = causal_training_document((5, 6, 7, 1))

    assert document.attention == AttentionLayout.CAUSAL
    assert tuple(document.token_ids) == (5, 6, 7, 1)
    assert tuple(document[QUERY]) == (True, True, True, False)
    assert len(document.score_ranges) == 1
    score_range = document.score_ranges[0]
    assert (score_range.start, score_range.stop) == (0, 4)
    assert score_range.scorer is next_token_score
    assert score_range.target_ids is None

    logits = np.zeros((4, 10), dtype=np.float32)
    logits[0, 6] = 20.0
    logits[1, 7] = 20.0
    logits[2, 1] = 20.0
    assert score_range.scorer is not None
    loss = score_range.scorer(logits, ScoreContext(document.token_ids))
    assert float(loss) < 1e-6


def test_with_targets_defaults_to_final_positions_and_composes_with_scorer() -> None:
    document = Document((10, 11, 12, 13), attention=AttentionLayout.FULL).with_targets(
        (20, 21)
    )
    document = document.scored_by(_custom_score, start=2)

    assert len(document.score_ranges) == 2
    prefix, targets = document.score_ranges
    assert (prefix.start, prefix.stop) == (0, 2)
    assert prefix.scorer is next_token_score
    assert prefix.target_ids is None
    assert (targets.start, targets.stop) == (2, 4)
    assert targets.scorer is _custom_score
    assert targets.target_ids == (20, 21)


def test_concatenate_shifts_and_preserves_scoring_ranges() -> None:
    context = Document((1, 2, 3), attention=AttentionLayout.FULL).unscored()
    queries = Document((4, 4), attention=AttentionLayout.FULL).with_targets((8, 9))

    document = context + queries

    assert tuple(document.token_ids) == (1, 2, 3, 4, 4)
    assert len(document.score_ranges) == 2
    context_range, query_range = document.score_ranges
    assert (context_range.start, context_range.stop, context_range.scorer) == (
        0,
        3,
        None,
    )
    assert (query_range.start, query_range.stop) == (3, 5)
    assert query_range.scorer is next_token_score
    assert query_range.target_ids == (8, 9)


def test_concatenate_coalesces_adjacent_default_scoring() -> None:
    first = Document((1, 2), attention=AttentionLayout.CAUSAL)
    second = Document((3, 4), attention=AttentionLayout.CAUSAL)

    document = first + second

    assert len(document.score_ranges) == 1
    assert (document.score_ranges[0].start, document.score_ranges[0].stop) == (0, 4)
    assert document.score_ranges[0].scorer is next_token_score


def test_take_keeps_explicit_targets_aligned_when_reordering() -> None:
    document = Document((10, 11, 12), attention=AttentionLayout.FULL).with_targets(
        (20, 21, 22)
    )

    reordered = document.take((2, 0, 1))

    assert tuple(reordered.token_ids) == (12, 10, 11)
    assert len(reordered.score_ranges) == 1
    assert reordered.score_ranges[0].target_ids == (22, 20, 21)


def test_document_tracks_vocabulary_and_rejects_mixed_concatenation() -> None:
    first_vocabulary = Vocabulary("first", ("<a>", "<target>"))
    second_vocabulary = Vocabulary("second", ("<a>", "<target>"))
    first = Document(
        (first_vocabulary.token("<a>"),), attention=AttentionLayout.FULL
    ).with_targets(first_vocabulary.token("<target>"))
    second = Document((second_vocabulary.token("<a>"),), attention=AttentionLayout.FULL)

    assert first.vocabulary == first_vocabulary.identity
    with pytest.raises(ValueError, match="share one vocabulary"):
        _ = first + second
    with pytest.raises(ValueError, match="share one vocabulary"):
        _ = first + Document((2,), attention=AttentionLayout.FULL)


def test_document_rejects_mixed_token_handles_and_raw_ids() -> None:
    vocabulary = Vocabulary("test", ("<a>",))

    with pytest.raises(ValueError, match="must not mix"):
        Document((vocabulary.token("<a>"), 2), attention=AttentionLayout.FULL)


def test_pack_preserves_document_boundaries_padding_and_scoring_ranges() -> None:
    first = causal_training_document((5, 6, 7))
    second = causal_training_document((8, 9))

    batch = pack((first, second), max_seq_len=6)

    assert batch.token_ids.shape == (1, 6)
    assert tuple(batch.segment_ids[0]) == (0, 0, 0, 1, 1, -1)
    assert tuple(batch.document_indices[0]) == (0, 0, 0, 1, 1, -1)
    assert np.all(batch.token_ids[0, :5] == (5, 6, 7, 8, 9))
    assert len(batch.score_ranges) == 2
    first_range, second_range = batch.score_ranges
    assert (first_range.row, first_range.start, first_range.stop) == (0, 0, 3)
    assert first_range.document_index == 0
    assert first_range.scorer is next_token_score
    assert first_range.target_ids is None
    assert (second_range.row, second_range.start, second_range.stop) == (0, 3, 5)
    assert second_range.document_index == 1
    assert second_range.scorer is next_token_score
    assert second_range.target_ids is None
