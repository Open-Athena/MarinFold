# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from marinfold.document_structures.core import Vocabulary
from marinfold.document_structures.documents import (
    ATTENTION_BLOCK,
    QUERY,
    AttentionLayout,
    Document,
    causal_training_document,
    pack,
)


def test_causal_training_document_uses_shifted_next_token_targets() -> None:
    document = causal_training_document((5, 6, 7, 1))

    assert document.attention == AttentionLayout.CAUSAL
    assert tuple(document.token_ids) == (5, 6, 7, 1)
    assert tuple(document[QUERY]) == (True, True, True, False)
    assert len(document.score_ranges) == 1
    target_range = document.score_ranges[0]
    assert (target_range.start, target_range.stop) == (0, 4)
    assert target_range.scored
    assert not target_range.has_explicit_targets
    assert target_range.target_count == 3


def test_with_targets_builds_shared_sparse_one_hot_matrix() -> None:
    document = Document((10, 11, 12, 13), attention=AttentionLayout.FULL).with_targets(
        (20, 21, 20)
    )

    prefix, targets = document.score_ranges
    assert (prefix.start, prefix.stop) == (0, 1)
    assert prefix.scored
    assert not prefix.has_explicit_targets
    assert (targets.start, targets.stop) == (1, 4)
    assert targets.target_ids == (20, 21)
    np.testing.assert_array_equal(
        targets.target_weights,
        np.asarray(((1.0, 0.0), (0.0, 1.0), (1.0, 0.0))),
    )


def test_with_target_distribution_normalizes_each_position() -> None:
    document = Document(
        (10, 11), attention=AttentionLayout.FULL
    ).with_target_distribution(
        (20, 21),
        ((4.0, 1.0), (0.0, 3.0)),
    )

    target_range = document.score_ranges[0]
    np.testing.assert_allclose(
        target_range.target_weights,
        np.asarray(((0.8, 0.2), (0.0, 1.0))),
    )


def test_target_distribution_rejects_duplicate_candidates_and_empty_rows() -> None:
    document = Document((10,), attention=AttentionLayout.FULL)

    with pytest.raises(ValueError, match="unique"):
        document.with_target_distribution((20, 20), ((1.0, 1.0),))
    with pytest.raises(ValueError, match="positive mass"):
        document.with_target_distribution((20, 21), ((0.0, 0.0),))


def test_concatenate_shifts_and_preserves_target_ranges() -> None:
    context = Document((1, 2, 3), attention=AttentionLayout.FULL).unscored()
    queries = Document((4, 4), attention=AttentionLayout.FULL).with_targets((8, 9))

    document = context + queries

    assert tuple(document.token_ids) == (1, 2, 3, 4, 4)
    assert len(document.score_ranges) == 2
    context_range, query_range = document.score_ranges
    assert (context_range.start, context_range.stop, context_range.scored) == (
        0,
        3,
        False,
    )
    assert (query_range.start, query_range.stop) == (3, 5)
    assert query_range.target_ids == (8, 9)


def test_concatenate_coalesces_adjacent_default_targets() -> None:
    first = Document((1, 2), attention=AttentionLayout.CAUSAL)
    second = Document((3, 4), attention=AttentionLayout.CAUSAL)

    document = first + second

    assert len(document.score_ranges) == 1
    assert (document.score_ranges[0].start, document.score_ranges[0].stop) == (0, 4)
    assert document.score_ranges[0].scored
    assert not document.score_ranges[0].has_explicit_targets


def test_concatenate_orders_block_causal_attention_spans() -> None:
    context = Document(
        (1, 2, 3),
        {ATTENTION_BLOCK: (0, 0, 0)},
        attention=AttentionLayout.BLOCK_CAUSAL,
    )
    queries = Document(
        (4, 4, 4),
        {ATTENTION_BLOCK: (0, 0, 1)},
        attention=AttentionLayout.BLOCK_CAUSAL,
    )

    document = context + queries

    assert tuple(document[ATTENTION_BLOCK]) == (0, 0, 0, 1, 1, 2)


def test_block_causal_attention_requires_ordered_contiguous_blocks() -> None:
    with pytest.raises(ValueError, match="nondecreasing contiguous"):
        Document(
            (1, 2, 3),
            {ATTENTION_BLOCK: (0, 2, 1)},
            attention=AttentionLayout.BLOCK_CAUSAL,
        )


def test_take_keeps_target_weights_aligned_when_reordering() -> None:
    document = Document((10, 11, 12), attention=AttentionLayout.FULL).with_targets(
        (20, 21, 22)
    )

    reordered = document.take((2, 0, 1))

    assert tuple(reordered.token_ids) == (12, 10, 11)
    assert len(reordered.score_ranges) == 1
    assert reordered.score_ranges[0].target_ids == (20, 21, 22)
    np.testing.assert_array_equal(
        reordered.score_ranges[0].target_weights,
        np.asarray(
            (
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        ),
    )


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


def test_pack_preserves_boundaries_and_sparse_targets() -> None:
    first = causal_training_document((5, 6, 7))
    second = (
        Document((8, 9), attention=AttentionLayout.CAUSAL)
        .unscored()
        .with_target_distribution((3, 4), ((4.0, 1.0),), start=0)
    )

    batch = pack((first, second), max_seq_len=6)

    assert batch.token_ids.shape == (1, 6)
    assert tuple(batch.segment_ids[0]) == (0, 0, 0, 1, 1, -1)
    assert tuple(batch.document_indices[0]) == (0, 0, 0, 1, 1, -1)
    assert len(batch.score_ranges) == 3
    explicit = next(
        target_range
        for target_range in batch.score_ranges
        if target_range.target_ids is not None
    )
    assert (explicit.row, explicit.start, explicit.stop) == (0, 3, 4)
    np.testing.assert_allclose(explicit.target_weights, ((0.8, 0.2),))
