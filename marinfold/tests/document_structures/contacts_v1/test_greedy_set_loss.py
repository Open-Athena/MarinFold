# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np

from marinfold.document_structures.contacts_v1.greedy_set_loss import (
    greedy_matched_contact_block_loss,
    parse_contact_block_targets,
)


def test_parse_contact_block_targets_reads_contact_triples() -> None:
    contact = 5
    begin = 9
    end = 10
    position_token_ids = np.arange(20, 30, dtype=np.int64)
    token_ids = np.asarray(
        [2, 8, 20, 86, begin, contact, 23, 24, contact, 27, 21, end, 1],
        dtype=np.int64,
    )

    targets = parse_contact_block_targets(
        token_ids,
        begin_statements_token_id=begin,
        contact_token_id=contact,
        end_token_id=end,
        position_token_ids=position_token_ids,
    )

    assert targets.begin_position == 4
    assert targets.end_position == 11
    np.testing.assert_array_equal(targets.slot_positions, [[5, 6, 7], [8, 9, 10]])
    np.testing.assert_array_equal(targets.pairs, [[3, 4], [1, 7]])


def test_greedy_matched_contact_block_loss_ignores_pair_order_and_orientation() -> None:
    contact = 5
    begin = 9
    end = 10
    position_token_ids = np.arange(20, 30, dtype=np.int64)
    token_ids = np.asarray(
        [2, 8, 20, 86, begin, contact, 23, 24, contact, 27, 21, end, 1],
        dtype=np.int64,
    )
    log_probs = np.full((len(token_ids), 128), -100.0, dtype=np.float32)

    # Exact prefix: positions 0..3 predict token_ids[1..4].
    for pos in range(4):
        log_probs[pos, token_ids[pos + 1]] = -0.01

    # Slot 0 is serialized as pair (3, 4), but the model best matches pair (1, 7)
    # in reverse orientation: left=<p7>, right=<p1>.
    log_probs[4, contact] = -0.02
    log_probs[5, int(position_token_ids[7])] = -0.03
    log_probs[6, int(position_token_ids[1])] = -0.04
    log_probs[5, int(position_token_ids[3])] = -2.0
    log_probs[6, int(position_token_ids[4])] = -2.0

    # Slot 1 then must match the remaining pair (3, 4).
    log_probs[7, contact] = -0.05
    log_probs[8, int(position_token_ids[3])] = -0.06
    log_probs[9, int(position_token_ids[4])] = -0.07

    log_probs[10, end] = -0.08

    result = greedy_matched_contact_block_loss(
        log_probs,
        token_ids,
        begin_statements_token_id=begin,
        contact_token_id=contact,
        end_token_id=end,
        position_token_ids=position_token_ids,
    )

    assert [choice.pair for choice in result.choices] == [(1, 7), (3, 4)]
    assert result.choices[0].oriented_tokens == (
        int(position_token_ids[7]),
        int(position_token_ids[1]),
    )
    expected = 4 * 0.01 + (0.02 + 0.03 + 0.04) + (0.05 + 0.06 + 0.07) + 0.08
    assert math.isclose(result.loss, expected, rel_tol=1e-6)
