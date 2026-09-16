# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for target-anchored slot assignment."""

import numpy as np
import pytest

from marinfold.document_structures.contacts_set_v1.loss import (
    sigmoid_binary_cross_entropy,
    softmax_cross_entropy,
    target_anchored_slot_assignment,
)


def test_assignment_is_target_anchored_and_reports_extra_slots():
    cost = np.array([[4.0, 2.0, 0.1]])

    assignment = target_anchored_slot_assignment(cost)

    assert assignment.pairs == ((0, 2),)
    assert assignment.total_cost == pytest.approx(0.1)
    assert assignment.extra_slots == (0, 1)


def test_duplicate_candidate_slot_cannot_explain_two_targets():
    # Slot 0 is best for both targets, but it can be used only once.  The second
    # target must pay for another slot instead of receiving duplicate credit.
    cost = np.array([
        [0.0, 5.0, 9.0],
        [0.0, 5.0, 9.0],
    ])

    assignment = target_anchored_slot_assignment(cost)

    assert assignment.total_cost == pytest.approx(5.0)
    assert len({slot for _, slot in assignment.pairs}) == 2


def test_target_order_does_not_change_minimum_total_cost():
    cost = np.array([
        [0.1, 2.0, 4.0],
        [2.0, 0.2, 4.0],
        [4.0, 2.0, 0.3],
    ])

    forward = target_anchored_slot_assignment(cost)
    reversed_targets = target_anchored_slot_assignment(cost[::-1])

    assert forward.total_cost == pytest.approx(reversed_targets.total_cost)


def test_assignment_chooses_global_solution_not_rowwise_greedy():
    # If target 0 greedily takes slot 0 (cost 0.0), target 1 pays 100.  The
    # target-anchored assignment is still global one-to-one, so it gives target 0
    # slot 1 and target 1 slot 0 for total cost 1.0.
    cost = np.array([
        [0.0, 1.0],
        [0.0, 100.0],
    ])

    assignment = target_anchored_slot_assignment(cost)

    assert assignment.pairs == ((0, 1), (1, 0))
    assert assignment.total_cost == pytest.approx(1.0)


def test_empty_target_set_makes_every_prediction_extra():
    assignment = target_anchored_slot_assignment(np.zeros((0, 4)))

    assert assignment.pairs == ()
    assert assignment.total_cost == 0.0
    assert assignment.extra_slots == (0, 1, 2, 3)


def test_more_targets_than_slots_rejected():
    with pytest.raises(ValueError, match="cannot assign"):
        target_anchored_slot_assignment(np.zeros((3, 2)))


def test_reference_cross_entropy_helpers_are_stable():
    assert sigmoid_binary_cross_entropy(100.0, True) == pytest.approx(0.0, abs=1e-40)
    assert sigmoid_binary_cross_entropy(-100.0, False) == pytest.approx(0.0, abs=1e-40)
    assert softmax_cross_entropy(np.array([1000.0, 999.0]), 0) == pytest.approx(np.log1p(np.exp(-1.0)))
