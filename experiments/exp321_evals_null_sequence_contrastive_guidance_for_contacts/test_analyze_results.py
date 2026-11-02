"""Unit tests for oracle rollout scoring."""

import pytest

from analyze_results import rollout_r_precision


def test_rollout_r_precision_uses_emission_order_and_fixed_r_denominator() -> None:
    truth = {(0, 6), (0, 7), (1, 8)}
    contacts = [(0, 6), (9, 15), (0, 6), (0, 7), (1, 8)]

    assert rollout_r_precision(contacts, truth) == pytest.approx(2 / 3)
    assert rollout_r_precision([(0, 6)], truth) == pytest.approx(1 / 3)


def test_rollout_r_precision_is_undefined_without_true_contacts() -> None:
    assert rollout_r_precision([(0, 6)], set()) != rollout_r_precision([(0, 6)], set())
