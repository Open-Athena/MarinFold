"""Check the conditional nulls against independently enumerable distributions."""

from itertools import combinations

import numpy as np
import pytest

from score_sampling_controls import amino_acid_draws, degree_draws, metric_samples


def test_amino_acid_control_conditions_on_contact_type_bias() -> None:
    # Every type-0 / type-0 pair is true, although only half of all pairs are.
    # The null must preserve this advantage rather than recover uniform chance.
    truth = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    maps = [np.array([[0, 0]], dtype=np.int32), np.empty((0, 2), dtype=np.int32)]
    actual = amino_acid_draws(maps, truth, np.array([0, 1]), np.array([0, 0]), 100, 15)
    np.testing.assert_array_equal(actual[:, 0], np.ones(100))
    np.testing.assert_array_equal(actual[:, 1], np.zeros(100))


def test_degree_sampler_matches_exhaustive_simple_graph_distribution() -> None:
    initial = np.array([[0, 0], [0, 1], [1, 1], [2, 2]], dtype=np.int32)
    truth = np.zeros((3, 3), dtype=np.uint8)
    truth[initial[:, 0], initial[:, 1]] = 1
    row_degrees, col_degrees = truth.sum(axis=1), truth.sum(axis=0)
    exact_tp = []
    for chosen in combinations(range(9), 4):
        graph = np.zeros((3, 3), dtype=np.uint8)
        graph.ravel()[list(chosen)] = 1
        if np.array_equal(graph.sum(axis=1), row_degrees) and np.array_equal(graph.sum(axis=0), col_degrees):
            exact_tp.append(int((graph * truth).sum()))
    assert len(exact_tp) > 1
    actual, diagnostics = degree_draws([initial], truth, 30000, 100, 20, 999, 1)
    exact_histogram = np.bincount(exact_tp, minlength=5) / len(exact_tp)
    sampled_histogram = np.bincount(actual[:, 0], minlength=5) / len(actual)
    np.testing.assert_allclose(sampled_histogram, exact_histogram, atol=.015, rtol=0)
    assert diagnostics[0]['acceptance_rate'] > 0


def test_forced_degree_graph_is_retained_and_failed_attempts_get_zero() -> None:
    maps = [np.array([[0, 0], [0, 1]], dtype=np.int32), np.empty((0, 2), dtype=np.int32)]
    truth = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    tp, diagnostics = degree_draws(maps, truth, 20, 100, 20, 42, 2)
    np.testing.assert_array_equal(tp[:, 0], np.ones(20))
    assert diagnostics[0]['acceptance_rate'] == 0
    scores = metric_samples(tp, np.array([2, 0]), 1, 4, np.array([True, False]))
    assert scores['f1'][0, 0] == pytest.approx(2 / 3)
    assert scores['r_precision'][0, 0] == .5
    np.testing.assert_array_equal(scores['r_precision'][:, 1], np.zeros(20))
