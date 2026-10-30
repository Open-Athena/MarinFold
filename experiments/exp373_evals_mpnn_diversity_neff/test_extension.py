"""Check oracle scoring on known contact sets and paired-bootstrap equivalence."""

import numpy as np

from analysis_stats import bootstrap_mean
from extend_analysis import contact_f1, f1_trajectory, summarize_matrix


def test_f1_penalizes_false_contacts_and_missing_truth() -> None:
    truth = {1, 2, 3}
    assert contact_f1({1, 2, 3}, truth) == 1
    assert contact_f1({1, 2, 4}, truth) == 2 / 3
    assert contact_f1({1}, truth) == 0.5
    assert contact_f1(set(), truth) == 0
    assert np.isnan(contact_f1(set(), set()))


def test_oracle_keeps_prefix_order_and_mean_is_not_best() -> None:
    scores = np.array([0.2, 0.6, 0.4, 0.8, 0.1])
    best, mean = f1_trajectory(scores)
    np.testing.assert_allclose(best, [0.2, 0.6, 0.6, 0.8, 0.8])
    np.testing.assert_allclose(mean, [0.2, 0.4, 0.4, 0.5, 0.42])
    assert best[0] == scores[0] and mean[-1] < best[-1]


def test_curve_bootstrap_matches_scalar_protein_bootstrap() -> None:
    values = np.array([[0.1, 0.3], [0.4, 0.8], [0.2, 0.6], [0.9, 1.0]])
    mean, lo, hi = summarize_matrix(values)
    for column in range(2):
        expected = bootstrap_mean(values[:, column])
        np.testing.assert_allclose(
            [mean[column], lo[column], hi[column]], expected[:3], rtol=0, atol=1e-14
        )
