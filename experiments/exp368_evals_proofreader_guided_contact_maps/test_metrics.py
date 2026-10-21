"""Protect full-universe metrics from candidate-only evaluation bias."""

import numpy as np
from sklearn.metrics import average_precision_score

from metrics import map_metrics


def test_sparse_ap_includes_missing_positives_and_all_zero_ties() -> None:
    length = 12
    pairs = [(a, b) for a in range(length) for b in range(a + 6, length)]
    truth = {(0, 6), (0, 7), (0, 8), (1, 9)}
    scores = {(0, 6): .8, (1, 7): .8, (0, 7): .2}
    expected = average_precision_score([p in truth for p in pairs], [scores.get(p, 0) for p in pairs])
    actual = map_metrics(scores, truth, length)
    np.testing.assert_allclose(actual['average_precision'], expected, rtol=1e-12)
    assert actual['oracle_coverage'] == .5
    assert actual['r_precision'] == .5
    assert map_metrics({}, truth, length)['average_precision'] == len(truth) / len(pairs)
