"""Protect full-universe metrics from candidate-only evaluation bias."""

import numpy as np
from sklearn.metrics import average_precision_score

from metrics import aggregate, diversity_weights, map_metrics


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


def test_confidence_tiebreak_cannot_reverse_an_extra_vote() -> None:
    rows = [dict(L=20, pairs=[[0, 8], [1, 9], [2, 10]], probabilities=[.999, .001, .5]),
            dict(L=20, pairs=[[1, 9]], probabilities=[.001])]
    scores = aggregate(rows, 'frequency_then_meanq')
    assert scores[(1, 9)] > scores[(0, 8)] > scores[(2, 10)]


def test_raw_frequency_counts_distinct_eligible_votes_without_reader() -> None:
    rows = [dict(L=20, pairs=[[0, 8], [0, 8], [1, 9], [0, 2], [-1, 8], [9, 1], [0, 20]]),
            dict(L=20, pairs=[[0, 8], [2, 10]])]
    assert aggregate(rows, 'frequency') == {(0, 8): 2., (1, 9): 1., (2, 10): 1.}


def test_f1_power_emphasizes_quality_without_changing_single_power_weights() -> None:
    rows = [dict(L=32, pairs=[[0, 24]], probabilities=[.6], predicted_recall=.6),
            dict(L=32, pairs=[[0, 24]], probabilities=[.6], predicted_recall=.6),
            dict(L=32, pairs=[[1, 25]], probabilities=[.9], predicted_recall=.9)]
    linear = aggregate(rows, 'f1_weighted_frequency')
    squared = aggregate(rows, 'f1_power2')
    np.testing.assert_allclose([linear[(0, 24)], linear[(1, 25)]], [1.2, .9], rtol=1e-12)
    assert linear[(0, 24)] > linear[(1, 25)]
    assert squared[(0, 24)] < squared[(1, 25)]


def test_bitset_discounts_match_exact_set_jaccard_with_boundary_and_empty_rows() -> None:
    rng = np.random.default_rng(368)
    universe = [(i, i + 6) for i in range(100)]
    sets = [set(), set(), set(universe[:4]), set(universe[:5]), set(universe[:4])]
    for _ in range(100):
        sets.append({pair for pair in universe if rng.random() < .85})
    expected = [1 / max(1, sum(bool(a | b) and len(a & b) / len(a | b) >= .8 for b in sets))
                for a in sets]
    actual = diversity_weights([{pair: .5 for pair in row} for row in sets])
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == actual[1] == 1.
    assert actual[2] == actual[3] == actual[4] == 1 / 3
