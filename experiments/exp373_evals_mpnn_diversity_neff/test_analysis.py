"""Check that diversity/ensemble scoring obey the published measurement universe."""

import numpy as np

from analyze import METRICS, candidate_sets, r_precision


def test_unresolved_contacts_are_not_false_positives() -> None:
    truth = {
        "L": 20,
        "resolved": [0, 7, 10],
        "contacts": [[0, 7, 0.001], [0, 10, 0.0009]],
    }
    record = {
        "samples": [
            {
                "offset": 0,
                "text": "<contact> <p0> <p7> <contact> <p0> <p12>",
                "contacts": [[0, 7], [0, 12]],
            }
        ]
    }
    sets, true, ids, labels = candidate_sets(record, truth)
    assert sets == [{7}] and true == {7}
    assert list(ids) == [7, 10] and list(labels) == [True, False]


def test_curve_matches_canonical_ranking_with_zero_ties() -> None:
    length = 20
    truth = METRICS.true_matrix(length, [[0, 7, 0.5], [0, 12, 0.5], [1, 9, 0.5]])
    pi, pj, sep = METRICS.resolved_pairs(np.arange(length))
    score = np.zeros((length, length))
    score[0, 7] = score[2, 15] = 4
    score[1, 9] = 2
    rows = METRICS.metric_rows(score, truth, pi, pj, sep, length, with_precision=True)
    for region, minimum in [("all", 6), ("long", 24)]:
        selected = sep >= minimum
        actual = r_precision(
            score.ravel(),
            pi[selected] * length + pj[selected],
            truth[pi[selected], pj[selected]],
        )
        expected = next(
            r["precision"] for r in rows if r["range"] == region and r["cut"] == "R"
        )
        assert np.isclose(actual, expected, equal_nan=True)
