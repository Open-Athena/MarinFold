"""Independent fixtures for complete-map scoring and distinct sampled modes."""

import unittest

import numpy as np
import pandas as pd

from analyze_whole_map import analyze_protein, jaccard_matrix, map_metrics


class WholeMapTest(unittest.TestCase):
    """Guard against cardinality and mixture artifacts in the interpretation."""

    def test_short_accurate_map_is_incomplete(self) -> None:
        metrics = map_metrics([(0, 30)], {(0, 30), (1, 31)})
        self.assertEqual(metrics["precision"], 1)
        self.assertEqual(metrics["recall"], .5)
        self.assertAlmostEqual(metrics["f1"], 2/3)
        self.assertEqual(metrics["ordered_fixed_r"], .5)

    def test_whole_map_score_ignores_emission_order(self) -> None:
        truth = {(0, 30), (1, 31)}
        a = map_metrics([(0, 30), (1, 31), (2, 32)], truth)
        b = map_metrics([(2, 32), (0, 30), (1, 31)], truth)
        self.assertEqual(a["f1"], b["f1"])
        self.assertNotEqual(a["ordered_fixed_r"], b["ordered_fixed_r"])

    def test_clustering_recovers_modes_hidden_by_marginal_ties(self) -> None:
        first, second = [(0, 30), (1, 31)], [(0, 31), (1, 32)]
        frame = pd.DataFrame({"contacts": [first if i % 2 == 0 else second for i in range(200)],
                              "mean_logprob": np.zeros(200)})
        item = {"truth": {"stem": "mixture", "L": 40, "resolved": list(range(40)),
                           "contacts": [[i, j, 1.] for i, j in first]}}
        _, methods, _, _ = analyze_protein(item, frame)
        scores = pd.DataFrame(methods).query("range == 'all'").groupby("method").f1.mean()
        self.assertEqual(scores["pooled_200"], .5)
        self.assertEqual(scores["sample_oracle_200"], 1.)
        self.assertEqual(scores["cluster_oracle_k2"], 1.)
        np.testing.assert_array_equal(jaccard_matrix([first, second]), np.eye(2))

    def test_blind_sample_selection_does_not_change_with_truth(self) -> None:
        first, second = [(0, 30), (1, 31)], [(0, 31), (1, 32)]
        frame = pd.DataFrame({"contacts": [first if i % 2 == 0 else second for i in range(200)],
                              "mean_logprob": [-1. if i % 2 == 0 else -.1 for i in range(200)]})
        choices = []
        for truth in (first, second):
            item = {"truth": {"stem": "changed_truth", "L": 40, "resolved": list(range(40)),
                               "contacts": [[i, j, 1.] for i, j in truth]}}
            _, methods, _, _ = analyze_protein(item, frame)
            choices.append([(r["method"], r["fold"], r["range"], r["sample"])
                            for r in methods if r["method"] in {"cross_pool_medoid", "mean_token_logprob"}])
        self.assertEqual(choices[0], choices[1])


if __name__ == "__main__":
    unittest.main()
