"""Tests for AF2Rank benchmark metrics and pilot selection."""

import math
import unittest

from benchmark import af2rank_composite, rankdata, spearman_correlation
from select_pilot import quantile_indices, select_targets


class BenchmarkTest(unittest.TestCase):
    def test_rankdata_averages_ties(self) -> None:
        self.assertEqual(rankdata([30.0, 10.0, 20.0, 20.0]), [4.0, 1.0, 2.5, 2.5])

    def test_spearman_perfect_monotone_relationships(self) -> None:
        self.assertTrue(math.isclose(spearman_correlation([1, 2, 3], [4, 5, 6]), 1.0))
        self.assertTrue(math.isclose(spearman_correlation([1, 2, 3], [6, 5, 4]), -1.0))

    def test_af2rank_composite(self) -> None:
        row = {"plddt": "80", "ptm": "0.5", "tm_diff": "0.75"}
        self.assertEqual(af2rank_composite(row), 30.0)

    def test_quantile_indices_include_endpoints(self) -> None:
        self.assertEqual(quantile_indices(10, 4), [0, 3, 6, 9])

    def test_target_grid_is_unique_and_deterministic(self) -> None:
        rows = [
            {
                "target": f"t{index}",
                "n_native_residues": str(index),
                "n_decoys": str(20 - index),
            }
            for index in range(1, 11)
        ]
        first = select_targets(rows, [0.1, 0.9])
        second = select_targets(rows, [0.1, 0.9])
        self.assertEqual(first, second)
        self.assertEqual(len({row["target"] for row in first}), 4)


if __name__ == "__main__":
    unittest.main()
