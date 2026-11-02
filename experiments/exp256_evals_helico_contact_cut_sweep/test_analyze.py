# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Protect matched populations and prompt/token contact interpretation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze import CUT_ORDER, contact_metrics, curve_table, load_arm
from score_coordinates import sparse_lddt


def result_rows() -> pd.DataFrame:
    """Return one valid result row for completeness checks."""
    return pd.DataFrame([dict(target_id="a", dataset="eval-val", status="ok", lddt=0.5)])


class AnalysisTest(unittest.TestCase):
    def test_missing_or_duplicate_target_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_rows().to_csv(root / "mf_L.csv", index=False)
            with self.assertRaisesRegex(ValueError, "target IDs"):
                load_arm(root, "mf_L", {"a", "b"})
            pd.concat([result_rows(), result_rows()]).to_csv(root / "mf_L.csv", index=False)
            with self.assertRaisesRegex(ValueError, "target IDs"):
                load_arm(root, "mf_L", {"a"})

    def test_missing_arm_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                load_arm(Path(directory), "mf_L", {"a"})

    def test_other_eval_set_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_rows().assign(dataset="eval-test").to_csv(root / "mf_L.csv", index=False)
            with self.assertRaisesRegex(ValueError, "dataset"):
                load_arm(root, "mf_L", {"a"})

    def test_nonfinite_success_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_rows().assign(lddt=float("nan")).to_csv(root / "mf_L.csv", index=False)
            with self.assertRaisesRegex(ValueError, "finite"):
                load_arm(root, "mf_L", {"a"})

    def test_contact_recall_uses_resolved_truth_and_token_map(self) -> None:
        record = dict(L=20, resolved=[1, 9, 18],
                      contacts=[[1, 9, 1.0], [9, 18, 1.0], [0, 18, 1.0]])
        metrics = contact_metrics(record, {"1": 0, "9": 1, "18": 2}, [[1, 0], [0, 2]])
        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 0.5)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            contact_metrics(record, {"1": 0, "9": 1, "18": 2}, [[1, 0], [0, 1]])

    def test_coordinate_lddt_respects_thresholds_and_cutoff(self) -> None:
        gt = np.array([[0, 0, 0], [10, 0, 0], [100, 0, 0]], dtype=np.float32)
        pred = np.array([[0, 0, 0], [11, 0, 0], [200, 0, 0]], dtype=np.float32)
        self.assertEqual(sparse_lddt(pred, gt), 0.5)
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
        self.assertEqual(sparse_lddt(gt @ rotation + 10, gt), 1.0)

    def test_contact_accuracy_uses_folding_population(self) -> None:
        wide = pd.DataFrame({tag: [0.5] for tag in CUT_ORDER}, index=["matched"])
        accuracy = pd.DataFrame([
            dict(target_id=target, arm=tag, precision=value, recall=value, pairs_over_L=1.0)
            for tag in CUT_ORDER for target, value in [("matched", 0.7), ("unmatched", 0.1)]])
        provenance = {"files": {f"results/{tag}.csv": {"run": "test"} for tag in CUT_ORDER}}
        curve = curve_table(wide, accuracy, provenance)
        self.assertTrue(curve.precision.eq(0.7).all())
        self.assertTrue(curve.recall.eq(0.7).all())
        self.assertTrue(curve.n.eq(1).all())


if __name__ == "__main__":
    unittest.main()
