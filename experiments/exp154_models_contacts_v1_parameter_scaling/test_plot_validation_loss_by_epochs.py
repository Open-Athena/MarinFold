# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import unittest

from plot_validation_loss_by_epochs import CLIPPED_Y, build_plot_rows


class PlotValidationLossByEpochsTest(unittest.TestCase):
    def test_best_run_is_selected_per_model_size_and_epoch(self) -> None:
        source_rows = [
            self.row("a", issue=75, model_size="1_5b", epochs=2, loss=3.0),
            self.row("b", issue=75, model_size="1_5b", epochs=2, loss=2.9),
            self.row("c", issue=146, model_size="3b", epochs=2, loss=2.95),
            self.row("d", issue=146, model_size="3b", epochs=2, loss=2.92),
            self.row("e", issue=117, model_size="1_5b", epochs=8, loss=3.4),
        ]

        rows = build_plot_rows(source_rows)

        best_ids = {row["run_id"] for row in rows if row["is_group_best"]}
        self.assertEqual(best_ids, {"b", "d", "e"})
        clipped = next(row for row in rows if row["run_id"] == "e")
        self.assertTrue(clipped["is_y_clipped"])
        self.assertEqual(clipped["plot_y"], CLIPPED_Y)
        self.assertTrue(all(-0.17 <= row["plot_x"] <= 1.17 for row in rows))

    @staticmethod
    def row(run_id: str, *, issue: int, model_size: str, epochs: int, loss: float) -> dict[str, str]:
        return {
            "issue": str(issue),
            "run_id": run_id,
            "run_name": f"run-{run_id}",
            "state": "finished",
            "model_size": model_size,
            "epochs": str(epochs),
            "val_loss": str(loss),
        }


if __name__ == "__main__":
    unittest.main()
