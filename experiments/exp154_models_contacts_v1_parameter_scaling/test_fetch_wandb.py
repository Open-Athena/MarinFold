# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace

from fetch_wandb import Source, latest_subversion_rows, normalize_run, parse_count


class FakeSummary:
    def __init__(self, values: dict) -> None:
        self._json_dict = values


def fake_run(*, name: str, tags: list[str], loss_key: str, loss: float) -> SimpleNamespace:
    return SimpleNamespace(
        id="abc123",
        name=name,
        state="finished",
        tags=tags,
        config={
            "optimizer": {"learning_rate": 0.001, "weight_decay": 0.2},
            "trainer": {"train_batch_size": 128},
        },
        summary=FakeSummary({loss_key: loss, "_step": 42}),
        created_at="2026-01-01T00:00:00Z",
        heartbeatAt="2026-01-02T00:00:00Z",
    )


class FetchWandbTest(unittest.TestCase):
    def test_parse_count_suffixes(self) -> None:
        self.assertEqual(parse_count("1.471B"), 1_471_000_000)
        self.assertEqual(parse_count("4676648960"), 4_676_648_960)

    def test_normalize_exp75_metric_alias_and_config_hyperparameters(self) -> None:
        source = Source(issue=75, project="eric-czech/marin", tag="exp75")
        run = fake_run(
            name="prot-exp75-cv1-1_5b-e8-lr1e-3-wd0p2-v1",
            tags=[
                "exp75",
                "1_5b",
                "sweep=v1",
                "params=1.471B",
                "params_exact=1471369216",
                "tokens=37.413B",
                "tokens_exact=37413191680",
                "epochs=8",
                "lr=0.001",
                "wd=0.2",
            ],
            loss_key="eval/contacts-v1-val/loss",
            loss=2.7566,
        )

        row = normalize_run(source, run)

        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row["sweep_subversion"], 1)
        self.assertEqual(row["num_params"], 1_471_369_216)
        self.assertEqual(row["learning_rate_source"], "config:optimizer.learning_rate")
        self.assertEqual(row["weight_decay_source"], "config:optimizer.weight_decay")
        self.assertEqual(row["batch_size"], 128)
        self.assertEqual(row["batch_size_source"], "config:trainer.train_batch_size")
        self.assertEqual(row["val_loss_key"], "eval/contacts-v1-val/loss")

    def test_config_and_tag_mismatch_fails(self) -> None:
        source = Source(issue=117, project="eric-czech/marin", tag="exp117")
        run = fake_run(
            name="prot-exp117-cv1-s02-mismatch",
            tags=[
                "exp117",
                "sweep_subversion=2",
                "params=1471371264",
                "tokens=4676648960",
                "epochs=1",
                "lr=0.002",
                "wd=0.2",
                "global_batch=128",
            ],
            loss_key="eval/tokenized/contacts-v1-val/loss",
            loss=2.8,
        )

        with self.assertRaisesRegex(ValueError, "learning_rate mismatch"):
            normalize_run(source, run)

    def test_latest_subversion_is_selected_per_issue(self) -> None:
        rows = [
            {"issue": 75, "sweep_subversion": 1, "run_name": "a"},
            {"issue": 117, "sweep_subversion": 0, "run_name": "b"},
            {"issue": 117, "sweep_subversion": 2, "run_name": "c"},
            {"issue": 146, "sweep_subversion": 1, "run_name": "d"},
        ]

        selected, latest = latest_subversion_rows(rows)

        self.assertEqual(latest, {75: 1, 117: 2, 146: 1})
        self.assertEqual([row["run_name"] for row in selected], ["a", "c", "d"])


if __name__ == "__main__":
    unittest.main()
