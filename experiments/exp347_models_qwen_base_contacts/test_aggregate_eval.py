"""A missing or capped protein must never produce a headline evaluation result."""

import json
from pathlib import Path

import numpy as np
import pytest

from aggregate_eval import aggregate
from eval_contract import load_metric_reference, score_votes


def test_aggregate_rejects_incomplete_and_mismatched_results(tmp_path: Path) -> None:
    record = {
        "dataset": "test",
        "stem": "protein",
        "L": 30,
        "resolved": list(range(30)),
        "contacts": [[0, 7, 1], [0, 14, 1], [0, 29, 1]],
        "eval_set": "eval-val",
    }
    targets = tmp_path / "targets.jsonl"
    targets.write_text(json.dumps(record) + "\n")
    out = tmp_path / "result"
    reference = load_metric_reference(
        Path(__file__).resolve().parent.parent
        / "exp89_evals_contacts_v1_model_on_eval_set/compute_metrics.py"
    )
    votes = np.zeros((30, 30), dtype=int)
    for a, b, _ in record["contacts"]:
        votes[a, b] = votes[b, a] = 100
    with pytest.raises(FileNotFoundError):
        aggregate(str(out), targets, "model", "contacts_v1")
    assert not (out / "summary.json").exists()
    marker = out / "test__protein/complete.json"
    marker.parent.mkdir(parents=True)
    value = {
        "checkpoint": "model",
        "document_format": "contacts_v1",
        "n_rollouts": 100,
        "unfinished_rollouts": 1,
        "metrics": score_votes(votes, record, reference),
        "timing": {"stem": "protein", "elapsed_seconds": 1.0},
    }
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Incomplete rollouts"):
        aggregate(str(out), targets, "model", "contacts_v1")
    assert not (out / "summary.json").exists()
    value["unfinished_rollouts"] = 0
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Wrong checkpoint"):
        aggregate(str(out), targets, "other", "contacts_v1")
    result = aggregate(str(out), targets, "model", "contacts_v1")
    assert result["r_precision"] == dict.fromkeys(
        ["all", "short", "medium", "long"], 1.0
    )
    assert (out / "timings.csv").exists()
