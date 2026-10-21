"""Check held-out exclusion, checkpoint bundling, and incomplete-result rejection."""

import hashlib
import json
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
import pytest

from experiments.exp364_models_mpnn_only_lr_sweep.eval_prepare import select_targets
from experiments.exp364_models_mpnn_only_lr_sweep.eval_run import (
    markers_for,
    score_checkpoint,
    worker_request,
)
from experiments.exp364_models_mpnn_only_lr_sweep.launch_eval import build_bundle

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "exp245_evals_foldbench_held_out_monomers/data"


@pytest.fixture
def published_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    return (
        pd.read_parquet(SOURCE / "eval_targets_foldbench_monomers.parquet"),
        pd.read_csv(SOURCE / "eval_sets.csv"),
        [
            json.loads(line)
            for line in (SOURCE / "gt_universe_scored.jsonl").read_text().splitlines()
        ],
    )


def test_selection_excludes_held_out_and_unscorable(published_inputs) -> None:
    targets, sets, truth = published_inputs
    selected, gt = select_targets(targets, sets, truth)
    assert selected.eval_set.value_counts().to_dict() == {
        "eval-val": 97,
        "eval-denovo": 19,
    }
    held_out = set(
        sets.loc[(sets.eval_set == "eval-test") | (sets.scorable == 0), "stem"]
    )
    assert not set(selected.stem) & held_out
    assert {r["stem"] for r in gt} == set(selected.stem)


def test_selection_rejects_missing_truth(published_inputs) -> None:
    targets, sets, truth = published_inputs
    _, gt = select_targets(targets, sets, truth)
    missing = gt[0]["stem"]
    with pytest.raises(ValueError, match="Ground truth and targets differ"):
        select_targets(targets, sets, [r for r in truth if r["stem"] != missing])


def test_selection_rejects_wrong_coordinates(published_inputs) -> None:
    targets, sets, truth = published_inputs
    _, gt = select_targets(targets, sets, truth)
    changed = gt[0]["stem"]
    inconsistent = [
        {**r, "L": r["L"] + 1} if r["stem"] == changed else r for r in truth
    ]
    with pytest.raises(ValueError, match="lengths differ"):
        select_targets(targets, sets, inconsistent)


@pytest.mark.parametrize("fault", ["capped", "duplicate", "missing", "short"])
def test_completion_rejects_invalid_results(fault: str) -> None:
    fs = fsspec.filesystem("memory")
    root = f"/exp364-test-{fault}"
    unit = {
        "dataset": "foldbench_monomer",
        "stem": "example",
        "n_rollouts": 100,
        "unfinished_rollouts": 0,
    }
    marker = {"total_rollouts": 100, "unfinished_rollouts": 0, "units": [unit]}
    if fault == "capped":
        marker["unfinished_rollouts"] = 1
    elif fault == "duplicate":
        marker["units"] = [unit, unit]
        marker["total_rollouts"] = 200
    elif fault == "short":
        marker["units"] = [{**unit, "n_rollouts": 99}]
    with fs.open(root + "/complete/part.json", "wt") as handle:
        json.dump(marker, handle)
    with pytest.raises(ValueError):
        markers_for(fs, root, 2 if fault == "missing" else 1)


def test_bundle_contains_pinned_inputs_and_metric_implementation() -> None:
    bundle = build_bundle()
    manifest = json.loads((bundle / "bundle_manifest.json").read_text())
    assert any(name.endswith("compute_metrics.py") for name in manifest)
    assert any(name.endswith("score_rollout_worker.py") for name in manifest)
    for name, digest in manifest.items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest
    assert not any("eval-test" in name for name in manifest)


def test_worker_stays_local_at_batch_priority() -> None:
    plan = json.loads((HERE / "data/contact_eval/plan.json").read_text())
    job = worker_request(plan, plan["checkpoints"][0], {"files": []}, 0, False)
    assert job.priority == 3
    assert job.resources.target_cluster == "cw-us-east-02a"
    assert job.resources.device.count == 1
    assert job.resources.device.variant == "H100"
    assert job.environment.setup_scripts == []


@pytest.mark.parametrize("invalid_coordinates", [False, True])
def test_sparse_votes_use_ground_truth_coordinates(
    published_inputs, invalid_coordinates: bool
) -> None:
    targets, sets, truth = published_inputs
    _, selected = select_targets(targets, sets, truth)
    record = next(r for r in selected if len(r["contacts"]) > 100)
    key = {"dataset": record["dataset"], "stem": record["stem"], "L": record["L"]}
    fs = fsspec.filesystem("memory")
    plan = {"root": f"/exp364-test-votes-{invalid_coordinates}"}
    checkpoint = {"label": "test-model-step-123"}
    root = f"{plan['root']}/smoke/{checkpoint['label']}"
    pairs = {
        (int(i), int(j))
        for i, j, degree in record["contacts"]
        if degree >= 0.001 and j - i >= 6
    }
    rows = [{**key, "i": i, "j": j, "votes": 100} for i, j in sorted(pairs)]
    if invalid_coordinates:
        rows[0] = {**rows[0], "j": record["L"]}
    with fs.open(root + "/scores.parquet", "wb") as handle:
        pd.DataFrame(rows).to_parquet(handle, index=False)
    timing = {
        **key,
        "n_rollouts": 100,
        "stopped_rollouts": 100,
        "unfinished_rollouts": 0,
    }
    with fs.open(root + "/timings.parquet", "wb") as handle:
        pd.DataFrame([timing]).to_parquet(handle, index=False)
    marker = {
        "units": [timing],
        "total_rollouts": 100,
        "unfinished_rollouts": 0,
        "score_uri": root + "/scores.parquet",
        "timing_uri": root + "/timings.parquet",
    }
    with fs.open(root + "/complete/part.json", "wt") as handle:
        json.dump(marker, handle)
    if invalid_coordinates:
        with pytest.raises(ValueError, match="Invalid sparse vote row"):
            score_checkpoint(fs, plan, checkpoint, [record], phase="smoke")
        return
    metrics, _ = score_checkpoint(fs, plan, checkpoint, [record], phase="smoke")
    assert len(metrics) == 20
    assert (metrics.loc[metrics.cut == "R", "precision"] == 1).all()
    path = f"{plan['root']}/dense/smoke/{checkpoint['label']}/{record['dataset']}__{record['stem']}.npz"
    with fs.open(path, "rb") as handle:
        matrix = np.load(handle)["score"]
    assert matrix.shape == (record["L"], record["L"])
    assert np.array_equal(matrix, matrix.T)
