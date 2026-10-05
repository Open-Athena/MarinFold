# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from analyze_conditioning import (
    ARMS,
    DERIVED_ARMS,
    PRIMARY_REFERENCE,
    aggregate_scores,
    analyze,
    candidate_universes,
    complete_votes,
    margin_status,
    precision_at_r,
    summarize_scores,
)


def target_record() -> dict:
    contexts = {arm: [] for arm in ARMS}
    for dose in ("small", "large"):
        contexts[f"true_{dose}"] = [[0, 6]]
        contexts[f"false_{dose}"] = [[3, 9]]
        contexts[f"pred_{dose}"] = [[1, 7]]
    return {
        "dataset": "test",
        "stem": "unit",
        "L": 12,
        "input_seq": "A" * 12,
        "resolved": list(range(12)),
        "truth": [[0, 6], [1, 7], [2, 8]],
        "contexts": [contexts, contexts],
    }


def matrix_with_pairs(pairs: list[tuple[int, int, int]]) -> np.ndarray:
    matrix = np.zeros((12, 12), dtype=np.int16)
    for i, j, count in pairs:
        matrix[i, j] = matrix[j, i] = count
    return matrix


def write_plan(tmp_path: Path) -> Path:
    plan = {
        "targets": [target_record()],
        "n_rollouts": 4,
        "n_repeats": 2,
        "source_n_rollouts": 4,
        "practical_margin": 0.03,
        "prediction_source": "archived_iid_100_consensus",
        "source_votes_file": "source_votes.npz",
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    np.savez(tmp_path / "source_votes.npz", unit=matrix_with_pairs([(0, 6, 4)]))
    return path


def test_union_exclusion_removes_other_arms_givens_and_copied_context():
    target = target_record()
    pi, pj, truth, keep = candidate_universes(target, target["contexts"][0])
    excluded = set(zip(pi[~keep], pj[~keep]))
    assert excluded == {(0, 6), (1, 7), (3, 9)}
    # The conditioned continuation copies both supplied and other-arm context.
    # Shared withholding leaves the same one true pair for it and the control.
    conditioned = matrix_with_pairs([(0, 6, 4), (1, 7, 4), (2, 8, 2)])
    baseline = matrix_with_pairs([(3, 9, 4), (2, 8, 1)])
    assert precision_at_r(conditioned, truth, pi[keep], pj[keep])[::2] == (1.0, 1)
    assert precision_at_r(baseline, truth, pi[keep], pj[keep])[::2] == (1.0, 1)


def test_complete_context_is_present_once_per_rollout_without_mutation():
    votes = matrix_with_pairs([(0, 6, 3), (1, 7, 2)])
    completed = complete_votes(votes, [[0, 6], [2, 8]], 4)
    assert completed[0, 6] == completed[6, 0] == 4
    assert completed[2, 8] == 4
    assert completed[1, 7] == 2
    assert votes[0, 6] == 3
    assert votes[2, 8] == 0


def test_replicates_are_averaged_before_bootstrapping_proteins():
    rows = []
    for stem, predictions in (("a", [0.5, 1.0]), ("b", [0.25, 0.25])):
        for repeat in range(2):
            for arm in ARMS + DERIVED_ARMS:
                rows.append(
                    {
                        "dataset": "test",
                        "stem": stem,
                        "repeat": repeat,
                        "scope": "full_pipeline",
                        "arm": arm,
                        "precision": predictions[repeat]
                        if arm == "pred_large"
                        else 0.5,
                        "L": 12,
                        "n_true": 3,
                        "n_candidates": 21,
                        "n_context": 0,
                        "n_rollouts": 4,
                    }
                )
    per_protein = aggregate_scores(pd.DataFrame(rows))
    summary = summarize_scores(per_protein, 0.03)
    primary = summary[summary.role == "practical_primary"].iloc[0]
    assert primary.reference == PRIMARY_REFERENCE
    assert primary.n_proteins == 2
    assert primary.mean_delta == 0.0
    assert primary.ci_low == -0.25
    assert primary.ci_high == 0.25
    assert primary.margin_status == "inconclusive_at_margin"


def test_missing_unit_is_an_error_before_any_report(tmp_path):
    plan = write_plan(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing completed unit"):
        analyze(plan, tmp_path / "run", tmp_path / "results")
    assert not (tmp_path / "results").exists()


def test_end_to_end_primary_uses_same_archived_first_pass(tmp_path):
    plan = write_plan(tmp_path)
    units = tmp_path / "run" / "units"
    units.mkdir(parents=True)
    arrays = {}
    for repeat in range(2):
        for arm in ARMS:
            votes = matrix_with_pairs([(2, 8, 3), (4, 10, 4)])
            arrays[f"r{repeat}__{arm}__votes"] = votes
            if arm != "iid_repeat":
                arrays[f"r{repeat}__{arm}__prob"] = votes.astype(float) / 4
    np.savez(units / "unit.npz", **arrays)
    result = analyze(plan, tmp_path / "run", tmp_path / "results")
    assert result["primary"]["reference"] == PRIMARY_REFERENCE
    assert result["primary"]["n_proteins"] == 1
    scores = pd.read_csv(tmp_path / "results" / "conditioning_per_repeat.csv")
    row = scores[
        (scores.scope == "full_pipeline") & (scores.arm == PRIMARY_REFERENCE)
    ].iloc[0]
    assert row.n_rollouts == 8
    assert row.precision == pytest.approx(2 / 3)
    assert (tmp_path / "results" / "conditioning_changes.csv").is_file()


@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (0.04, 0.08, "gain_at_least_margin_supported"),
        (-0.01, 0.025, "target_gain_ruled_out"),
        (0.02, 0.04, "inconclusive_at_margin"),
    ],
)
def test_decision_uses_practical_margin_not_zero(lo, hi, expected):
    assert margin_status(lo, hi, 0.03) == expected
