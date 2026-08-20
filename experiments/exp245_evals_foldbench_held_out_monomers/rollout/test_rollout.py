# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pieces of exp245's evaluation that would fail silently.

Three classes of thing are worth a test here. That the two modules copied
verbatim from PR #244 really are verbatim -- a hand-edited copy of a scoring
worker is exactly the kind of drift that produces numbers nobody can reconcile.
That the reporting cuts partition the eval sets rather than quietly dropping or
double-counting proteins. And that the input validation rejects an eval set that
is not the published one.

    uv run --extra test pytest test_rollout.py
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import checkpoint_specs as specs  # noqa: E402
from finalize_coreweave import aggregate_subsets  # noqa: E402

#: The exp232 rollout harness this one is copied from, in the same worktree.
EXP232_ROLLOUT = HERE.parents[1] / "exp232_sweep_cv1_decontam" / "evals" / "rollout_v2"

#: Modules that must stay byte-identical to PR #244's, because the scores are
#: only comparable to #244's if the scorer is literally the same program.
VERBATIM = ("hf_to_s3.py", "score_rollout_worker.py")


@pytest.mark.parametrize("name", VERBATIM)
def test_copied_modules_are_verbatim(name: str) -> None:
    if not EXP232_ROLLOUT.is_dir():
        pytest.skip("exp232 rollout harness not in this worktree")
    mine = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    theirs = hashlib.sha256((EXP232_ROLLOUT / name).read_bytes()).hexdigest()
    assert mine == theirs, f"{name} has diverged from PR #244's copy"


def test_checkpoint_suite_is_three_distinct_checkpoints() -> None:
    suite = specs.CHECKPOINT_SUITES["exp245"]
    assert len(suite) == 3
    assert len({c.label for c in suite}) == 3
    assert len({c.coreweave_uri for c in suite}) == 3
    # Every checkpoint is read in place; nothing is copied for this evaluation.
    assert all(c.coreweave_uri and c.hf_repo_id is None for c in suite)


def test_expected_sizes_sum_to_the_universe() -> None:
    assert sum(specs.EXPECTED_SET_SIZES.values()) == specs.EXPECTED_UNITS
    assert set(specs.EXPECTED_SET_SIZES) == set(specs.EVAL_SETS)


def _fake_inputs(sizes: dict[str, int], viral: dict[str, int]):
    """A metric frame and set manifest with the given per-set and viral counts."""
    rows, manifest = [], []
    index = 0
    for name, size in sizes.items():
        for member in range(size):
            stem = f"p{index:04d}_A"
            index += 1
            manifest.append({
                "dataset": "foldbench_monomer", "stem": stem, "eval_set": name,
                "is_viral": int(member < viral[name]),
            })
            for metric_range in ("all", "long"):
                rows.append({
                    "model": "m", "dataset": "foldbench_monomer", "stem": stem,
                    "range": metric_range, "cut": "R", "precision": 0.5,
                })
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def test_reporting_cuts_partition_each_set() -> None:
    sizes = specs.EXPECTED_SET_SIZES
    viral = {"eval-val": 6, "eval-test": 13, "eval-denovo": 0}
    precision, manifest = _fake_inputs(sizes, viral)
    units = list(zip(manifest.dataset, manifest.stem, strict=True))
    aggregate, counts = aggregate_subsets(
        precision, ordered_units=units, sets_manifest=manifest)
    assert counts["universe"] == specs.EXPECTED_UNITS
    for name, size in sizes.items():
        assert counts[name] == size
        assert counts[f"{name}-viral"] + counts[f"{name}-nonviral"] == size
    # An empty viral cell emits no rows rather than an all-NaN row.
    assert "eval-denovo-viral" not in set(aggregate.subset)


def test_reporting_cuts_reject_a_reassigned_protein() -> None:
    """A universe of the right size, with one protein in the wrong set."""
    sizes = specs.EXPECTED_SET_SIZES
    precision, manifest = _fake_inputs(sizes, {k: 0 for k in sizes})
    moved = manifest.index[manifest.eval_set == "eval-test"][0]
    manifest.loc[moved, "eval_set"] = "eval-val"
    units = list(zip(manifest.dataset, manifest.stem, strict=True))
    with pytest.raises(ValueError, match="eval-val has 98, not 97"):
        aggregate_subsets(precision, ordered_units=units, sets_manifest=manifest)
