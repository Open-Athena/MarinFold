# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""The quantizers, the length strata, and the US-align TM-score wrapper.

The TM-score tests need the binary from ``setup_usalign.sh`` and are skipped
without it — they are the only part of the harness with a compiled dependency.
"""

import math

import numpy as np
import pytest

import usalign
from baseline_predictions import BOX_WIDTH_A, degrade, select_boxes
from canonical_pdb import build_atom_array, write_structure
from score_structures import length_bin

_HAS_USALIGN = usalign.DEFAULT_BINARY.exists()
_needs_usalign = pytest.mark.skipif(
    not _HAS_USALIGN, reason="run `bash setup_usalign.sh` to build US-align"
)


def _chain(n: int, spacing: float = 3.8):
    return build_atom_array(
        [(i + 1, "ALA", "CA", spacing * i, 0.0, 0.0, 0.0) for i in range(n)]
    )


def test_tenths_quantizer_matches_the_formats_own_digit_rule():
    # SPEC → "The <xyz-DDD> vocabulary": quantize once as n = round(v * 10).
    coord = np.array([[205.34, 71.86, 6.44], [0.04, 999.86, 12.35]])
    out = degrade(coord, "tenths")
    assert np.allclose(out, np.round(coord * 10.0) / 10.0)
    assert np.all(np.abs(out - coord) <= 0.05 + 1e-9)


def test_box10_quantizer_lands_on_cell_centers():
    coord = np.array([[0.0, 9.99, 10.01], [205.3, 71.8, 6.4]])
    out = degrade(coord, "box10")
    assert np.allclose(out % BOX_WIDTH_A, BOX_WIDTH_A / 2.0)
    assert np.all(np.abs(out - coord) <= BOX_WIDTH_A / 2.0 + 1e-9)


def test_exact_mode_is_the_identity():
    coord = np.array([[1.234, 5.678, 9.012]])
    assert np.allclose(degrade(coord, "exact"), coord)


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        degrade(np.zeros((1, 3)), "quarters")


def test_select_boxes_keeps_whole_cells_and_respects_the_budget():
    rng = np.random.default_rng(0)
    # Three atoms per 10 Å cell, ten cells.
    coord = np.repeat(np.arange(10.0) * BOX_WIDTH_A, 3).reshape(-1, 1)
    coord = np.hstack([coord, np.zeros((30, 2))])
    keep = select_boxes(coord, 0.5, rng)
    assert keep.sum() <= 15
    # Every kept cell is kept whole.
    for cell in range(10):
        members = keep[cell * 3 : (cell + 1) * 3]
        assert members.all() or not members.any()


def test_select_boxes_is_a_noop_at_full_coverage():
    coord = np.random.default_rng(0).normal(size=(20, 3)) * 30.0
    assert select_boxes(coord, 1.0, np.random.default_rng(0)).all()


@pytest.mark.parametrize(
    "length,expected",
    [(30, "<=100"), (100, "<=100"), (101, "101-200"), (200, "101-200"),
     (201, "201-400"), (400, "201-400"), (401, ">400"), (2000, ">400")],
)
def test_length_bins_partition_the_range(length, expected):
    assert length_bin(length) == expected


@_needs_usalign
def test_tm_score_of_a_structure_against_itself_is_one(tmp_path):
    path = tmp_path / "self.pdb"
    write_structure(_chain(60), path)
    result = usalign.tm_score(path, path)
    assert result.tm_score == pytest.approx(1.0, abs=1e-3)
    assert result.n_aligned == 60


@_needs_usalign
def test_tm_score_is_normalized_by_the_ground_truth_length(tmp_path):
    # Half the residues predicted, each of them perfectly. The score must fall
    # to ~0.5 — this is the property that makes TM-score the coverage-penalized
    # headline metric rather than a "how good is what you emitted" one.
    gt = _chain(60)
    gt_path = tmp_path / "gt.pdb"
    pred_path = tmp_path / "pred.pdb"
    write_structure(gt, gt_path)
    write_structure(gt[np.arange(len(gt)) % 2 == 0], pred_path)

    result = usalign.tm_score(pred_path, gt_path)
    assert result.tm_score == pytest.approx(0.5, abs=0.02)
    # Normalized by its own length, the partial prediction still looks perfect.
    assert result.tm_score_pred_normalized == pytest.approx(1.0, abs=1e-3)
    assert result.len_gt == 60
    assert result.len_pred == 30


@_needs_usalign
def test_tm_score_reports_a_version_string():
    version = usalign.binary_version()
    assert "US-align" in version


@_needs_usalign
def test_missing_binary_raises_with_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="setup_usalign.sh"):
        usalign.require_binary(tmp_path / "nope")


def test_box_sigma_matches_a_uniform_over_the_cell():
    from baseline_predictions import BOX_SIGMA, TENTH_SIGMA

    assert BOX_SIGMA == pytest.approx(BOX_WIDTH_A / math.sqrt(12.0))
    assert TENTH_SIGMA == pytest.approx(0.1 / math.sqrt(12.0))


def _write_bundle(root, n_records=2, n_residues=40):
    """A minimal ground-truth bundle: index + structures, nothing else."""
    import json

    index = []
    for k in range(n_records):
        array = _chain(n_residues)
        path = root / "gt_structures" / "toy" / f"p{k}.pdb"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_structure(array, path)
        index.append(
            {
                "record_id": f"toy/p{k}",
                "stem": f"p{k}",
                "dataset": "toy",
                "L": n_residues,
                "n_gt_atoms": len(array),
                "n_gt_ca": len(array),
                "n_gt_residues": n_residues,
            }
        )
    (root / "gt_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index)
    )
    return index


@_needs_usalign
def test_end_to_end_scoring_of_a_perfect_and_a_missing_prediction(tmp_path):
    import pandas as pd

    import score_structures

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()
    _write_bundle(gt_dir)

    # One record predicted exactly, the other not predicted at all.
    pred_dir = tmp_path / "pred"
    (pred_dir / "toy").mkdir(parents=True)
    write_structure(_chain(40), pred_dir / "toy" / "p0.pdb")

    out = tmp_path / "scores.csv"
    assert (
        score_structures.main(
            [
                "--gt-dir", str(gt_dir),
                "--pred-dir", str(pred_dir),
                "--model-name", "toy",
                "--out", str(out),
                "--jobs", "1",
            ]
        )
        == 0
    )

    scores = pd.read_csv(out).set_index("record_id")
    assert scores.loc["toy/p0", "status"] == "ok"
    assert scores.loc["toy/p0", "lddt_all"] == pytest.approx(1.0)
    assert scores.loc["toy/p0", "tm_score"] == pytest.approx(1.0, abs=1e-3)

    # A record with no prediction file is scored as a total miss, not skipped:
    # dropping it would inflate the mean over whatever the predictor finished.
    assert scores.loc["toy/p1", "status"] == "missing"
    assert scores.loc["toy/p1", "lddt_all"] == 0.0
    assert scores.loc["toy/p1", "tm_score"] == 0.0
    assert scores.loc["toy/p1", "atom_coverage"] == 0.0

    summary = pd.read_csv(out.with_suffix(".summary.csv"))
    overall = summary[summary["stratum"] == "all"].iloc[0]
    assert overall["n"] == 2
    assert overall["n_missing"] == 1
    assert overall["mean_lddt_all"] == pytest.approx(0.5)
