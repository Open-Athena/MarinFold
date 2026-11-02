"""Independent checks of distance thresholds, sparse truth, and tied rankings."""

import numpy as np
import pytest
from analyze import protein_rows, tie_statistics


def test_boundary_tie_extremes() -> None:
    result = tie_statistics(np.array([9, 5, 5, 5, 0]), np.array([0, 1, 1, 0, 1]), 2)
    assert result["precision_tie_expected"] == pytest.approx(1 / 3)
    assert result["precision_tie_min"] == 0
    assert result["precision_tie_max"] == 0.5
    assert result["zero_vote_fraction"] == 0


def test_long_range_distance_boundary_and_missing_coordinates() -> None:
    # Only (0,24) and (0,29) span >=24 among these resolved positions.
    # The former lies at 7A; the latter at exactly 8A. Missing residue 25
    # gets the highest vote count but must not enter the scoring universe.
    item = {
        "truth": {
            "L": 30,
            "stem": "toy",
            "resolved": [0, 6, 12, 18, 24, 29],
            "contacts": [],
        },
        "votes": [[0, 24, 9], [0, 29, 8], [0, 25, 100]],
        "xyz": [
            [0, 0, 0, 0],
            [6, 4, 0, 0],
            [12, 100, 0, 0],
            [18, 104, 0, 0],
            [24, 7, 0, 0],
            [29, 8, 0, 0],
        ],
        "is_viral": False,
    }
    rows = protein_rows(item)
    selected = [
        r
        for r in rows
        if r["definition"] == "cb8" and r["range"] == "long" and r["cut"] == "L/5"
    ]
    assert len(selected) == 1
    row = selected[0]
    assert row["n_candidate"] == 2
    assert row["n_top"] == 2
    assert row["n_true"] == 1
    assert row["precision"] == 0.5
    assert row["pairs_at_exactly_8"] == 1


def test_zero_votes_are_ranked_and_accounted_for() -> None:
    result = tie_statistics(np.array([2, 0, 0, 0]), np.array([1, 1, 0, 0]), 3)
    assert result["zero_vote_fraction"] == pytest.approx(2 / 3)
    assert result["precision_tie_expected"] == pytest.approx(5 / 9)
    assert result["precision_tie_min"] == pytest.approx(1 / 3)
    assert result["precision_tie_max"] == pytest.approx(2 / 3)
