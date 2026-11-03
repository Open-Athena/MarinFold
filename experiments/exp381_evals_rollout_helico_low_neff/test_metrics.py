"""Scientific checks for invariance, coverage penalties, and residue identity."""

import numpy as np
import pytest

from score import atom_lddt, tmscore
from analyze import selections
import pandas as pd


def helix(n: int = 40) -> np.ndarray:
    """Make a non-collinear synthetic CA chain for scorer validation."""
    t = np.arange(n)
    return np.column_stack([2.3 * np.cos(t * 1.7), 2.3 * np.sin(t * 1.7), 1.5 * t])


def test_rigid_transform_and_reference_normalization(tmp_path) -> None:
    reference = helix()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    predicted = reference @ rotation + np.array([17, -5, 9])
    ids = np.arange(len(reference))
    full = tmscore(predicted, reference, ids, tmp_path / "full")
    assert full == {"tm_score": 1.0, "gdt_ts": 1.0}
    predicted[:20] = np.nan
    half = tmscore(predicted, reference, ids, tmp_path / "half")
    assert half["tm_score"] == pytest.approx(0.5, abs=0.0001)
    assert half["gdt_ts"] == pytest.approx(0.5, abs=0.0001)


def test_lddt_excludes_same_residue_and_penalizes_missing() -> None:
    reference = np.array([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]])
    ids = np.array([0, 0, 1, 1])
    names, elements = np.array(["CA", "N", "CA", "N"]), np.array(["C", "N", "C", "N"])
    assert atom_lddt(reference, reference + 10, ids, names, elements) == 1.0
    moved = reference.copy()
    moved[2:] += 100
    # Intra-residue distances remain perfect, but all scored distances are broken.
    assert atom_lddt(reference, moved, ids, names, elements) == 0.0
    moved[2:] = np.nan
    assert atom_lddt(reference, moved, ids, names, elements) == 0.0


def test_numbering_prevents_structural_realignment(tmp_path) -> None:
    reference = np.random.default_rng(381).normal(size=(40, 3)) * 10
    shifted = np.roll(reference, 8, axis=0)
    result = tmscore(shifted, reference, np.arange(40), tmp_path)
    assert result["tm_score"] < 0.5


def test_confidence_pick_shared_across_metrics_and_ties() -> None:
    frame = pd.DataFrame(
        dict(
            rollout=[2, 0, 1],
            ranking_score=[0.8, 0.8, 0.5],
            tm_score=[0.9, 0.4, 0.7],
            lddt=[0.3, 0.6, 0.8],
            gdt_ts=[0.5, 0.2, 0.9],
        )
    )
    rows = selections(frame)
    assert {r["selected_rollout"] for r in rows if r["selector"] == "top_pick"} == {0}
    assert {
        r["metric"]: r["selected_rollout"] for r in rows if r["selector"] == "oracle"
    } == {"tm_score": 2, "lddt": 1, "gdt_ts": 1}
