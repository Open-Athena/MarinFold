# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Smoke test: feed score.py a fake "best" tree built from the GT itself.

If the prediction == GT, dRMSD must be ~0. The MAE will be whatever a
delta-function distogram (probability 1.0 in the bin containing the GT
CB-CB distance) yields — which equals the per-pair distance from the GT
distance to that bin's midpoint, averaged. We don't pin a number on
that, just check it's finite and small (≪ 1 Å on average given 0.3 Å
bin width).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest


# Make the experiment dir importable for tests run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score import (  # noqa: E402
    _DISTOGRAM_BIN_MIDPOINTS,
    _DISTOGRAM_N_BINS,
    _read_protein_coords_from_cif,
    _pairwise_distance_matrix,
    score,
)


@pytest.fixture
def repo_inputs() -> Path:
    """The inputs/ dir we prepared by smoke-test earlier (10 proteins)."""
    p = Path(__file__).resolve().parents[1] / "inputs"
    if not p.exists():
        pytest.skip("inputs/ not built — run prepare-inputs first.")
    return p


def _make_delta_distogram(gt_rep_d: np.ndarray) -> np.ndarray:
    """Build a [N, N, 64] distogram with mass 1.0 in the bin containing the GT distance.

    Used to construct an oracle "perfect" distogram for the smoke test.
    For NaN entries (unresolved residue pairs) we put mass in the last
    bin — those pairs are masked out of MAE anyway.
    """
    n = gt_rep_d.shape[0]
    probs = np.zeros((n, n, _DISTOGRAM_N_BINS), dtype=np.float32)
    safe = np.where(np.isfinite(gt_rep_d), gt_rep_d, _DISTOGRAM_BIN_MIDPOINTS[-1])
    # Find nearest-bin index for each pair.
    idx = np.argmin(
        np.abs(safe[..., None] - _DISTOGRAM_BIN_MIDPOINTS[None, None, :]),
        axis=-1,
    )
    ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    probs[ii, jj, idx] = 1.0
    return probs


def test_score_perfect_prediction_has_near_zero_drmsd(tmp_path: Path, repo_inputs: Path):
    """Build a fake best/ from GT CIF + an oracle distogram; verify the metric shape."""
    manifest = repo_inputs / "manifest.csv"
    assert manifest.exists()
    import csv as _csv
    rows = list(_csv.DictReader(manifest.open()))
    assert rows, "manifest is empty"

    # Build best/{mode}/{stem}/ for a single (mode, stem) pair using the GT.
    target = rows[0]
    mode = "single_seq"
    stem = target["stem"]
    gt_cif = repo_inputs / "gt" / f"{stem}.cif"

    best_dir = tmp_path / "best" / mode / stem
    best_dir.mkdir(parents=True)

    # 1. structure.cif = GT itself (perfect prediction).
    (best_dir / "structure.cif").write_bytes(gt_cif.read_bytes())

    # 2. confidence.json = stub (only ranking_score is read).
    (best_dir / "confidence.json").write_text(json.dumps({"ranking_score": 0.99}))

    # 3. distogram.npz = oracle (delta on GT CB-CB distance per pair).
    coords = _read_protein_coords_from_cif(gt_cif)
    gt_rep_d, _ = _pairwise_distance_matrix(coords.rep)
    probs = _make_delta_distogram(gt_rep_d)
    np.savez_compressed(best_dir / "distogram.npz", probs=probs)

    # 4. provenance.json = stub (only seed / sample_idx / ranking_score are read).
    (best_dir / "provenance.json").write_text(json.dumps({
        "seed": 1, "sample_idx": 0, "ranking_score": 0.99,
    }))

    out_csv = tmp_path / "scores.csv"
    results = score(
        best_dir=tmp_path / "best",
        inputs_dir=repo_inputs,
        out_csv=out_csv,
        modes=[mode],
    )
    assert len(results) == 1
    r = results[0]
    # Prediction == GT → all structure-derived metrics should be ~0.
    assert r.drmsd_ca_angstrom == pytest.approx(0.0, abs=1e-6)
    assert r.mae_structure_ca_angstrom == pytest.approx(0.0, abs=1e-6)
    assert r.rmsd_ca_angstrom == pytest.approx(0.0, abs=1e-6)
    assert r.rmsd_all_heavy_angstrom == pytest.approx(0.0, abs=1e-6)
    # Oracle distogram → expected distance = bin midpoint of GT bin.
    # Max per-pair error is half the bin width (~0.15 Å); mean / RMS
    # are well below. Both the in-range and contact-regime variants
    # should land here.
    for metric_name in (
        "mae_distogram_cb_angstrom", "drmsd_distogram_cb_angstrom",
        "mae_distogram_cb_contact_angstrom", "drmsd_distogram_cb_contact_angstrom",
    ):
        v = getattr(r, metric_name)
        # `nan` only OK if there were zero usable pairs for that variant.
        if v == v:  # not NaN
            assert 0.0 <= v < 0.2, f"{metric_name} too high: {v}"
    # dRMSD >= MAE always (Jensen).
    assert r.drmsd_distogram_cb_angstrom >= r.mae_distogram_cb_angstrom
    # Oracle distogram: all true-contact pair scores are 1.0, all
    # non-contact pair scores are 0.0, so the ranking is perfect —
    # precision @ top L = min(n_contacts, L) / L. We just sanity-check
    # the value is in [0, 1] and the class denominators are non-negative.
    for class_name in ("short", "medium", "long"):
        for top in ("L", "L_2", "L_5"):
            col = f"prec_{class_name}_{top}"
            v = getattr(r, col)
            if v == v:  # not NaN (NaN only if class has no eligible pairs)
                assert 0.0 <= v <= 1.0, f"{col} out of range: {v}"
        assert getattr(r, f"n_{class_name}_contacts") >= 0
    assert out_csv.exists()
