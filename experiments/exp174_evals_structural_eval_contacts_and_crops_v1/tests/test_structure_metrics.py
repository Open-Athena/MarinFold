# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Metric semantics, on structures small enough to check by hand.

The two things worth testing here are the ones a reader of the results table
has to trust: that a perfect prediction scores perfectly under every metric,
and that an *incomplete* prediction is penalized by exactly the amount the
documented convention says it should be. The lDDT-with-missing-atoms case is
computed by parking unpredicted atoms at mutually distant sentinels, which is
subtle enough that it gets an exact arithmetic check rather than an
"approximately lower" one.
"""

import math

import numpy as np
import pytest

from canonical_pdb import build_atom_array
from structure_metrics import score_prediction


def _chain(n: int, spacing: float = 4.0):
    """``n`` single-CA residues on a line, ``spacing`` Å apart.

    At 4 Å spacing every pair of a 4-residue chain is inside lDDT's 15 Å
    inclusion radius, so the contact count is exactly ``n * (n - 1) / 2`` and
    the expected scores are hand-computable.
    """
    return build_atom_array(
        [(i + 1, "ALA", "CA", spacing * i, 0.0, 0.0, 0.0) for i in range(n)]
    )


def _rigid_transform(array, angle: float = 0.7, shift=(11.0, -4.0, 3.0)):
    moved = array.copy()
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    moved.coord = (array.coord @ rotation.T + np.array(shift)).astype(np.float32)
    return moved


def test_identical_prediction_is_perfect():
    gt = _chain(8)
    m = score_prediction(gt, gt)
    assert m["atom_coverage"] == 1.0
    assert m["lddt_all"] == pytest.approx(1.0)
    assert m["lddt_ca"] == pytest.approx(1.0)
    assert m["rmsd_all"] == pytest.approx(0.0, abs=1e-4)
    assert m["rmsd_ca"] == pytest.approx(0.0, abs=1e-4)
    assert m["n_pred_extra"] == 0.0


def test_metrics_are_invariant_to_a_rigid_transform():
    # The format places every document in a random rotated + translated frame,
    # so a prediction arrives in an arbitrary frame by construction. Nothing
    # here may depend on it.
    gt = _chain(8)
    m = score_prediction(gt, _rigid_transform(gt))
    assert m["lddt_all"] == pytest.approx(1.0)
    assert m["rmsd_all"] == pytest.approx(0.0, abs=1e-3)


def test_noise_degrades_lddt_and_rmsd_monotonically():
    gt = _chain(30)
    rng = np.random.default_rng(0)
    previous_lddt, previous_rmsd = 1.01, -1.0
    for sigma in (0.1, 0.5, 1.0, 3.0):
        noisy = gt.copy()
        noisy.coord = (gt.coord + rng.normal(0, sigma, gt.coord.shape)).astype(
            np.float32
        )
        m = score_prediction(gt, noisy)
        assert m["lddt_all"] < previous_lddt
        assert m["rmsd_all"] > previous_rmsd
        previous_lddt, previous_rmsd = m["lddt_all"], m["rmsd_all"]


def test_missing_atoms_are_penalized_by_exactly_the_broken_contacts():
    # 4 residues at 4 Å spacing: all 6 pairs are contacts.
    gt = _chain(4)

    # Drop residue 2. It breaks its 3 contacts; 3 of 6 survive.
    dropped_one = score_prediction(gt, gt[np.array([True, False, True, True])])
    assert dropped_one["atom_coverage"] == pytest.approx(0.75)
    assert dropped_one["lddt_all"] == pytest.approx(3 / 6)
    # The atoms that *were* predicted are exact, so the covered-only reading
    # is 1.0 — this is the gap the coverage columns exist to explain.
    assert dropped_one["lddt_all_covered"] == pytest.approx(1.0)

    # Drop residues 2 and 3. Only the 1-4 contact survives. Two atoms are
    # missing here, so this also pins down that the sentinel positions break
    # missing-to-missing pairs and not just missing-to-present ones.
    dropped_two = score_prediction(gt, gt[np.array([True, False, False, True])])
    assert dropped_two["lddt_all"] == pytest.approx(1 / 6)
    assert dropped_two["lddt_all_covered"] == pytest.approx(1.0)


def test_rmsd_is_covered_only_and_therefore_optimistic():
    # A predictor that emits a handful of atoms perfectly gets RMSD 0. This is
    # the documented convention, and the reason no RMSD may be read without
    # its coverage column.
    gt = _chain(20)
    partial = gt[np.arange(len(gt)) < 4]
    m = score_prediction(gt, partial)
    assert m["rmsd_all"] == pytest.approx(0.0, abs=1e-4)
    assert m["atom_coverage"] == pytest.approx(4 / 20)
    # At 4 Å spacing a pair is a contact iff |i - j| <= 3 (12 Å < 15 Å), so a
    # 20-residue chain has 19 + 18 + 17 = 54 of them and the first four
    # residues preserve the 6 among themselves.
    assert m["lddt_all"] == pytest.approx(6 / 54)


def test_extra_predicted_atoms_are_ignored_and_counted():
    gt = _chain(4)
    extra = build_atom_array(
        [(i + 1, "ALA", "CA", 4.0 * i, 0.0, 0.0, 0.0) for i in range(4)]
        # A residue index the ground truth never resolved.
        + [(99, "ALA", "CA", 500.0, 500.0, 500.0, 0.0)]
    )
    m = score_prediction(gt, extra)
    assert m["n_pred_extra"] == 1.0
    assert m["atom_coverage"] == 1.0
    assert m["lddt_all"] == pytest.approx(1.0)


def test_refined_fraction_splits_on_the_uncertainty_column():
    gt = _chain(10)
    pred = gt.copy()
    # Half the atoms claim 0.1 Å precision, half claim a 10 Å box.
    pred.b_factor = np.array([0.03] * 5 + [2.887] * 5)
    m = score_prediction(gt, pred, refined_max_sigma=1.0)
    assert m["frac_refined_of_covered"] == pytest.approx(0.5)
    assert m["frac_refined_of_gt"] == pytest.approx(0.5)
    # Refined atoms are exact here, so their own lDDT is perfect.
    assert m["lddt_all_refined"] == pytest.approx(1.0)
