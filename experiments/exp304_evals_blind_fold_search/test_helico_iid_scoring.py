"""Checks for the structure-level scoring and mutually exclusive hit rule."""

import numpy as np
import pandas as pd

from analyze_helico_iid import calibrated_hits, hits
from score_helico_iid import gdt, superpose


def test_superpose_recovers_rigid_transform() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                          [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    predicted = reference @ rotation + np.array([8.0, -3.0, 2.0])
    aligned, rmsd = superpose(predicted, reference)
    expected_rmsd = np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1)))
    assert rmsd == expected_rmsd
    assert rmsd < 1e-10
    assert gdt(np.linalg.norm(aligned - reference, axis=1)) == 1.0


def test_structural_hits_require_absolute_quality_and_reference_margin() -> None:
    frame = pd.DataFrame({
        "marinfold_finished": [True, True, True, False],
        "gdt_common_fold1": [0.70, 0.62, 0.40, 0.80],
        "gdt_common_fold2": [0.55, 0.72, 0.30, 0.10],
        "gdt_region_global_fit_fold1": [0.65, 0.25, 0.70, 0.90],
        "gdt_region_global_fit_fold2": [0.35, 0.55, 0.20, 0.10],
    })
    fold1, fold2 = hits(frame, global_gdt=0.50, region_margin=0.10)
    assert fold1.tolist() == [True, False, False, False]
    assert fold2.tolist() == [False, True, False, False]


def test_calibrated_hits_require_a_successful_true_contact_control() -> None:
    maps = pd.DataFrame({
        "marinfold_finished": [True, True],
        "gdt_common_fold1": [0.55, 0.20], "gdt_common_fold2": [0.20, 0.55],
        "gdt_region_global_fit_fold1": [0.60, 0.20],
        "gdt_region_global_fit_fold2": [0.20, 0.60],
    })
    controls = pd.DataFrame({
        "kind": ["true_fold1", "true_fold2"], "marinfold_finished": [True, True],
        "gdt_common_fold1": [0.60, 0.30], "gdt_common_fold2": [0.20, 0.30],
        "gdt_region_global_fit_fold1": [0.60, 0.40],
        "gdt_region_global_fit_fold2": [0.20, 0.40],
    })
    fold1, fold2, control1, control2, threshold1, threshold2 = calibrated_hits(maps, controls)
    assert fold1.tolist() == [True, False]
    assert not fold2.any()
    assert control1 is True and control2 is False
    assert threshold1 == 0.54 and threshold2 == 0.35
