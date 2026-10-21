"""Checks for original-chain eligibility and linker coordinate removal."""

import pytest
from score_complex_rollout_worker import parse_rollout_contacts, remove_linker_positions
from score_four_model_eval import candidate_pairs, precision_at_r


def test_linker_removal_restores_boundary_contacts() -> None:
    mapping = remove_linker_positions({100 + i: i for i in range(18)}, boundary=4)
    assert set(mapping) == set(range(100, 104)) | set(range(114, 118))
    contacts, _ = parse_rollout_contacts(
        "<contact> <p103> <p114> <contact> <p103> <p104> <contact> <p114> <p117>",
        mapping,
        [4, 4],
    )
    assert contacts == [(3, 4)]


def test_complex_bins_keep_cross_boundary_pairs_and_gt_mask() -> None:
    target = {
        "resolved_positions_by_chain": [[0, 6], [7, 13]],
        "all_contacts": [[0, 6, 0.5], [7, 13, 0.5], [6, 7, 0.5]],
    }
    assert candidate_pairs(target, "intra") == [(0, 6), (7, 13)]
    assert (6, 7) in candidate_pairs(target, "inter")
    scores = {(0, 6): 5, (7, 13): 5, (6, 7): 4, (1, 8): 100}
    assert precision_at_r(target, scores, "intra")["r_precision"] == 1
    assert precision_at_r(target, scores, "inter")["r_precision"] == 1
    assert precision_at_r(target, {}, "inter")[
        "tie_expected_r_precision"
    ] == pytest.approx(0.25)
