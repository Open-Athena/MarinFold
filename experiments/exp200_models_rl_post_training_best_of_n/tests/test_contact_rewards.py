# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exp200's dense per-contact reward.

The parity test against exp163's ``rollout_metrics.score_rollout`` is the
important one: exp200's document-level return has to be the SAME number the
published #163 best-of-N figures were computed from, or the primary metric is
not comparable to its own baseline.
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

import contact_rewards as cr

V = cr.DEFAULT_VOCAB


def _load_exp163_rollout_metrics():
    """Import exp163's scorer from the sibling experiment dir, or skip."""
    path = (
        Path(__file__).resolve().parents[2]
        / "exp163_models_teach_contacts_v1_to_refine_a"
        / "rollout_metrics.py"
    )
    if not path.exists():
        pytest.skip(f"exp163 rollout_metrics.py not found at {path}")
    spec = importlib.util.spec_from_file_location("exp163_rollout_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_response(sections: list[list[tuple[int, int]]], *, terminate: bool = True):
    """Build (token_ids, text) for a completion whose prompt ended on <begin_statements>.

    ``sections`` holds POSITION indices (``<pN>``), not sequence indices.
    """
    ids: list[int] = []
    parts: list[str] = []
    for k, section in enumerate(sections):
        if k > 0:
            ids.append(V.begin_id)
            parts.append("<begin_statements>")
        for a, b in section:
            ids += [V.contact_id, V.p0_id + a, V.p0_id + b]
            parts.append(f"<contact> <p{a}> <p{b}>")
    if terminate:
        ids.append(V.end_id)
        parts.append("<end>")
    return ids, " ".join(parts)


IDENTITY_MAP = {i: i for i in range(500)}


def test_all_correct_gets_positive_reward_only():
    gt = {(0, 10), (1, 20), (2, 30)}
    ids, _ = build_response([[(0, 10), (1, 20), (2, 30)]])
    out = cr.dense_rewards(ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.25)

    assert out.diagnostics["n_contacts_correct"] == 3
    assert out.diagnostics["precision"] == 1.0
    assert out.episode_reward == pytest.approx(1.0)
    # 3 contacts x 3 tokens, each (1 - 0.25)/3
    nonzero = out.token_rewards[out.token_rewards != 0]
    assert len(nonzero) == 9
    assert np.allclose(nonzero, 0.75 / 3)


def test_penalty_decays_within_a_section():
    gt: set[tuple[int, int]] = set()
    ids, _ = build_response([[(0, 10), (1, 20), (2, 30)]])
    out = cr.dense_rewards(
        ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.25, err_decay=0.5
    )
    # gt empty -> F1 undefined -> document return falls back to 0.0
    assert out.episode_reward == 0.0
    # First error full penalty, then halved, then quartered.
    per_contact = [out.token_rewards[i : i + 3].sum() for i in (0, 3, 6)]
    assert per_contact == pytest.approx([-0.25, -0.125, -0.0625])


def test_err_decay_extremes():
    gt: set[tuple[int, int]] = set()
    ids, _ = build_response([[(0, 10), (1, 20), (2, 30)]])

    flat = cr.dense_rewards(ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.4, err_decay=1.0)
    assert [flat.token_rewards[i : i + 3].sum() for i in (0, 3, 6)] == pytest.approx([-0.4] * 3)

    first_only = cr.dense_rewards(
        ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.4, err_decay=0.0
    )
    assert [first_only.token_rewards[i : i + 3].sum() for i in (0, 3, 6)] == pytest.approx(
        [-0.4, 0.0, 0.0]
    )


def test_decay_counter_resets_per_section():
    gt: set[tuple[int, int]] = set()
    ids, _ = build_response([[(0, 10)], [(1, 20)]])
    out = cr.dense_rewards(
        ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.5, err_decay=0.5
    )
    # Section 0's contact at 0; <begin_statements> at 3; section 1's contact at 4.
    assert out.token_rewards[0:3].sum() == pytest.approx(-0.5)
    assert out.token_rewards[4:7].sum() == pytest.approx(-0.5)


def test_duplicate_within_section_counts_as_incorrect():
    gt = {(0, 10)}
    ids, _ = build_response([[(0, 10), (0, 10)]])
    out = cr.dense_rewards(ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.25)

    assert out.diagnostics["n_duplicate"] == 1
    assert out.token_rewards[0:3].sum() == pytest.approx(0.75)
    assert out.token_rewards[3:6].sum() == pytest.approx(-0.25)
    # ...but the prediction SET still has one element, so F1 is unaffected.
    assert out.episode_reward == pytest.approx(1.0)


def test_same_pair_in_a_later_section_is_not_a_duplicate():
    gt = {(0, 10)}
    ids, _ = build_response([[(0, 10)], [(0, 10)]])
    out = cr.dense_rewards(ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.25)
    assert out.diagnostics["n_duplicate"] == 0
    assert out.diagnostics["n_contacts_correct"] == 2


def test_too_close_and_unmapped_are_incorrect_but_not_predictions():
    gt = {(0, 10)}
    # (0, 3) is below MIN_SEP; (0, 400) maps to a position outside the map.
    ids, _ = build_response([[(0, 3), (0, 400)]])
    partial_map = {i: i for i in range(100)}
    out = cr.dense_rewards(ids, partial_map, gt, mode="multi", precision_baseline=0.5)

    assert out.diagnostics["n_too_close"] == 1
    assert out.diagnostics["n_unmapped"] == 1
    assert out.diagnostics["n_pred"] == 0
    assert out.token_rewards.sum() < 0


def test_malformed_and_truncated_triples():
    # <contact> followed by a non-position token, then a bare trailing <contact>.
    ids = [V.contact_id, V.begin_id, V.p0_id + 1, V.contact_id]
    out = cr.dense_rewards(ids, IDENTITY_MAP, {(0, 10)}, mode="multi", precision_baseline=0.5)
    assert out.diagnostics["n_malformed"] == 1
    # A trailing <contact> with no room for its two positions is the sampler's
    # truncation, not the policy's mistake — no penalty.
    assert out.diagnostics["n_truncated"] == 1
    assert out.token_rewards[3] == 0.0


def test_plain_mode_scores_only_the_first_section():
    gt = {(0, 10), (1, 20)}
    ids, _ = build_response([[(0, 10)], [(1, 20)]])
    out = cr.dense_rewards(ids, IDENTITY_MAP, gt, mode="plain", precision_baseline=0.25)

    assert out.scored_sections == 1
    assert out.diagnostics["n_sections_raw"] == 2
    # Section 1's contact gets no stepwise reward at all.
    assert out.token_rewards[4:7].sum() == 0.0
    assert out.token_rewards[0:3].sum() == pytest.approx(0.75)
    assert out.section_f1 == pytest.approx([2 / 3])


def test_max_sections_cap_matches_the_sampler_contract():
    gt = {(0, 10), (1, 20), (2, 30)}
    ids, _ = build_response([[(0, 10)], [(1, 20)], [(2, 30)]])
    out = cr.dense_rewards(
        ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.25, max_sections=2
    )
    assert out.scored_sections == 2
    assert out.diagnostics["n_sections_raw"] == 3
    assert len(out.section_f1) == 2
    # The third section is outside the scored contract.
    assert out.token_rewards[8:11].sum() == 0.0


def test_episode_reward_is_best_of_sections_not_last():
    gt = {(0, 10), (1, 20), (2, 30)}
    # Section 0 nails 2/3; section 1 gets nothing. best > last is the whole point.
    ids, _ = build_response([[(0, 10), (1, 20)], [(50, 100)]])
    out = cr.dense_rewards(ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.25)
    assert out.diagnostics["last_f1"] == pytest.approx(0.0)
    assert out.episode_reward == pytest.approx(0.8)
    assert out.episode_reward > out.diagnostics["last_f1"]


def test_rewards_land_on_the_contact_triple_positions():
    gt = {(1, 20)}
    ids, _ = build_response([[(0, 10), (1, 20)]])
    out = cr.dense_rewards(ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.25)
    # Triple 0 at 0..2 (wrong), triple 1 at 3..5 (right), <end> at 6 untouched.
    assert np.all(out.token_rewards[0:3] < 0)
    assert np.all(out.token_rewards[3:6] > 0)
    assert out.token_rewards[6] == 0.0
    assert len(out.token_rewards) == len(ids)


def test_precision_baseline_makes_expected_reward_vanish():
    """At p_bar == the realized precision, the stepwise term nets ~zero.

    This is the property that stops the policy collapsing to empty sections, so
    it gets an explicit test rather than a comment.
    """
    gt = {(0, 10)}
    ids, _ = build_response([[(0, 10), (1, 20), (2, 30), (3, 40)]])
    out = cr.dense_rewards(
        ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.25, err_decay=1.0
    )
    assert out.diagnostics["precision"] == pytest.approx(0.25)
    assert out.token_rewards.sum() == pytest.approx(0.0, abs=1e-6)


def test_starts_in_section_false_ignores_leading_contacts():
    gt = {(0, 10)}
    ids, _ = build_response([[(0, 10)], [(0, 10)]])
    out = cr.dense_rewards(
        ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.25, starts_in_section=False
    )
    # With a full document the header precedes section 1, so the first chunk is
    # not a section: only the post-<begin_statements> contact is section 0.
    assert out.scored_sections == 1


def test_rejects_bad_hyperparameters():
    ids, _ = build_response([[(0, 10)]])
    with pytest.raises(ValueError):
        cr.dense_rewards(ids, IDENTITY_MAP, set(), mode="multi", precision_baseline=1.5)
    with pytest.raises(ValueError):
        cr.dense_rewards(ids, IDENTITY_MAP, set(), mode="multi", precision_baseline=0.2, err_decay=2.0)
    with pytest.raises(ValueError):
        cr.dense_rewards(ids, IDENTITY_MAP, set(), mode="bogus", precision_baseline=0.2)


@pytest.mark.parametrize(
    "sections",
    [
        [[(0, 10), (1, 20), (2, 30)]],
        [[(0, 10)], [(1, 20), (2, 30)], [(3, 40)]],
        [[(0, 10), (0, 10)], [(5, 50)]],
        [[(0, 3), (0, 10)], [(2, 30)]],
        [[]],
    ],
)
def test_section_f1_matches_exp163_scorer(sections):
    """exp200's per-section F1 must equal exp163's published metric, exactly."""
    rm = _load_exp163_rollout_metrics()
    gt = {(0, 10), (1, 20), (5, 50)}

    ids, text = build_response(sections)
    mine = cr.dense_rewards(ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.3)

    gtb = rm.gt_by_band(gt)
    theirs = [rm.score_rollout(p, gtb)["all_f1"] for p in rm.parse_sections(text, IDENTITY_MAP)]

    assert len(mine.section_f1) == len(theirs)
    for a, b in zip(mine.section_f1, theirs):
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b)


def test_best_f1_matches_exp163_best_of_n():
    rm = _load_exp163_rollout_metrics()
    gt = {(0, 10), (1, 20), (5, 50), (7, 70)}
    sections = [[(0, 10)], [(0, 10), (1, 20), (5, 50)], [(9, 99)]]

    ids, text = build_response(sections)
    mine = cr.dense_rewards(ids, IDENTITY_MAP, gt, mode="multi", precision_baseline=0.3)

    gtb = rm.gt_by_band(gt)
    theirs = max(rm.score_rollout(p, gtb)["all_f1"] for p in rm.parse_sections(text, IDENTITY_MAP))
    assert mine.episode_reward == pytest.approx(theirs)
