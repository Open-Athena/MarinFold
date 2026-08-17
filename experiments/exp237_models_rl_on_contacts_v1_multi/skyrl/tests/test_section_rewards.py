# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for exp237's section-level rewards — issue #237.

These pin the four things whose failure would be silent: the section/token
alignment (a reward landing on the wrong candidate reads as noise, not as a bug),
the ``<end>`` truncation, the centring identity ``E[A] = 0`` that #208's
reward-design invariant demands, and the token-broadcast scale.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import contact_rewards as cr  # noqa: E402
import section_rewards as sr  # noqa: E402

BEGIN, END, CONTACT = cr.BEGIN_STATEMENTS_ID, cr.END_ID, cr.CONTACT_ID
P0 = cr.P0_ID


def pos(n: int) -> int:
    return P0 + n


def contact(i: int, j: int) -> list[int]:
    return [CONTACT, pos(i), pos(j)]


IDENTITY = {i: i for i in range(64)}


def test_section_bounds_tile_the_response():
    ids = contact(0, 10) + [BEGIN] + contact(1, 11) + contact(2, 12) + [BEGIN] + contact(3, 13)
    bounds = sr.section_bounds(ids)
    assert bounds == [(0, 3), (3, 10), (10, 14)]
    # The spans must partition the response exactly: a gap is a token with no
    # advantage and an overlap is a token with two.
    covered = [t for a, b in bounds for t in range(a, b)]
    assert covered == list(range(len(ids)))


def test_section_index_agrees_with_walk_contacts():
    """`walk_contacts`' section tag and `section_bounds`' span must be the same k."""
    ids = contact(0, 10) + [BEGIN] + contact(1, 11) + [BEGIN] + contact(2, 12)
    bounds = sr.section_bounds(ids)
    for c in cr.walk_contacts(ids, IDENTITY, set()):
        start, end = bounds[c.section]
        assert start <= c.start < end, f"contact at {c.start} tagged section {c.section}"


def test_end_truncates_the_scored_region():
    """A rollout that runs on past `<end>` must not fold the continuation in."""
    ids = contact(0, 10) + [END] + [BEGIN] + contact(1, 11)
    assert sr.scored_length(ids) == 4
    walk = sr.walk_rollout(ids, IDENTITY, {(0, 10)})
    assert walk.n_sections == 1
    assert walk.sections[0] == {(0, 10)}
    assert walk.finished
    adv = sr.token_advantages(np.array([1.0]), walk.bounds, walk.n_response_tokens)
    assert adv.shape == (len(ids),)
    assert np.array_equal(adv, np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.float32))


def test_walk_assigns_contacts_to_the_right_sections():
    ids = contact(0, 10) + [BEGIN] + contact(1, 11) + contact(2, 12) + [END]
    walk = sr.walk_rollout(ids, IDENTITY, {(0, 10), (2, 12)})
    assert [sorted(s) for s in walk.sections] == [[(0, 10)], [(1, 11), (2, 12)]]
    assert walk.n_scored == 3
    assert walk.n_correct == 2


def test_duplicate_within_a_section_is_not_a_second_vote():
    """A section is a SET of pairs; repeating one must not double its vote."""
    ids = contact(0, 10) + contact(0, 10)
    walk = sr.walk_rollout(ids, IDENTITY, set())
    assert walk.sections[0] == {(0, 10)}
    assert walk.diagnostics["total_votes"] == 1.0


def test_marginal_is_zero_for_a_duplicate_section():
    """The property arm M-C is built on: a section that adds nothing scores nothing.

    Two identical sections plus a third: removing either duplicate leaves the
    other's votes in place, so the consensus is unchanged and the marginal is 0.
    """
    gt = {(0, 10), (1, 12), (2, 20)}
    sections = [{(0, 10), (1, 12)}, {(0, 10), (1, 12)}, {(2, 20)}]
    _, marg = sr.consensus_and_marginals(sections, gt, 32)
    assert marg[0] == pytest.approx(0.0)
    assert marg[1] == pytest.approx(0.0)


def test_centring_makes_the_expectation_exactly_zero():
    """#208's reward-design invariant, at the section level.

    ``E[A] = 0`` over the whole prompt group is what stops the reward being a
    one-sided pressure on section count. It has to hold exactly, not approximately.
    """
    marginals = {"0": np.array([0.02, 0.0, -0.01, 0.0]), "1": np.array([0.0, 0.05, 0.0])}
    centred = sr.centred_section_advantages(marginals)
    pooled = np.concatenate(list(centred.values()))
    assert pooled.mean() == pytest.approx(0.0, abs=1e-12)
    assert pooled.std() == pytest.approx(1.0, abs=1e-9)


def test_zero_spread_group_yields_zero_not_infinity():
    """A prompt whose sections all score alike contributes nothing, deliberately.

    Dividing by ``eps`` instead would amplify float noise into a full-scale
    gradient on the least informative prompts in the batch.
    """
    centred = sr.centred_section_advantages({"0": np.zeros(5), "1": np.zeros(3)})
    assert all(np.all(v == 0.0) for v in centred.values())


def test_token_advantage_is_broadcast_not_spread():
    """Every token of a section carries the FULL advantage.

    Spreading ``A_k`` over the section's tokens would make M-C's gradient ~300x
    smaller than a GRPO scalar at the same learning rate — #208's `lam_doc` 4.5
    failure in the other direction.
    """
    adv = sr.token_advantages(np.array([2.0, -1.0]), [(0, 3), (3, 8)], 8)
    assert np.array_equal(adv, np.array([2, 2, 2, -1, -1, -1, -1, -1], dtype=np.float32))


def test_scalar_rewards_select_the_right_section():
    gt = {(0, 10), (1, 12), (2, 20)}
    ids = (contact(0, 10) + contact(1, 12) + contact(2, 20)      # section 0: perfect
           + [BEGIN] + contact(3, 30))                            # section 1: wrong
    walk = sr.walk_rollout(ids, IDENTITY, gt)
    assert sr.scalar_reward("best_f1", walk, gt) == pytest.approx(1.0)
    assert sr.scalar_reward("final_f1", walk, gt) == pytest.approx(0.0)


def test_scalar_reward_floor_is_zero_not_nan():
    """An empty rollout must score 0.0 so a GRPO baseline reads it as negative."""
    walk = sr.walk_rollout([], IDENTITY, {(0, 10)})
    assert sr.scalar_reward("final_f1", walk, {(0, 10)}) == 0.0
    assert sr.scalar_reward("best_f1", walk, {(0, 10)}) == 0.0


def test_consensus_matches_a_hand_computed_vote():
    """Sanity-check the within-rollout consensus against the metric's own rule.

    Three sections all vote for (0, 10); one also votes for (1, 12). R = 2 true
    contacts, so the top-2 by vote count is [(0,10) with 3, (1,12) with 1] and
    both are true -> consensus 1.0.
    """
    gt = {(0, 10), (1, 12)}
    sections = [{(0, 10)}, {(0, 10)}, {(0, 10), (1, 12)}]
    consensus, _ = sr.consensus_and_marginals(sections, gt, 32)
    assert consensus == pytest.approx(1.0)


def test_no_ground_truth_is_nan_not_a_crash():
    consensus, marg = sr.consensus_and_marginals([{(0, 10)}], set(), 32)
    assert np.isnan(consensus)
    assert marg.shape == (1,)
    assert not np.any(marg)


def test_diagnostics_report_the_diversity_gates():
    """union pairs / total votes / votes-per-pair are #237's reported columns."""
    sections = [{(0, 10), (1, 12)}, {(0, 10)}, {(2, 20)}]
    ids: list[int] = []
    for k, sec in enumerate(sections):
        if k:
            ids.append(BEGIN)
        for i, j in sorted(sec):
            ids += contact(i, j)
    walk = sr.walk_rollout(ids, IDENTITY, set())
    d = walk.diagnostics
    assert d["n_sections"] == 3
    assert d["union_pairs"] == 3          # (0,10), (1,12), (2,20)
    assert d["total_votes"] == 4
    assert d["votes_per_pair"] == pytest.approx(4 / 3)
