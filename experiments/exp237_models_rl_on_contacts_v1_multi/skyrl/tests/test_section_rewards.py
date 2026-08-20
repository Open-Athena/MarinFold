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


# --- arm M-BC: GRPO() reproduced exactly, and the blend's scale-freedom -------

def test_grpo_standardise_matches_skyrls_formula():
    """Mean 0, UNBIASED sd 1, and eps on the sd rather than the variance.

    SkyRL uses `torch.std` (ddof=1). Using numpy's default population sd would
    make the denominator 7 % too small on a group of 8 — a silent, uniform
    inflation of every advantage in the run.
    """
    v = [0.10, 0.50, 0.30, 0.70, 0.25, 0.65, 0.40, 0.55]
    a = sr.grpo_standardise(v)
    assert a.mean() == pytest.approx(0.0, abs=1e-12)
    assert a.std(ddof=1) == pytest.approx(1.0, abs=1e-5)
    expect = (np.array(v) - np.mean(v)) / (np.std(v, ddof=1) + 1e-6)
    assert np.allclose(a, expect)


def test_grpo_standardise_singleton_passes_through():
    """SkyRL takes mean 0 / std 1 for a group of one, so the reward is uncentred."""
    assert sr.grpo_standardise([0.42]) == pytest.approx([0.42])


def test_grpo_standardise_constant_group_is_zero_not_huge():
    """A degenerate group must contribute nothing, not 0/eps amplified noise."""
    assert np.allclose(sr.grpo_standardise([0.3] * 8), 0.0)


def test_blend_terms_are_separately_standardised():
    """`lam` must weight STANDARDISED terms, so the raw scales cannot decide.

    The two objectives are not commensurable: on a typical group the
    best-section F1 spreads ~4x wider than the rollout consensus. Summing them
    raw would silently hand most of the gradient to whichever happens to vary
    more — the calibration #208 got wrong twice with `lam_doc`. Standardising
    each term first makes `lam = 1` mean what it says.
    """
    best = [0.30, 0.50, 0.40, 0.60]
    cons = [0.52, 0.55, 0.51, 0.58]          # ~4x narrower raw spread
    assert np.std(best, ddof=1) > 3 * np.std(cons, ddof=1)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    separate = sr.grpo_standardise(best) + 1.0 * sr.grpo_standardise(cons)
    raw_sum = np.array(best) + 1.0 * np.array(cons)

    # Separately standardised: the two objectives pull EQUALLY on the result.
    # Tolerance 1e-4, not 1e-6: SkyRL adds eps to the standard DEVIATION, and the
    # two terms have different sds, so eps perturbs them by slightly different
    # relative amounts. The residual asymmetry is ~1e-6 and is real, not noise.
    assert corr(separate, best) == pytest.approx(corr(separate, cons), abs=1e-4)
    # Raw sum: the wider-spread term dominates, and the narrow one is nearly lost.
    assert corr(raw_sum, best) > corr(raw_sum, cons) + 0.05
    assert np.var(best) / np.var(raw_sum) > 10 * (np.var(cons) / np.var(raw_sum))


def test_blend_is_scale_free_in_section_count():
    """Neither M-BC term can be raised by emitting fewer sections.

    `max_k F1` ignores the count outright, and the rollout's own consensus FALLS
    when sections are dropped — the opposite of M-C's marginal, which is 366x
    larger at one section than at 22.
    """
    gt = {(0, 10), (1, 12), (2, 20), (3, 30)}
    full = [{(0, 10), (1, 12)}, {(2, 20)}, {(3, 30)}, {(0, 10), (2, 20)}]
    c_full, _ = sr.consensus_and_marginals(full, gt, 64)
    c_one, _ = sr.consensus_and_marginals(full[:1], gt, 64)
    assert c_one <= c_full, "dropping sections must not raise the rollout's consensus"
    # max_k F1 over a prefix can only be <= the max over the whole set.
    f_full, f_one = sr.section_f1s(full, gt), sr.section_f1s(full[:1], gt)
    assert max(f_one) <= max(f_full)


# --- arm M-FC: synthesis is a better target than selection -------------------

def test_synthesis_beats_selection_on_a_constructed_rollout():
    """The measurement that motivates M-FC, as an invariant.

    On M-B's real generations the ORACLE best single section reads 0.5646 while
    the consensus of the rollout's own preceding drafts reads 0.5750 — selecting
    one draft is *dominated* by aggregating them. This pins the mechanism that
    makes that possible: complementary drafts, none individually complete.
    """
    gt = {(0, 10), (1, 12), (2, 20), (3, 30)}
    drafts = [{(0, 10), (1, 12)}, {(2, 20), (3, 30)}, {(0, 10), (2, 20)}]
    best_single = max(sr.section_f1s(drafts, gt))
    pooled = set().union(*drafts)
    tp = len(pooled & gt)
    pooled_f1 = 2 * (tp / len(pooled)) * (tp / len(gt)) / ((tp / len(pooled)) + (tp / len(gt)))
    assert pooled_f1 > best_single, "aggregation must beat the best individual draft here"


def test_final_plus_consensus_is_a_registered_mode():
    assert "final_plus_consensus" in sr.REWARD_MODES


def test_consensus_term_opposes_a_section_count_runaway():
    """The restoring force M-F lacked.

    Arm M-F ran to 259 sections carrying ~1.4 contacts each. C(all) must FALL
    under that, so blending it in penalises exactly the direction M-F ran.
    """
    gt = {(i, i + 10) for i in range(12)}
    healthy = [{(i, i + 10) for i in range(6)}, {(i, i + 10) for i in range(4, 12)}]
    shredded = [{(i, i + 10)} for i in range(12)] + [set() for _ in range(30)]
    c_healthy, _ = sr.consensus_and_marginals(healthy, gt, 64)
    c_shredded, _ = sr.consensus_and_marginals(shredded, gt, 64)
    assert c_healthy >= c_shredded


# --- arm M-K: the deployed metric as the reward ------------------------------

def test_consensus_only_is_a_registered_mode():
    assert "consensus_only" in sr.REWARD_MODES


def test_rollout_consensus_is_scale_correct_in_section_count():
    """M-K's reward FALLS when sections are dropped — the inverse of M-C's bug.

    Measured on real generations: 0.543 at 22 sections against 0.341 at one, and
    a group-centred advantage of +0.79 vs −1.37. Here the same property is pinned
    on a constructed rollout whose drafts are complementary.
    """
    gt = {(0, 10), (1, 12), (2, 20), (3, 30), (4, 40)}
    drafts = [{(0, 10), (1, 12)}, {(2, 20), (3, 30)}, {(4, 40), (0, 10)}]
    c_all, _ = sr.consensus_and_marginals(drafts, gt, 64)
    for k in (1, 2):
        c_fewer, _ = sr.consensus_and_marginals(drafts[:k], gt, 64)
        assert c_fewer <= c_all, f"dropping to {k} sections must not raise the consensus"


# ---------------------------------------------------------------- count penalty

def test_count_penalty_deadband_is_exact_above_the_floor():
    for k in (18, 19, 22, 60):
        assert sr.count_penalty(k, beta=0.03, floor=18.0) == 0.0


def test_count_penalty_is_linear_and_negative_below_the_floor():
    assert sr.count_penalty(11, 0.03, 18.0) == pytest.approx(-0.21)
    assert sr.count_penalty(1, 0.03, 18.0) == pytest.approx(-0.51)


def test_count_penalty_off_by_default_beta():
    assert sr.count_penalty(1, beta=0.0, floor=18.0) == 0.0


def test_a_constant_penalty_across_a_group_cancels_under_grpo():
    """The deadband survives standardisation, which is why it is added RAW.

    Every rollout clearing the floor gets the same 0.0, and GRPO subtracts the
    group mean -- so a healthy batch's advantages must be *bit-identical* to the
    unpenalised ones, not merely close. Standardising the penalty on its own
    column instead would turn a group of equal zeros into 0/0 or, worse, amplify
    a one-section difference into a full unit of advantage.
    """
    f1 = [0.51, 0.44, 0.62, 0.38, 0.55, 0.47, 0.60, 0.41]
    counts = [22, 19, 25, 18, 30, 21, 24, 20]          # all >= floor
    penalised = [v + sr.count_penalty(k, 0.03, 18.0) for v, k in zip(f1, counts)]
    assert np.allclose(sr.grpo_standardise(penalised), sr.grpo_standardise(f1), atol=0, rtol=0)


def test_the_penalty_reorders_a_group_when_one_rollout_is_short():
    """A short rollout with the BEST section must still lose to a long mediocre one."""
    f1 = [0.62, 0.50, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44]
    counts = [8, 22, 21, 20, 23, 19, 24, 20]           # the 0.62 came from 8 sections
    plain = sr.grpo_standardise(f1)
    pen = sr.grpo_standardise([v + sr.count_penalty(k, 0.03, 18.0) for v, k in zip(f1, counts)])
    assert plain[0] == max(plain)                       # unpenalised: the short one wins
    assert pen[0] < 0                                   # penalised: it is below the baseline


# ------------------------------------------------------- arm M-KS: zero-sum shaping

def test_prefix_marginals_telescope_to_the_full_consensus():
    """Sum of the causal marginals is C(all) - C(empty), by construction."""
    L = 40
    gt = {(2, 20), (3, 25), (5, 30), (8, 33), (1, 18)}
    sections = [{(2, 20), (9, 40)}, {(3, 25), (2, 20)}, {(5, 30), (8, 33)}, {(1, 18)}]
    m = sr.prefix_marginals(sections, gt, L)
    total, _ = sr.consensus_and_marginals(sections, gt, L)
    assert m.sum() == pytest.approx(total, abs=1e-9)


def test_prefix_marginal_of_a_duplicate_section_is_zero():
    """Repeating what is already in context adds nothing — the point of the term."""
    L = 40
    gt = {(2, 20), (3, 25)}
    base = [{(2, 20)}, {(3, 25)}]
    m = sr.prefix_marginals(base + [{(2, 20)}], gt, L)
    assert m[-1] == pytest.approx(0.0, abs=1e-12)


def test_shaping_is_exactly_zero_sum_within_the_rollout():
    """No matter the marginals or beta, the shaping cannot move the rollout's total.

    This is the entire safety argument for arm M-KS. Every count pathology
    measured in #237 was about a reward's LEVEL as a function of the section
    count; a term whose sum is identically zero has no level to move.
    """
    for beta in (0.0, 0.5, 3.0, -2.0):
        for m in ([0.4, 0.0, 0.0, 0.1], [2.03], [0.0] * 22, [-0.1, 0.9, 0.2]):
            adv = sr.shaped_section_advantages(1.7, np.asarray(m), beta)
            assert adv.mean() == pytest.approx(1.7, abs=1e-12)
            assert (adv - 1.7).sum() == pytest.approx(0.0, abs=1e-12)


def test_beta_zero_reduces_to_arm_m_k_exactly():
    adv = sr.shaped_section_advantages(-0.63, np.asarray([0.4, 0.0, 0.1]), 0.0)
    assert np.all(adv == -0.63)


def test_shaping_ranks_the_contributing_section_above_the_duplicate():
    adv = sr.shaped_section_advantages(1.0, np.asarray([0.30, 0.0, 0.0]), beta=1.0)
    assert adv[0] > adv[1] == adv[2]
    assert adv[1] < 1.0                     # duplicates land BELOW the rollout's base


# ---------------------------------------- the positional fix (arm M-KS2)

def test_positional_baseline_handles_ragged_groups():
    """Position k averages only the rollouts that reached k — no padded zeros."""
    b = sr.positional_baseline({"a": np.array([1.0, 2.0, 3.0]), "b": np.array([3.0, 4.0])})
    assert b[0] == pytest.approx(2.0)
    assert b[1] == pytest.approx(3.0)
    assert b[2] == pytest.approx(3.0)          # only "a" reached position 2


def test_positional_correction_removes_the_stop_early_gradient():
    """The bug arm M-KS died of, and the fix, in one test.

    A group whose rollouts all share the SAME positional decay and differ in
    nothing else must get zero shaping — under the plain centring it instead
    gets a strongly positive first section and negative everything after, which
    is a direct penalty on opening another candidate.
    """
    decay = np.array([0.36, 0.01, -0.01, -0.02, -0.02, -0.02])
    group = {"a": decay.copy(), "b": decay.copy(), "c": decay.copy()}

    plain = sr.shaped_section_advantages(1.0, decay, beta=3.0)
    assert plain[0] > 1.9 and plain[-1] < 1.0          # "write one section and stop"

    b = sr.positional_baseline(group)
    fixed = sr.shaped_section_advantages(1.0, decay, beta=3.0, positional=b)
    assert np.allclose(fixed, 1.0, atol=1e-12)          # nothing to say: all identical


def test_positional_correction_still_credits_a_genuinely_better_section():
    group = {"a": np.array([0.36, 0.01, 0.20]), "b": np.array([0.36, 0.01, 0.00]),
             "c": np.array([0.36, 0.01, 0.00])}
    b = sr.positional_baseline(group)
    adv = sr.shaped_section_advantages(1.0, group["a"], beta=3.0, positional=b)
    assert adv[2] > adv[0]                              # section 2 beat its own position
    assert adv.mean() == pytest.approx(1.0, abs=1e-12)  # still zero-sum


# ------------------------------------------- arm M-KS3: novelty scored directly

def test_novelty_credits_new_true_and_debits_new_false():
    gt = {(1, 10), (2, 20), (3, 30)}
    secs = [{(1, 10)}, {(2, 20), (5, 50)}, {(1, 10)}]     # 3rd repeats, adds nothing
    m = sr.novelty_marginals(secs, gt, 60)
    assert m[0] == pytest.approx(1 / 3)                   # one new true
    assert m[1] == pytest.approx(0.0)                     # one new true, one new false
    assert m[2] == pytest.approx(0.0)                     # nothing new at all


def test_novelty_prices_padding_rather_than_rewarding_it():
    """A section that dumps junk to catch one new true pair must score NEGATIVE.

    Plain recall-gain (`new_true / R`) would score this positive, which is the
    incentive that produced arm M-F's 259 sections carrying 1.4 contacts each.
    """
    gt = {(1, 10), (2, 20)}
    junk = {(i, i + 30) for i in range(3, 13)}            # 10 new false
    m = sr.novelty_marginals([{(1, 10)}, {(2, 20)} | junk], gt, 60)
    assert m[1] < 0


def test_novelty_is_zero_for_an_exact_repeat_anywhere_in_the_rollout():
    gt = {(1, 10), (2, 20)}
    m = sr.novelty_marginals([{(1, 10)}, {(2, 20)}, {(1, 10), (2, 20)}], gt, 60)
    assert m[2] == pytest.approx(0.0)
