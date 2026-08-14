# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the consensus document term's rollout -> reward mapping — issue #208.

This is the piece that gates arms B and F, and its failure mode is the nasty
kind: a marginal attributed to the wrong rollout is still a plausible number, so
a mis-mapped run trains happily and reports nothing unusual. The tests below
therefore attack the mapping rather than the arithmetic (the arithmetic is
pinned against the published metric implementation in `test_consensus.py`).

The load-bearing test is `test_mapping_follows_trajectory_ids_not_row_order`:
`GeneratorOutput` rows are handed over SHUFFLED relative to the group order, so
any implementation that assumes "row i is rollout i" fails it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("skyrl", reason="dense_generator subclasses SkyRLGymGenerator")

import consensus as cs  # noqa: E402
from dense_generator import DenseContactsGenerator  # noqa: E402

L = 30
GT = {(0, 10), (1, 20), (5, 15)}
# Deliberately unequal quality, so the marginals differ across reps and a
# mis-mapping changes the answer rather than shuffling identical numbers.
ROLLOUTS = {
    "0": {(0, 10), (1, 20)},   # two correct
    "1": {(0, 10)},            # one correct
    "2": {(2, 9)},             # none correct
}
ROW_LEN = 4


class _TID:
    """Stand-in for SkyRL's `TrajectoryID` (a frozen dataclass upstream)."""

    def __init__(self, instance_id, repetition_id):
        self.instance_id = instance_id
        self.repetition_id = repetition_id


def _generator(lam_doc=2.0, lam_step=1.0, instances=("A", "B")):
    """A generator with group state populated but no SkyRL machinery.

    `object.__new__` skips `SkyRLGymGenerator.__init__`, which would want a
    tokenizer, an inference engine and a Ray cluster. The document term only
    touches the attributes set here.
    """
    gen = object.__new__(DenseContactsGenerator)
    gen.doc_term = "consensus"
    gen.lam_step = lam_step
    gen.lam_doc = lam_doc
    gen._doc_term_failures = 0
    gen._diag = {}
    gen._group_pairs = {inst: dict(ROLLOUTS) for inst in instances}
    gen._group_meta = {inst: {"gt": set(GT), "L": L} for inst in instances}
    return gen


def _expected_marginals():
    """Recompute the group-centred marginals independently of the generator."""
    reps = sorted(ROLLOUTS)
    pairs, position = cs.candidate_index(L)
    is_true = cs.truth_mask(pairs, GT)
    votes = cs.vote_counts([ROLLOUTS[r] for r in reps], position, len(pairs))
    _, marginals = cs.loo_marginals(votes, is_true, int(is_true.sum()))
    marginals = marginals - marginals.mean()
    return dict(zip(reps, [float(m) for m in marginals]))


def _out(rows):
    """A minimal GeneratorOutput: `rows` is a list of (instance, rep) pairs."""
    return {
        "rewards": [[0.0] * ROW_LEN for _ in rows],
        "trajectory_ids": [_TID(i, r) for i, r in rows],
    }


def test_mapping_follows_trajectory_ids_not_row_order():
    """Each rollout's marginal must land on ITS row, whatever order rows arrive in."""
    gen = _generator()
    # Shuffled: no instance is contiguous and no rep is in position order.
    rows = [("B", "2"), ("A", "0"), ("B", "0"), ("A", "2"), ("A", "1"), ("B", "1")]
    out = gen._fold_document_term(_out(rows))

    expected = _expected_marginals()
    for row, reward in zip(rows, out["rewards"]):
        _, rep = row
        share = gen.lam_doc * expected[rep] / ROW_LEN
        assert reward == pytest.approx([share] * ROW_LEN), f"wrong marginal on {row}"

    # Guard the guard: the assertion above is only meaningful if a positional
    # implementation would actually FAIL it. Spell out what "row i is rollout i"
    # would have produced (rows grouped and in rep order) and require a difference.
    applied = [row[0] for row in out["rewards"]]
    positional = [gen.lam_doc * expected[rep] / ROW_LEN for _, rep in sorted(rows)]
    assert applied != pytest.approx(positional), (
        "shuffled and positional mappings agree, so this test cannot detect the bug "
        "it exists to detect — pick rollouts whose marginals differ more"
    )


def test_row_order_permutation_changes_nothing():
    """Two orderings of the same group must produce the same per-rollout rewards."""
    rows_a = [("A", "0"), ("A", "1"), ("A", "2")]
    rows_b = [("A", "2"), ("A", "0"), ("A", "1")]
    out_a = _generator(instances=("A",))._fold_document_term(_out(rows_a))
    out_b = _generator(instances=("A",))._fold_document_term(_out(rows_b))

    by_rep_a = {r: v for (_, r), v in zip(rows_a, out_a["rewards"])}
    by_rep_b = {r: v for (_, r), v in zip(rows_b, out_b["rewards"])}
    assert by_rep_a == pytest.approx(by_rep_b)


def test_incomplete_group_is_skipped_not_silently_rescored():
    """A rollout missing from `_group_pairs` invalidates C(all) for its instance."""
    gen = _generator(instances=("A",))
    del gen._group_pairs["A"]["2"]          # one sibling never scored
    rows = [("A", "0"), ("A", "1"), ("A", "2")]
    out = gen._fold_document_term(_out(rows))
    # Stepwise reward stands, untouched, rather than a marginal from a 2-rollout group.
    assert all(r == [0.0] * ROW_LEN for r in out["rewards"])


def test_duplicate_trajectory_ids_are_loud():
    """Two rows claiming one rollout would make the mapping ambiguous."""
    gen = _generator(instances=("A",))
    rows = [("A", "0"), ("A", "0"), ("A", "1"), ("A", "2")]
    with pytest.raises(RuntimeError, match="duplicate trajectory ids"):
        gen._fold_document_term(_out(rows))


def test_missing_trajectory_ids_refuses_to_guess():
    """Without per-row ids, positional mapping is a guess — refuse it."""
    gen = _generator(instances=("A",))
    out = _out([("A", "0"), ("A", "1"), ("A", "2")])
    del out["trajectory_ids"]
    with pytest.raises(RuntimeError, match="refusing to map"):
        gen._fold_document_term(out)


def test_lam_step_scales_the_stepwise_term():
    """`lam_step` must reach the rewards; it is the other half of the blend."""
    gen = _generator(lam_doc=0.0, lam_step=3.0, instances=("A",))
    rows = [("A", "0"), ("A", "1"), ("A", "2")]
    out = _out(rows)
    out["rewards"] = [[1.0] * ROW_LEN for _ in rows]
    folded = gen._fold_document_term(out)
    assert all(r == pytest.approx([3.0] * ROW_LEN) for r in folded["rewards"])


def _collapse_generator(ratio=0.2):
    gen = _generator(instances=("A",))
    gen.collapse_ratio = ratio
    gen._pred_per_gt_baseline = None
    return gen


def test_collapse_tripwire_fires_on_the_observed_collapse():
    """The measured 8-way collapse (pred/gt 1.11 -> 0.006) must abort the run."""
    gen = _collapse_generator()
    gen._check_for_collapse(1.11, gt=140.0)          # step 0 sets the baseline
    with pytest.raises(RuntimeError, match="POLICY COLLAPSE"):
        gen._check_for_collapse(0.006, gt=140.0)


@pytest.mark.parametrize("pred_per_gt", [0.9941, 1.0208, 1.0493, 1.1033, 1.3108])
def test_collapse_tripwire_tolerates_healthy_variation(pred_per_gt):
    """Every pred/gt observed in a non-collapsed run must pass.

    These are the actual values logged by the stable 1-GPU run; a tripwire that
    fires on any of them would be worse than none, because it would train people
    to ignore it.
    """
    gen = _collapse_generator()
    gen._check_for_collapse(1.1298, gt=140.0)
    gen._check_for_collapse(pred_per_gt, gt=140.0)   # must not raise


def test_collapse_tripwire_ignores_batches_without_ground_truth():
    """pred/gt is 0/0 when no ground truth is present; that is not a collapse."""
    gen = _collapse_generator()
    gen._check_for_collapse(1.10, gt=140.0)
    gen._check_for_collapse(0.0, gt=0.0)             # must not raise
    assert gen._pred_per_gt_baseline == pytest.approx(1.10)


def test_reward_row_lengths_are_preserved():
    """A folded row must stay token-aligned with its response."""
    gen = _generator(instances=("A",))
    rows = [("A", "0"), ("A", "1"), ("A", "2")]
    out = gen._fold_document_term(_out(rows))
    assert [len(r) for r in out["rewards"]] == [ROW_LEN] * len(rows)


# --- document_f1 reward mode ------------------------------------------------
#
# The second exp208 arm: one scalar per rollout (section F1) instead of a
# per-token per-contact reward, with the baseline coming from the GROUP rather
# than from p_bar's EMA. Arm S showed why that matters -- its p_bar drifted
# ABOVE the true precision (0.5501 vs 0.4733 by the end), which makes every
# contact net-negative and shrinks the policy's output.


def test_document_f1_mode_is_validated():
    """A typo in reward_mode must not fall through to dense."""
    with pytest.raises(ValueError, match="reward_mode"):
        DenseContactsGenerator.__init__(
            object.__new__(DenseContactsGenerator), reward_mode="f1")


def test_reward_mode_and_estimator_must_agree():
    """document_f1 + contacts_dense fails minutes in; dense + grpo fails silently."""
    from main_exp208 import ADV_ESTIMATOR, check_reward_mode

    class _Cfg:
        def __init__(self, mode, est, lam_step=1.0):
            self.reward_mode = mode
            self.lam_step = lam_step
            self.trainer = type("T", (), {"algorithm": type("A", (), {"advantage_estimator": est})()})()

    with pytest.raises(ValueError, match="one scalar per rollout"):
        check_reward_mode(_Cfg("document_f1", ADV_ESTIMATOR))
    # The dangerous direction: a group estimator silently discards the dense signal.
    with pytest.raises(ValueError, match="WITHOUT error"):
        check_reward_mode(_Cfg("dense", "grpo"))
    check_reward_mode(_Cfg("document_f1", "grpo"))       # both valid pairings
    check_reward_mode(_Cfg("dense", ADV_ESTIMATOR))
    # novelty is dense per-token, so it needs the dense estimator too.
    check_reward_mode(_Cfg("novelty", ADV_ESTIMATOR))
    with pytest.raises(ValueError, match="novelty"):
        check_reward_mode(_Cfg("novelty", "grpo"))
    # lam_step=0 makes even "dense" sequence-level, which needs a group estimator.
    check_reward_mode(_Cfg("dense", "grpo", lam_step=0.0))


def test_document_term_total_is_lam_doc_times_marginal():
    """The doc term's whole contribution to a rollout is `lam_doc * marginal`.

    This is the identity the lam_doc calibration rests on: the per-token share is
    `lam_doc*marg/len(row)`, so summing over the response recovers `lam_doc*marg`
    exactly. Getting it wrong is how lam_doc=4.5 shipped -- a value that carries
    0.42% of the stepwise term's spread and made arm B a bit-for-bit rerun of
    arm S. Pin the identity so the calibration can be checked by arithmetic.
    """
    lam_doc = 7.0
    gen = _generator(lam_doc=lam_doc, instances=("A",))
    rows = [("A", "0"), ("A", "1"), ("A", "2")]
    out = gen._fold_document_term(_out(rows))
    expected = _expected_marginals()
    for (_, rep), reward in zip(rows, out["rewards"]):
        assert sum(reward) == pytest.approx(lam_doc * expected[rep]), (
            "summed document contribution must equal lam_doc * marginal")


# --- novelty-weighted reward ------------------------------------------------
#
# The synthesis of the two measured failure modes. The stepwise term alone
# SHARPENS: a contact pays the same whether or not every sibling already found
# it, so emitting only confident contacts is the cheapest way to score, and
# coverage fell 65%. The consensus marginal alone OVER-EMITS: it nets correct
# against wrong at the document level, so volume is nearly free, and at lr 4e-5
# the run diverged to KL 3.96 with precision below the base model.


def _novelty_generator(floor=0.25, p_bar=0.26):
    gen = object.__new__(DenseContactsGenerator)
    gen.reward_mode = "novelty"; gen.novelty_floor = floor; gen.p_bar = p_bar
    gen._diag = {}; gen._group_contacts = {}
    return gen


def test_novelty_pays_more_for_contacts_the_group_missed():
    """A correct contact only this rollout found must outscore a unanimous one."""
    gen = _novelty_generator()
    # rep 0 finds (0,10) alone; all three find (1,20). Both correct.
    gen._group_contacts["A"] = {
        "0": ([((0, 10), 0, True), ((1, 20), 3, True)], 6),
        "1": ([((1, 20), 0, True)], 3),
        "2": ([((1, 20), 0, True)], 3),
    }
    rows = [("A", "0"), ("A", "1"), ("A", "2")]
    out = {"rewards": [[0.0] * 6, [0.0] * 3, [0.0] * 3],
           "trajectory_ids": [_TID(i, r) for i, r in rows]}
    got = gen._apply_novelty(out)["rewards"][0]
    unique_total, shared_total = sum(got[0:3]), sum(got[3:6])
    assert unique_total > shared_total, "a contact nobody else found must pay more"
    # floor=0.25: unanimous pays 0.25 of the full (1-p_bar); unique pays all of it.
    assert shared_total == pytest.approx(0.25 * (1 - gen.p_bar))
    assert unique_total == pytest.approx(1.0 * (1 - gen.p_bar))


def test_novelty_keeps_the_full_penalty_on_wrong_contacts():
    """Volume must never be free — this is what stops the over-emission failure."""
    gen = _novelty_generator()
    gen._group_contacts["A"] = {
        "0": ([((2, 9), 0, False)], 3),
        "1": ([((5, 15), 0, False)], 3),
    }
    rows = [("A", "0"), ("A", "1")]
    out = {"rewards": [[0.0] * 3, [0.0] * 3],
           "trajectory_ids": [_TID(i, r) for i, r in rows]}
    got = gen._apply_novelty(out)["rewards"][0]
    # A wrong contact costs -p_bar regardless of how novel it is: novelty must not
    # make junk cheaper, or the reward degenerates into spam.
    assert sum(got) == pytest.approx(-gen.p_bar)
