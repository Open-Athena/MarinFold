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
        def __init__(self, mode, est):
            self.reward_mode = mode
            self.trainer = type("T", (), {"algorithm": type("A", (), {"advantage_estimator": est})()})()

    with pytest.raises(ValueError, match="one scalar per rollout"):
        check_reward_mode(_Cfg("document_f1", ADV_ESTIMATOR))
    # The dangerous direction: a group estimator silently discards the dense signal.
    with pytest.raises(ValueError, match="WITHOUT error"):
        check_reward_mode(_Cfg("dense", "grpo"))
    check_reward_mode(_Cfg("document_f1", "grpo"))       # both valid pairings
    check_reward_mode(_Cfg("dense", ADV_ESTIMATOR))
