# Copyright The MarinFold Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pure backtracking engine (#159), no GPU / no marinfold model.

Run from the marinfold/ dir (for the `read` fold import)::

    uv run pytest ../experiments/exp159_data_backtracking_corpus/test_backtrack_engine.py -q

A stub Proposer scripts the base model's contact proposals; a stub Scorer
plants a low belief on chosen pairs so the posterior-collapse trigger fires.
The load-bearing invariant checked everywhere: the rendered structure section
folds (via the real `read.live_contacts`) to exactly GT.
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))

from backtrack_engine import (  # noqa: E402
    RetractionPolicy,
    build_backtracking_structure,
    canon,
)

from marinfold.document_structures.contacts_v1.read import live_contacts  # noqa: E402


class ScriptedProposer:
    """Emits a fixed list of pairs (skipping any already live), then stops."""

    def __init__(self, script):
        self._script = [canon(*p) for p in script]
        self._i = 0

    def propose(self, live):
        live = set(live)
        while self._i < len(self._script):
            p = self._script[self._i]
            self._i += 1
            if p not in live:
                return p
        return None


class StubbornProposer:
    """Always proposes the same pair (adversarial: exercises the loop guard)."""

    def __init__(self, pair):
        self._pair = canon(*pair)

    def propose(self, live):
        return self._pair


class PlantedScorer:
    """High belief for everyone except ``low`` pairs (which read below floor)."""

    def __init__(self, low, high=0.9, low_val=1e-6):
        self._low = {canon(*p) for p in low}
        self._high = high
        self._low_val = low_val

    def score(self, committed, targets):
        return {
            canon(*t): (self._low_val if canon(*t) in self._low else self._high)
            for t in targets
        }


def _fold(res):
    """Fold the rendered structure section back to a live pair set."""
    return live_contacts(res.render_structure())


def test_happy_path_final_equals_gt_and_folds():
    gt = frozenset({(1, 10), (2, 20), (3, 30)})
    fps = [(5, 50), (6, 60)]
    proposer = ScriptedProposer([(1, 10), (5, 50), (2, 20), (6, 60), (3, 30)])
    scorer = PlantedScorer(low=fps)
    policy = RetractionPolicy(min_delay=1, eval_cadence=1, tau=0.35, s_floor=1e-3)

    res = build_backtracking_structure(
        gt, proposer, scorer, policy, max_statements=100, rng=random.Random(0)
    )

    assert res.correct                      # final live set == GT
    assert res.live_final == gt
    assert _fold(res) == gt                 # the real fold agrees
    assert not res.truncated
    # Both false positives were emitted then removed.
    for fp in fps:
        assert canon(*fp) not in res.live_final
    assert res.n_retract_statements >= 2
    # At least one FP left via the posterior trigger (not only the flush).
    assert any(t in ("collapse", "floor") for *_, t in res.retractions)


def test_min_delay_respected():
    gt = frozenset({(1, 10), (2, 20), (3, 30), (4, 40)})
    proposer = ScriptedProposer(
        [(1, 10), (9, 90), (2, 20), (3, 30), (4, 40)]
    )
    scorer = PlantedScorer(low=[(9, 90)])
    policy = RetractionPolicy(min_delay=3, eval_cadence=1, s_floor=1e-3)

    res = build_backtracking_structure(
        gt, proposer, scorer, policy, max_statements=100, rng=random.Random(1)
    )
    assert res.correct and _fold(res) == gt
    for pair, delay, _was_true, trigger in res.retractions:
        if trigger != "flush":
            assert delay >= policy.min_delay


def test_missing_gt_is_appended_at_flush():
    # Proposer never emits (4, 40); the flush must add it to reach GT.
    gt = frozenset({(1, 10), (2, 20), (4, 40)})
    proposer = ScriptedProposer([(1, 10), (2, 20)])
    scorer = PlantedScorer(low=[])
    res = build_backtracking_structure(
        gt, proposer, scorer, RetractionPolicy(), max_statements=100, rng=random.Random(2)
    )
    assert res.correct and _fold(res) == gt
    assert (4, 40) in res.live_final


def test_loop_guard_bans_repeated_fp_and_terminates():
    gt = frozenset({(1, 10), (2, 20)})
    proposer = StubbornProposer((7, 70))  # a persistent false positive
    scorer = PlantedScorer(low=[(7, 70)])
    policy = RetractionPolicy(min_delay=0, eval_cadence=1, s_floor=1e-3, loop_cap=2)

    res = build_backtracking_structure(
        gt, proposer, scorer, policy, max_statements=200, rng=random.Random(3)
    )
    # Terminates (no infinite loop), ends correct, and the FP is gone.
    assert res.correct and _fold(res) == gt
    assert (7, 70) not in res.live_final
    # It was retracted at most loop_cap times before being banned.
    cycles = sum(1 for p, *_ in res.retractions if p == (7, 70))
    assert cycles <= policy.loop_cap


def test_noise_retraction_of_true_contact_is_reemitted():
    gt = frozenset({(1, 10), (2, 20), (3, 30)})
    proposer = ScriptedProposer([(1, 10), (2, 20), (3, 30)])
    scorer = PlantedScorer(low=[])  # nothing collapses on its own
    # Force noise retraction on every true contact; all must return by <end>.
    policy = RetractionPolicy(min_delay=0, eval_cadence=1, noise_retract_prob=1.0)

    res = build_backtracking_structure(
        gt, proposer, scorer, policy, max_statements=100, rng=random.Random(4)
    )
    assert res.correct and _fold(res) == gt      # recall preserved
    assert any(was_true for *_, was_true, _t in
               [(p, d, w, t) for (p, d, w, t) in res.retractions])
    assert res.n_reemit >= 1


def test_truncation_flag_when_budget_too_small():
    gt = frozenset({(i, i + 10) for i in range(1, 8)})  # 7 contacts
    proposer = ScriptedProposer([(i, i + 10) for i in range(1, 8)])
    scorer = PlantedScorer(low=[])
    # Far too small a budget to hold all of GT.
    res = build_backtracking_structure(
        gt, proposer, scorer, RetractionPolicy(), max_statements=4, rng=random.Random(5)
    )
    assert res.truncated
    assert res.live_final.issubset(gt)   # never asserts a non-GT pair as final
