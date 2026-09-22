"""Check that prior-rollout counts change contact choices as intended."""

import random

from novelty_policy import PairCandidate, effective_epsilon, sample_candidate


def test_penalty_can_select_an_unseen_contact() -> None:
    candidates = [
        PairCandidate((11, 12), (2, 20), 0.0, 0),
        PairCandidate((13, 14), (5, 30), -1.0, 1),
    ]
    counts = {(2, 20): 10}
    unpenalized, _, _ = sample_candidate(candidates, random.Random(9), counts, 0.0)
    penalized, score, prior = sample_candidate(candidates, random.Random(9), counts, 0.2)
    assert unpenalized.pair == (2, 20)
    assert penalized.pair == (5, 30)
    assert score == -1.0
    assert prior == 0


def test_early_penalty_decays_then_vanishes() -> None:
    assert effective_epsilon(0.1, 0, 20) == 0.1
    assert effective_epsilon(0.1, 10, 20) == 0.05
    assert effective_epsilon(0.1, 20, 20) == 0.0
    assert effective_epsilon(0.1, 100, 0) == 0.1
