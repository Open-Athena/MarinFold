"""Protocol tests for interpreting vLLM two-position beams."""

import random
from types import SimpleNamespace

from beam_policy import pair_candidates, sample_candidate


def test_pair_beam_keeps_complete_separated_contacts() -> None:
    prompt = [100, 101, 102]
    sequences = [
        SimpleNamespace(tokens=prompt + [7, 7], cum_logprob=-0.1),
        SimpleNamespace(tokens=prompt + [7, 9], cum_logprob=-0.2),
        SimpleNamespace(tokens=prompt + [7, 8], cum_logprob=-1.0),
        SimpleNamespace(tokens=prompt + [7], cum_logprob=-0.05),
    ]
    candidates = pair_candidates(sequences, len(prompt), {7: 0, 8: 5, 9: 9})
    assert [(item.pair, item.rank) for item in candidates] == [((0, 9), 1)]
    assert sample_candidate(candidates, random.Random(306)).token_ids == (7, 9)
