"""Protect fixed-length control semantics and the nested-prefix comparison."""

import torch

from engine import Policy
from length_probe import advance_to_contacts
from test_engine import tiny_engine


def test_length_control_ignores_eos_and_scoring_does_not_change_continuation() -> None:
    engine = tiny_engine()
    prompt = [14, 26, 32, 2002]
    policy = Policy('top10', candidates=4, fraction=.25)
    rng = torch.Generator().manual_seed(42)
    first = advance_to_contacts(engine, engine.start(prompt), policy, 24, 0, rng, 4)
    tokens = first.tokens.clone()
    cache_length = first.cache.get_seq_length()
    engine.score(first)
    second = advance_to_contacts(engine, first, policy, 24, 0, rng, 8)
    direct = advance_to_contacts(engine, engine.start(prompt), policy, 24, 0,
                                 torch.Generator().manual_seed(42), 8)
    torch.testing.assert_close(second.tokens, direct.tokens, rtol=0, atol=0)
    torch.testing.assert_close(first.tokens, tokens, rtol=0, atol=0)
    assert first.cache.get_seq_length() == cache_length
    assert first.count == 4 and second.count == 8
    assert not first.finished and not second.finished
    assert engine.end_id not in second.tokens[0, len(prompt):].tolist()
