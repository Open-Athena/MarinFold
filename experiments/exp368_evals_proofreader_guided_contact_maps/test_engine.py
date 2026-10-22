"""Check cache isolation and exact readout semantics on a small real Qwen3."""

import torch
from torch import nn
from transformers import Qwen3Config, Qwen3Model

from engine import Engine, Policy, select, selection_weights
from runtime import load_release


class Tokenizer:
    def convert_tokens_to_ids(self, value: str) -> int:
        if value.startswith('<p'):
            return int(value[2:-1]) + 2
        return {'<contact>': 0, '<end>': 1, '<begin_statements>': 2002, '<assess>': 2003}[value]


def tiny_engine() -> Engine:
    torch.manual_seed(368)
    api = load_release()
    config = Qwen3Config(vocab_size=2010, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=512, layer_types=['full_attention'] * 2)
    reader = api.model.CausalContactProofreader(Qwen3Model(config), 2002,
        readout_width=32, readout_layers=2, readout_heads=4)
    return Engine(reader, nn.Linear(32, 2003, bias=False), Tokenizer(), device='cpu')


def test_cached_readout_and_fork_isolation() -> None:
    engine = tiny_engine()
    prompt = [14, 26, 32, 2002]
    parent = engine.start(prompt)
    before = [(key.clone(), value.clone()) for key, value, *_ in parent.cache]
    branches = select(parent, torch.tensor([0, 0, 0]))
    rng = torch.Generator().manual_seed(9)
    for _ in range(4):
        branches = engine.extend(branches, 24, 0, rng)
    assert parent.tokens.tolist() == [prompt]
    assert parent.cache.get_seq_length() == len(prompt)
    for (key, value, *_), (original_k, original_v) in zip(parent.cache, before, strict=True):
        torch.testing.assert_close(key, original_k, rtol=0, atol=0)
        torch.testing.assert_close(value, original_v, rtol=0, atol=0)
    actual, recall = engine.score(branches)
    ids = torch.cat([branches.tokens, torch.full((3, 1), 2003)], dim=1)
    ends = torch.arange(len(prompt) + 2, ids.shape[1] - 1, 3)[None].expand(3, -1)
    with torch.inference_mode():
        logits, expected_recall = engine.reader(ids, torch.ones_like(ids, dtype=torch.bool), ends)
    torch.testing.assert_close(actual, logits.sigmoid(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(recall, expected_recall, rtol=1e-5, atol=1e-6)
    for pairs in branches.pairs.tolist():
        assert all(b - a >= 6 and 0 <= a < b < 24 for a, b in pairs)
        assert len(set(map(tuple, pairs))) == len(pairs)
    finished = engine.finish(select(branches, torch.tensor([1])))
    actual, recall = engine.score(finished)
    ids = torch.cat([finished.tokens, torch.full((1, 1), 2003)], dim=1)
    with torch.inference_mode():
        logits, expected_recall = engine.reader(ids, torch.ones_like(ids, dtype=torch.bool), ends[:1])
    torch.testing.assert_close(actual, logits.sigmoid(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(recall, expected_recall, rtol=1e-5, atol=1e-6)


def test_resampling_uses_only_reader_and_preserves_exploration() -> None:
    scores = torch.tensor([.1, .2, .3, .4])
    torch.testing.assert_close(selection_weights(scores, Policy('soft', beta=1)), scores)
    torch.testing.assert_close(selection_weights(scores, Policy('control')), torch.ones(4) / 4)
    torch.testing.assert_close(selection_weights(scores, Policy('top', fraction=.25)),
                               torch.tensor([0., 0., 0., 1.]))
    torch.testing.assert_close(selection_weights(scores, Policy('explore', fraction=.25, exploration=.1)),
                               torch.tensor([.025, .025, .025, .925]))


def test_lookahead_commits_only_candidate_and_repair_preserves_prefix() -> None:
    engine = tiny_engine()
    prompt = [14, 26, 32, 2002]
    rng = torch.Generator().manual_seed(22)
    parent = engine.extend(engine.start(prompt), 24, 0, rng)
    policy = Policy('look', candidates=5, candidate_batch=2, fraction=.2, lookahead=3, repeats=2)
    child = engine.choose(parent, policy, 24, 0, rng, max_contacts=8)
    assert child.count == 2
    assert child.tokens.shape[1] == len(prompt) + 6
    assert parent.count == 1
    rebuilt = engine.reencode(prompt, child.tokens[0, len(prompt):], child.pairs)
    for actual, expected in zip(engine.score(child), engine.score(rebuilt), strict=True):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
