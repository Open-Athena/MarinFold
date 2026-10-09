"""Behavioral tests for bidirectional evidence, cropping and masked readouts."""

import copy

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from model import Proofreader, loss_and_metrics
from records import Example, collate, make_example, parse_contacts, prefix_length


@pytest.fixture
def reader() -> Proofreader:
    torch.manual_seed(356)
    torch.set_num_threads(2)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, attention_dropout=0.0, max_position_embeddings=128)
    config._attn_implementation = 'sdpa'
    return Proofreader(Qwen3Model(config)).eval()


def tensors(batch: dict) -> dict:
    return {k: torch.from_numpy(v) for k, v in batch.items()}


def test_later_contacts_change_earlier_judgment(reader):
    inputs = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
    mask = torch.ones_like(inputs, dtype=torch.bool)
    positions = torch.tensor([[3, 6]])
    original, _ = reader(inputs, mask, positions)
    changed = inputs.clone()
    changed[0, 5] = 25
    revised, _ = reader(changed, mask, positions)
    assert abs(float((original[0, 0] - revised[0, 0]).detach())) > 1e-5


def test_padding_cannot_change_real_readouts(reader):
    one = Example([2, 3, 4, 5], [2], [1.0], [True], 0.5, 1.0, 'a')
    two = Example([2, 8, 9, 10, 11, 12, 13, 14], [3, 6], [0.0, 1.0], [True, True], 0.5, 0.5, 'b')
    alone = tensors(collate([one], 0))
    padded = tensors(collate([one, two], 0))
    with torch.no_grad():
        a, ar = reader(**{k: alone[k] for k in ('input_ids', 'token_mask', 'contact_positions')})
        b, br = reader(**{k: padded[k] for k in ('input_ids', 'token_mask', 'contact_positions')})
    torch.testing.assert_close(a[0, 0], b[0, 0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(ar[0], br[0], atol=1e-6, rtol=1e-5)


def test_prefix_physically_excludes_future_tokens_and_completion():
    row = dict(prompt_ids=[2, 3], completion_ids=[4, 5, 6, 4, 7, 8, 1],
        contact_ends=[2, 5], labels=[1., 0.], unique=[True, True], finished=True,
        gt_count=4, identity='protein:0')
    prefix = make_example(row, 1, 20)
    changed = copy.deepcopy(row)
    changed['completion_ids'][3:] = [35, 36, 37, 38]
    assert prefix == make_example(changed, 1, 20)
    assert prefix.input_ids == [2, 3, 4, 5, 6, 20]
    assert prefix.precision == 1 and prefix.recall == 0.25
    assert make_example(row, 2, 20).input_ids[-2] == 1


def test_duplicates_do_not_inflate_precision_or_recall():
    tokens = '<contact> <p0> <p8> <contact> <p8> <p0> <contact> <p1> <p9> <end>'.split()
    parsed = parse_contacts(tokens, 0, 10, [[0, 8], [0, 9]])
    row = dict(parsed, prompt_ids=[2], completion_ids=list(range(len(tokens))), finished=True, gt_count=2, identity='x')
    example = make_example(row, 3, 20)
    assert example.labels == [1., 1., 0.]
    assert example.precision == example.recall == 0.5


def test_single_contact_and_variable_length_loss(reader):
    examples = [Example([2, 3, 4, 5], [2], [1.], [True], .2, 1., 'a'),
        Example([2, 3, 4, 5, 6, 7], [2, 5], [0., 1.], [True, True], .5, .5, 'b')]
    batch = tensors(collate(examples, 0))
    logits, recall = reader(**{k: batch[k] for k in ('input_ids', 'token_mask', 'contact_positions')})
    loss, _ = loss_and_metrics(logits, recall, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert reader.backbone.layers[0].self_attn.q_proj.weight.grad.abs().sum() > 0


def test_prefix_sampler_has_single_contact_and_full_coverage():
    sizes = [prefix_length(100, f'p:{i}', 0) for i in range(5000)]
    assert sizes.count(1) > 200
    assert sizes.count(100) > 1000
    assert min(sizes) == 1 and max(sizes) == 100
    assert prefix_length(1, 'one', 0) == 1
