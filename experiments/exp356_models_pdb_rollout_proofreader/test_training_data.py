"""Tests for dataset coverage and repeatable distributed recovery."""

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from model import ASSESSMENT, Proofreader, load_model
from training_data import epoch_order


def test_epoch_order_covers_every_rollout_and_is_reproducible():
    order = epoch_order(103, 2, 8, 4)
    assert order.shape == (4, 4, 8)
    assert set(order.ravel()) == set(range(103))
    np.testing.assert_array_equal(order[2:], epoch_order(103, 2, 8, 4)[2:])
    assert not np.array_equal(order, epoch_order(103, 3, 8, 4))
    assert len(order.ravel())-103 < 32


def test_checkpoint_roundtrip_keeps_heads_attention_and_tokenizer(tmp_path):
    torch.manual_seed(9)
    config = Qwen3Config(vocab_size=8, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    config._attn_implementation='sdpa'
    model=Proofreader(Qwen3Model(config)).eval()
    base=Tokenizer(WordLevel({token:i for i,token in enumerate(['<pad>','<unk>','a','b','c','d','e',ASSESSMENT])},unk_token='<unk>'))
    base.pre_tokenizer=WhitespaceSplit()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=base,pad_token='<pad>',unk_token='<unk>')
    inputs=torch.tensor([[2,3,4,5,6,7]])
    mask=torch.ones_like(inputs,dtype=torch.bool)
    positions=torch.tensor([[3]])
    with torch.no_grad():
        expected=model(inputs,mask,positions)
    model.save(tmp_path,tokenizer,{'attention':'bidirectional'})
    restored,restored_tokenizer=load_model(tmp_path)
    restored.eval()
    with torch.no_grad():
        actual=restored(inputs,mask,positions)
    torch.testing.assert_close(actual[0],expected[0])
    torch.testing.assert_close(actual[1],expected[1])
    assert restored_tokenizer.convert_tokens_to_ids(ASSESSMENT)==7
