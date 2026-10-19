"""Check that causal features stay fixed while proofreading uses later evidence."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

from model import CausalContactProofreader, loss_and_metrics
from records import Example, collate


@pytest.fixture
def reader() -> CausalContactProofreader:
    torch.manual_seed(356)
    torch.set_num_threads(2)
    config = Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
        attention_dropout=0.,max_position_embeddings=128)
    config._attn_implementation='sdpa'
    return CausalContactProofreader(Qwen3Model(config),2,32,2,4).eval()


def test_causal_features_stay_fixed_but_later_contact_changes_judgment(reader):
    inputs = torch.tensor([[2,3,4,5,6,7,8,9,10]])
    changed = inputs.clone()
    changed[0,6]=25
    mask = torch.ones_like(inputs,dtype=torch.bool)
    positions = torch.tensor([[4,7]])
    with torch.no_grad():
        a = reader.backbone(inputs,attention_mask=mask,use_cache=False).last_hidden_state
        b = reader.backbone(changed,attention_mask=mask,use_cache=False).last_hidden_state
        original,_ = reader(inputs,mask,positions)
        revised,_ = reader(changed,mask,positions)
    torch.testing.assert_close(a[:,:5],b[:,:5],atol=1e-7,rtol=1e-6)
    assert abs(float(original[0,0]-revised[0,0]))>1e-5


def test_padding_does_not_change_single_contact_prediction(reader):
    one = Example([2,3,4,5,6],[3],[1.],[True],.2,1.,'a')
    two = Example([2,3,4,5,6,7,8,9,10],[3,7],[0.,1.],[True,True],.5,.5,'b')
    results=[]
    for examples in [[one],[one,two]]:
        batch={k:torch.from_numpy(v) for k,v in collate(examples,0).items()}
        with torch.no_grad():
            results.append(reader(**{k:batch[k] for k in ['input_ids','token_mask','contact_positions']}))
    torch.testing.assert_close(results[0][0][0,0],results[1][0][0,0],atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(results[0][1][0],results[1][1][0],atol=1e-6,rtol=1e-5)


def test_training_updates_contact_encoder_without_unfreezing_backbone(reader):
    reader.train()
    row=Example([2,3,4,5,6],[3],[1.],[True],.2,1.,'a')
    batch={k:torch.from_numpy(v) for k,v in collate([row],0).items()}
    logits,recall=reader(**{k:batch[k] for k in ['input_ids','token_mask','contact_positions']})
    loss,_=loss_and_metrics(logits,recall,batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert reader.contact_head.triple_projection.weight.grad.abs().sum()>0
    assert not reader.backbone.training
    assert all(not p.requires_grad and p.grad is None for p in reader.backbone.parameters())
