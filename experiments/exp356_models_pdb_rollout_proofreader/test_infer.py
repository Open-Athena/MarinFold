"""Verify the public raw-rollout and contact-list interfaces preserve evidence."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

from infer import score_contacts, score_rollout
from model import ASSESSMENT, Proofreader
from records import AA, make_prompt


def test_raw_rollout_matches_contact_list_and_rejects_tokens_after_end():
    tokens = ['<pad>', '<unk>', ASSESSMENT, '<contacts-v1>', '<begin_sequence>',
        '<begin_statements>', '<n-term>', '<c-term>', '<end>', '<contact>']
    tokens += ['<'+value+'>' for value in AA.values()]
    tokens += [f'<p{i}>' for i in range(2000)]
    base = Tokenizer(WordLevel({token:i for i,token in enumerate(tokens)},unk_token='<unk>'))
    base.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base,pad_token='<pad>',unk_token='<unk>')
    config = Qwen3Config(vocab_size=len(tokens),hidden_size=32,intermediate_size=64,
        num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8)
    config._attn_implementation = 'sdpa'
    model = Proofreader(Qwen3Model(config)).eval()
    sequence = 'ACDEFGHIKLMNPQRS'
    contacts = [[0,8],[8,0],[2,13]]
    device = torch.device('cpu')
    result = score_contacts(model,tokenizer,sequence,contacts,finished=True,device=device)
    prompt,offset = make_prompt(sequence,'exp356:inference')
    completion = ' '.join(f'<contact> <p{(a+offset)%2000}> <p{(b+offset)%2000}>' for a,b in contacts)+' <end>'
    prompt_ids = tokenizer.encode(prompt,add_special_tokens=False)
    completion_ids = tokenizer.encode(completion,add_special_tokens=False)
    raw = score_rollout(model,tokenizer,prompt_ids,completion_ids,device=device)
    assert raw == result
    assert raw['n_unique_contacts'] == 2
    assert raw['n_residues'] == len(sequence)
    assert raw['contacts'][0]['pair'] == raw['contacts'][1]['pair'] == [0,8]
    with pytest.raises(ValueError,match='after the end'):
        score_rollout(model,tokenizer,prompt_ids,completion_ids+completion_ids,device=device)
