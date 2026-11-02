"""Proofread a sequence and ordered contact list with a published exp356 model."""

import argparse
import contextlib
import csv
import json
import os
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from model import ASSESSMENT, load_model
from records import AA, CONTEXT, make_prompt, parse_contacts
from storage import stage_directory


def score_rollout(model, tokenizer, prompt_ids: list[int], completion_ids: list[int], *,
                  device: torch.device) -> dict:
    """Score an actual tokenized rollout or prefix without labels or reserialization.

    Both arrays must use the checkpoint's tokenizer. The completion must end at a
    complete contact or a genuine end token; later tokens must be absent when
    assessing a prefix. Returns residue indices in the original zero-based sequence.
    """
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_ids)
    if not prompt_tokens or prompt_tokens[-1] != '<begin_statements>':
        raise ValueError('Prompt must end at <begin_statements>')
    if prompt_tokens[0] != '<contacts-v1>' or prompt_tokens.count('<n-term>') != 1 or prompt_tokens.count('<c-term>') != 1:
        raise ValueError('This checkpoint expects one protein chain in a contacts-v1 document')
    amino_tokens = {'<'+name+'>' for name in AA.values()}
    length = sum(token in amino_tokens for token in prompt_tokens)
    n_term = prompt_tokens.index('<n-term>')
    offset = int(prompt_tokens[n_term+1][2:-1])
    completion = tokenizer.convert_ids_to_tokens(completion_ids)
    parsed = parse_contacts(completion, offset, length, [])
    if parsed['malformed'] or not parsed['contact_ends']:
        raise ValueError('Supply a well-formed rollout with at least one complete contact')
    if '<end>' in completion[:-1]:
        raise ValueError('Tokens after the end marker are not a single rollout')
    token_ids = prompt_ids + completion_ids + [tokenizer.convert_tokens_to_ids(ASSESSMENT)]
    if len(token_ids)>CONTEXT:
        raise ValueError(f'Input has {len(token_ids)} tokens, exceeding {CONTEXT}; supply a shorter prefix')
    ids=torch.tensor([token_ids],device=device)
    positions=torch.tensor([[len(prompt_ids)+p for p in parsed['contact_ends']]],device=device)
    autocast=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
    with torch.inference_mode(),autocast:
        logits,recall=model(ids,torch.ones_like(ids,dtype=torch.bool),positions)
    probabilities=logits[0].sigmoid().cpu().tolist()
    unique=[q for q,use in zip(probabilities,parsed['unique'],strict=True) if use]
    return dict(contacts=[dict(pair=p,probability_correct=q) for p,q in zip(parsed['pairs'],probabilities,strict=True)],
        estimated_precision=sum(unique)/len(unique),estimated_recall=float(recall[0]),
        n_unique_contacts=len(unique),n_residues=length,finished=completion[-1]=='<end>')


def score_contacts(model, tokenizer, sequence: str, contacts: list[list[int]], *,
                   finished: bool = False, device: torch.device) -> dict:
    """Score ordered zero-based pairs using a freshly serialized sequence prompt."""
    prompt, offset = make_prompt(sequence, 'exp356:inference')
    statements = []
    for pair in contacts:
        if len(pair) != 2 or not all(isinstance(i, int) for i in pair):
            raise ValueError(f'Expected two integer residue indices: {pair}')
        a, b = pair
        if not 0 <= a < len(sequence) or not 0 <= b < len(sequence):
            raise ValueError(f'Contact outside sequence: {pair}')
        statements.append(f'<contact> <p{(a+offset)%2000}> <p{(b+offset)%2000}>')
    if finished:
        statements.append('<end>')
    return score_rollout(model, tokenizer, tokenizer.encode(prompt,add_special_tokens=False),
        tokenizer.encode(' '.join(statements),add_special_tokens=False),device=device)


def main() -> None:
    """Write predictions and contemporaneous timing metadata for one input."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True,help='Local directory or S3/HF checkpoint URI')
    parser.add_argument('--sequence')
    inputs=parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--contacts',type=Path,help='JSON array of zero-based contact pairs; requires --sequence')
    inputs.add_argument('--rollout',type=Path,help='JSON object with prompt_ids and completion_ids from an actual rollout')
    parser.add_argument('--finished',action='store_true')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.contacts and not args.sequence:
        parser.error('--contacts requires --sequence')
    started=time.perf_counter()
    checkpoint=Path(args.checkpoint)
    if '://' in args.checkpoint:
        checkpoint=Path('/tmp/exp356-infer-checkpoint')
        stage_directory(args.checkpoint,checkpoint,include_training_state=False)
    device=torch.device(args.device)
    model,tokenizer=load_model(checkpoint)
    model.to(device).eval()
    loaded=time.perf_counter()-started
    source=args.contacts or args.rollout
    supplied=json.loads(source.read_text())
    if device.type=='cuda':torch.cuda.synchronize()
    infer=time.perf_counter()
    if args.rollout:
        result=score_rollout(model,tokenizer,supplied['prompt_ids'],supplied['completion_ids'],device=device)
    else:
        result=score_contacts(model,tokenizer,args.sequence,supplied,finished=args.finished,device=device)
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-infer
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    gpu=torch.cuda.get_device_properties(device) if device.type=='cuda' else None
    timing=dict(stem=source.stem,n_residues=result['n_residues'],n_pairs=len(result['contacts']),mode='proofreader',
        elapsed_seconds=elapsed,model_load_seconds=loaded,total_seconds=time.perf_counter()-started,
        model_nickname=args.checkpoint,runner_tag='iris' if os.getenv('EXP356_JOB_ID') else 'local',gpu_name=gpu.name if gpu else '',
        gpu_total_memory_gb=gpu.total_memory/1e9 if gpu else 0,
        gpu_compute_capability=f'{gpu.major}.{gpu.minor}' if gpu else '',hostname=socket.gethostname(),
        platform=platform.platform(),torch_version=torch.__version__,timestamp_utc=datetime.now(UTC).isoformat())
    with args.out.with_suffix('.timings.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(timing));writer.writeheader();writer.writerow(timing)
    print(json.dumps({k:v for k,v in result.items() if k!='contacts'},indent=2))


if __name__=='__main__':
    main()
