"""Proofread a sequence and ordered contact list with a published exp356 model."""

import argparse
import contextlib
import csv
import json
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from model import ASSESSMENT, load_model
from records import CONTEXT, make_prompt
from storage import stage_directory


def score_contacts(model, tokenizer, sequence: str, contacts: list[list[int]], *,
                   finished: bool = False, device: torch.device) -> dict:
    """Predict every contact's correctness using all supplied sequence and contacts.

    Args:
        model: Loaded Proofreader in evaluation mode.
        tokenizer: Its co-located tokenizer.
        sequence: Complete amino-acid sequence in one-letter notation.
        contacts: Ordered, zero-based residue pairs from a rollout or prefix.
        finished: Whether the generator actually terminated after these contacts.
        device: Device holding the model.

    Returns:
        Per-contact probabilities and estimated unique-contact precision and recall.
    """
    if not contacts:
        raise ValueError('Supply at least one contact')
    prompt, offset = make_prompt(sequence, 'exp356:inference')
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    readouts, canonical = [], []
    for pair in contacts:
        if len(pair) != 2 or not all(isinstance(i, int) for i in pair):
            raise ValueError(f'Expected two integer residue indices: {pair}')
        a,b=pair
        if not 0 <= a < len(sequence) or not 0 <= b < len(sequence):
            raise ValueError(f'Contact outside sequence: {pair}')
        token_ids.extend(tokenizer.encode(f'<contact> <p{(a+offset)%2000}> <p{(b+offset)%2000}>',add_special_tokens=False))
        readouts.append(len(token_ids)-1)
        canonical.append(tuple(sorted(pair)))
    if finished:
        token_ids.append(tokenizer.convert_tokens_to_ids('<end>'))
    token_ids.append(tokenizer.convert_tokens_to_ids(ASSESSMENT))
    if len(token_ids)>CONTEXT:
        raise ValueError(f'Input has {len(token_ids)} tokens, exceeding {CONTEXT}; supply a shorter prefix')
    ids=torch.tensor([token_ids],device=device)
    positions=torch.tensor([readouts],device=device)
    autocast=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
    with torch.inference_mode(),autocast:
        logits,recall=model(ids,torch.ones_like(ids,dtype=torch.bool),positions)
    probabilities=logits[0].sigmoid().cpu().tolist()
    unique={}
    for pair,q in zip(canonical,probabilities,strict=True):
        unique.setdefault(pair,q)
    return dict(contacts=[dict(pair=p,probability_correct=q) for p,q in zip(contacts,probabilities,strict=True)],
        estimated_precision=sum(unique.values())/len(unique),estimated_recall=float(recall[0]),
        n_unique_contacts=len(unique),finished=finished)


def main() -> None:
    """Write predictions and contemporaneous timing metadata for one input."""
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True,help='Local directory or S3/HF checkpoint URI')
    parser.add_argument('--sequence',required=True)
    parser.add_argument('--contacts',type=Path,required=True,help='JSON array of zero-based contact pairs')
    parser.add_argument('--finished',action='store_true')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    started=time.perf_counter()
    checkpoint=Path(args.checkpoint)
    if '://' in args.checkpoint:
        checkpoint=Path('/tmp/exp356-infer-checkpoint')
        stage_directory(args.checkpoint,checkpoint)
    device=torch.device(args.device)
    model,tokenizer=load_model(checkpoint)
    model.to(device).eval()
    loaded=time.perf_counter()-started
    contacts=json.loads(args.contacts.read_text())
    if device.type=='cuda':torch.cuda.synchronize()
    infer=time.perf_counter()
    result=score_contacts(model,tokenizer,args.sequence,contacts,finished=args.finished,device=device)
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-infer
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2))
    gpu=torch.cuda.get_device_properties(device) if device.type=='cuda' else None
    timing=dict(stem=args.contacts.stem,n_residues=len(args.sequence),n_pairs=len(contacts),mode='proofreader',
        elapsed_seconds=elapsed,model_load_seconds=loaded,total_seconds=time.perf_counter()-started,
        model_nickname=args.checkpoint,runner_tag='local',gpu_name=gpu.name if gpu else '',
        gpu_total_memory_gb=gpu.total_memory/1e9 if gpu else 0,
        gpu_compute_capability=f'{gpu.major}.{gpu.minor}' if gpu else '',hostname=socket.gethostname(),
        platform=platform.platform(),torch_version=torch.__version__,timestamp_utc=datetime.now(UTC).isoformat())
    with args.out.with_suffix('.timings.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(timing));writer.writeheader();writer.writerow(timing)
    print(json.dumps({k:v for k,v in result.items() if k!='contacts'},indent=2))


if __name__=='__main__':
    main()
