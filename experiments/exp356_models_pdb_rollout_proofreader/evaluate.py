"""Evaluate calibrated contact judgments at fixed prefix lengths on held-out chains."""

import argparse
import json
import os
import platform
import socket
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score, roc_auc_score

from model import ASSESSMENT, load_model
from records import collate, make_example
from storage import ROOT, stage_directory, write_csv, write_json, write_rows
from training_data import read_table, stage_rollouts


def main() -> None:
    """Score each protein separately; preserve calibration and contextual diagnostics."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default=ROOT+'/data/v1/rollouts')
    parser.add_argument('--out', required=True)
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    parser.add_argument('--limit-proteins', type=int)
    args = parser.parse_args()
    rank, world = int(os.getenv('RANK','0')), int(os.getenv('WORLD_SIZE','1'))
    torch.cuda.set_device(int(os.getenv('LOCAL_RANK','0')))
    device = torch.device('cuda', int(os.getenv('LOCAL_RANK','0')))
    if world > 1:
        dist.init_process_group('nccl', device_id=device)
    torch.set_num_threads(4)
    started = time.perf_counter()
    local = Path('/tmp/exp356-evaluation')
    if rank == 0:
        stage_directory(args.checkpoint, local/'model',include_training_state=False)
        stage_rollouts(args.data, local/'data')
    if world > 1:
        dist.barrier()
    model, tokenizer = load_model(local/'model')
    model.to(device).eval()
    loaded = time.perf_counter()-started
    table = read_table(local/'data', args.split)
    ids = sorted(set(table['entry_id'].to_pylist()))
    if args.limit_proteins:
        ids = ids[:args.limit_proteins]
    mine = set(ids[rank::world])
    gpu = torch.cuda.get_device_properties(device)
    worker = dict(model_nickname=args.checkpoint.split('/checkpoints/')[-1].replace('/','-'),
        runner_tag='iris',gpu_name=gpu.name,gpu_total_memory_gb=gpu.total_memory/1e9,
        gpu_compute_capability=f'{gpu.major}.{gpu.minor}',hostname=socket.gethostname(),
        platform=platform.platform(),torch_version=torch.__version__)
    assess = tokenizer.convert_tokens_to_ids(ASSESSMENT)
    summaries, timings, contact_scores, calibration = [], [], [], defaultdict(lambda:np.zeros(3,dtype=float))
    for batch in table.to_batches(max_chunksize=128):
        for row in batch.to_pylist():
            if row['entry_id'] not in mine:
                continue
            n = len(row['contact_ends'])
            for label, k in [('1',1),('2',2),('4',4),('8',8),('16',16),('32',32),('64',64),('full',n)]:
                if label != 'full' and k > n:
                    continue
                begun = time.perf_counter()
                example = make_example(row,k,assess)
                tensors = {key:torch.from_numpy(value).to(device) for key,value in collate([example],tokenizer.pad_token_id).items()}
                torch.cuda.synchronize()
                infer = time.perf_counter()
                with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                    logits, recall = model(tensors['input_ids'],tensors['token_mask'],tensors['contact_positions'])
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-infer
                q = logits[0].sigmoid().cpu().numpy()
                y = np.asarray(example.labels)
                unique = np.asarray(example.unique)
                q, y = q[unique], y[unique]
                bce = -(y*np.log(np.clip(q,1e-7,1))+(1-y)*np.log(np.clip(1-q,1e-7,1))).mean()
                stats = dict(entry_id=row['entry_id'],identity=row['identity'],group_id=row['group_id'],
                    split=row['split'],L=row['L'],prefix=label,n_contacts=k,
                    precision=example.precision,predicted_precision=float(q.mean()),
                    recall=example.recall,predicted_recall=float(recall[0]),
                    brier=float(((q-y)**2).mean()),bce=float(bce),
                    auroc=float(roc_auc_score(y,q)) if len(set(y))==2 else None,
                    average_precision=float(average_precision_score(y,q)) if y.sum() else None,
                    first_contact_probability=float(q[0]),first_contact_label=float(y[0]))
                for fraction in (.25,.5,.75):
                    count=max(1,round(len(q)*fraction))
                    stats[f'precision_retained_{fraction}']=float(y[np.argsort(-q,kind='stable')[:count]].mean())
                    stats[f'precision_emission_{fraction}']=float(y[:count].mean())
                summaries.append(stats)
                if label == 'full':
                    contact_scores.append(dict(identity=row['identity'], entry_id=row['entry_id'],
                        group_id=row['group_id'], L=row['L'], gt_count=row['gt_count'],
                        pairs=np.asarray(row['pairs'])[unique].tolist(), labels=y.tolist(),
                        probabilities=q.tolist(), predicted_recall=float(recall[0])))
                for probability, truth in zip(q,y,strict=True):
                    bucket=min(9,int(probability*10))
                    calibration[(label,bucket)] += [1,float(probability),float(truth)]
                timings.append(dict(stem=row['entry_id'],identity=row['identity'],n_residues=row['L'],n_pairs=k,
                    mode=f'proofread-prefix-{label}',elapsed_seconds=elapsed,model_load_seconds=loaded,
                    total_seconds=time.perf_counter()-begun+loaded,batch_size=1,
                    timestamp_utc=datetime.now(UTC).isoformat(),**worker))
    write_rows(summaries,args.out+f'/per_rollout-rank-{rank}.parquet')
    write_rows(timings,args.out+f'/timings-rank-{rank}.parquet')
    write_csv(timings,args.out+f'/timings-rank-{rank}.csv')
    write_rows(contact_scores,args.out+f'/contact_scores-rank-{rank}.parquet')
    write_rows([dict(prefix=k[0],bin=k[1],count=v[0],sum_probability=v[1],sum_truth=v[2]) for k,v in calibration.items()],
               args.out+f'/calibration-rank-{rank}.parquet')
    write_json(dict(rank=rank,world=world,proteins=len(mine),rows=len(summaries),checkpoint=args.checkpoint,
                   code=json.loads(Path('code_manifest.json').read_text()),job_id=os.environ['EXP356_JOB_ID']),
               args.out+f'/rank-{rank}.complete.json')
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
