"""Proofread independent rollout pools with the published full-forward API."""

import argparse
import json
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from runtime import HERE, ROOT, TARGETS, load_release, pin_run, stage_reader


def main() -> None:
    """Batch complete rollouts, keeping full EOS and first-occurrence semantics."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='pilot.json')
    parser.add_argument('--input', default=ROOT + '/pool-v1')
    parser.add_argument('--out', default=ROOT + '/pool-scored-v1')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    api = load_release()
    plan = json.loads((HERE / args.plan).read_text())
    pin = pin_run(api, plan, args.out, args.shard, kind='ordinary-proofreading', input=args.input)
    targets = {row['entry_id']: row for row in api.storage.read_rows(TARGETS)}
    panel = sorted([targets[entry] for entry in plan['entries']], key=lambda row: (row['L'], row['entry_id']))
    if any(row['split'] != plan['split'] for row in panel):
        raise ValueError('Wrong data split')
    panel = panel[args.shard::args.shards]
    directory = Path('/tmp/exp368-reader')
    started = time.perf_counter()
    stage_reader(api, directory)
    reader, tokenizer = api.model.load_model(directory)
    reader = reader.to('cuda').eval()
    load_seconds = time.perf_counter() - started
    assessment = tokenizer.convert_tokens_to_ids('<assess>')
    gpu = torch.cuda.get_device_properties(0)
    worker = dict(model_nickname='exp356-step-18000', runner_tag='iris', gpu_name=gpu.name,
        gpu_total_memory_gb=gpu.total_memory / 1e9, gpu_compute_capability=f'{gpu.major}.{gpu.minor}',
        hostname=socket.gethostname(), platform=platform.platform(), torch_version=torch.__version__)
    fs, root = api.storage.filesystem(args.out)
    source_fs, source_root = api.storage.filesystem(args.input)
    for target in panel:
        for size in plan.get('pool_sizes', [8, 32, 128]):
            key = f"{target['entry_id']}/pool-{size}"
            if fs.exists(root + '/' + key + '.done.json'):
                continue
            if not source_fs.exists(source_root + '/' + key + '.done.json'):
                raise FileNotFoundError('Incomplete source wave: ' + key)
            rows = api.storage.read_rows(args.input + '/' + key + '.parquet')
            start = time.perf_counter()
            examples, eligible_indices = [], []
            for i, row in enumerate(rows):
                if not row['contact_ends']:
                    row.update(probabilities=[], predicted_recall=0.)
                    continue
                example_row = dict(row, identity=f"{row['entry_id']}:r{row['sample']}")
                examples.append(api.records.make_example(example_row, len(row['contact_ends']), assessment))
                eligible_indices.append(i)
            for offset in range(0, len(examples), 8):
                batch = examples[offset:offset + 8]
                arrays = api.records.collate(batch, tokenizer.pad_token_id)
                ids = torch.as_tensor(arrays['input_ids'], device='cuda')
                mask = torch.as_tensor(arrays['token_mask'], device='cuda')
                positions = torch.as_tensor(arrays['contact_positions'], device='cuda')
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits, recall = reader(ids, mask, positions)
                probabilities = logits.sigmoid().tolist()
                recalls = recall.tolist()
                for j, example in enumerate(batch):
                    row = rows[eligible_indices[offset + j]]
                    row['probabilities'] = probabilities[j][:len(example.labels)]
                    row['predicted_recall'] = recalls[j]
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            api.storage.write_rows(rows, args.out + '/' + key + '.parquet')
            timing = dict(stem=target['entry_id'], n_residues=target['L'],
                n_pairs=sum(len(row['pairs']) for row in rows), mode='proofread-ordinary',
                sample_end=size, elapsed_seconds=elapsed, model_load_seconds=load_seconds,
                total_seconds=time.perf_counter() - start + load_seconds,
                timing_scope='one protein, this wave only; add generation wave cost',
                timestamp_utc=datetime.now(UTC).isoformat(), **worker)
            api.storage.write_csv([timing], args.out + '/' + key + '.timings.csv')
            api.storage.write_json(dict(rollouts=len(rows), **pin), args.out + '/' + key + '.done.json')
            print(f'[exp368] scored {key} seconds={elapsed:.2f}', flush=True)
    api.storage.write_json(dict(proteins=len(panel), **pin), args.out + f'/shard-{args.shard:03d}.complete.json')


if __name__ == '__main__':
    main()
