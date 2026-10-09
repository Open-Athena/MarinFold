"""Generate ordered exp277 rollout supervision as resumable CoreWeave shards."""

import argparse
import hashlib
import json
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from records import CONTEXT, make_prompt, parse_contacts, stable_seed
from storage import GENERATOR, ROOT, filesystem, read_rows, stage_directory, write_csv, write_json, write_rows


def main() -> None:
    """Generate real model errors, persisting every batch with its timing ledger."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', default=ROOT + '/data/v1/targets.parquet')
    parser.add_argument('--out', default=ROOT + '/data/v1/rollouts')
    parser.add_argument('--model', default=GENERATOR)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=32)
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--batch-proteins', type=int, default=8)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    targets = sorted(read_rows(args.targets), key=lambda r: (r['L'], r['entry_id']))[args.shard::args.shards]
    if args.limit:
        targets = targets[:args.limit]
    fs, output_root = filesystem(args.out)
    plan = dict(model=args.model,targets=args.targets,shard=args.shard,shards=args.shards,
        rollouts=args.rollouts,batch_proteins=args.batch_proteins,limit=args.limit,
        target_fingerprint=hashlib.sha256(json.dumps(targets,sort_keys=True).encode()).hexdigest(),
        sampling=dict(temperature=1.0,top_p=.95,top_k=-1,budget='min(6L+128,8192-prompt_tokens-1)'))
    plan_key = output_root+f'/shard-{args.shard:03d}.plan.json'
    if fs.exists(plan_key):
        if json.loads(fs.cat(plan_key)) != plan:
            raise ValueError('Generation parameters or target shard changed on resume')
    else:
        if fs.glob(output_root+f'/shard-{args.shard:03d}-batch-*.done.json'):
            raise ValueError('Existing batches lack a pinned generation plan')
        write_json(plan,args.out+f'/shard-{args.shard:03d}.plan.json')
    code = json.loads(Path('code_manifest.json').read_text())
    local_model = Path('/tmp/exp356-generator')
    started = time.perf_counter()
    stage_directory(args.model, local_model)
    tokenizer = AutoTokenizer.from_pretrained(local_model)
    end_id = tokenizer.convert_tokens_to_ids('<end>')
    llm = LLM(model=str(local_model), dtype='bfloat16', max_model_len=CONTEXT,
        gpu_memory_utilization=0.88, enable_prefix_caching=False,
        generation_config='vllm', max_num_seqs=128, seed=356)
    load_seconds = time.perf_counter() - started
    gpu = torch.cuda.get_device_properties(0)
    worker = dict(model_nickname='contacts-v1-exp277-m2-p06-full-epoch-1.5B-step-266344',
        runner_tag='iris', gpu_name=gpu.name, gpu_total_memory_gb=gpu.total_memory/1e9,
        gpu_compute_capability=f'{gpu.major}.{gpu.minor}', hostname=socket.gethostname(),
        platform=platform.platform(), torch_version=torch.__version__)
    print(f'[exp356] shard={args.shard}/{args.shards} proteins={len(targets)} load={load_seconds:.1f}s', flush=True)
    for start in range(0, len(targets), args.batch_proteins):
        batch = targets[start:start+args.batch_proteins]
        key = f'shard-{args.shard:03d}-batch-{start//args.batch_proteins:05d}'
        if fs.exists(f'{output_root}/{key}.done.json'):
            continue
        batch_started = time.perf_counter()
        work, prompts, settings = [], [], []
        for target in batch:
            for rollout in range(args.rollouts):
                identity = f"{target['entry_id']}:r{rollout}"
                prompt, offset = make_prompt(target['sequence'], 'exp356:' + identity)
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                budget = min(6 * target['L'] + 128, CONTEXT - len(prompt_ids) - 1)
                if budget < 1:
                    raise ValueError(f'No rollout context available: {identity}')
                work.append((target, identity, offset, prompt_ids))
                prompts.append(prompt)
                settings.append(SamplingParams(temperature=1.0, top_p=0.95, top_k=-1,
                    max_tokens=budget, stop_token_ids=[end_id], skip_special_tokens=False,
                    seed=stable_seed('sample:' + identity)))
        # vLLM 0.9.2's V1 offline API returns metrics=None. Measure one protein's
        # rollout batch directly so timings remain attributable to that protein.
        # Durable files still group several proteins to amortize S3 writes.
        results = []
        per_target = {}
        inference_started = time.perf_counter()
        for index, target in enumerate(batch):
            first = index * args.rollouts
            last = first + args.rollouts
            protein_started = time.perf_counter()
            results.extend(llm.generate(prompts[first:last], settings[first:last], use_tqdm=False))
            per_target[target['entry_id']] = time.perf_counter() - protein_started
        batch_seconds = time.perf_counter() - inference_started
        records, timing = [], []
        for (target, identity, offset, prompt_ids), result in zip(work, results, strict=True):
            output = result.outputs[0]
            completion_ids = list(output.token_ids)
            tokens = tokenizer.convert_ids_to_tokens(completion_ids)
            parsed = parse_contacts(tokens, offset, target['L'], target['contacts'])
            finished = output.finish_reason == 'stop' and bool(completion_ids) and completion_ids[-1] == end_id
            records.append(dict(identity=identity, entry_id=target['entry_id'], split=target['split'],
                group_id=target['group_id'], L=target['L'], gt_count=len(target['contacts']),
                prompt_ids=prompt_ids, completion_ids=completion_ids, finished=finished,
                finish_reason=output.finish_reason, **parsed))
        total_seconds = time.perf_counter() - batch_started
        for target in batch:
            timing.append(dict(stem=target['entry_id'], n_residues=target['L'], n_pairs=len(target['contacts']),
                mode='rollout-supervision', elapsed_seconds=per_target[target['entry_id']],
                model_load_seconds=load_seconds, total_seconds=total_seconds+load_seconds,
                batch_elapsed_seconds=batch_seconds, timing_scope='direct per-protein rollout-batch wall time',
                total_scope='shared batch setup+inference+serialization, plus full worker load',
                n_rollouts=args.rollouts, batch_size=args.rollouts, batch_id=key,
                timestamp_utc=datetime.now(UTC).isoformat(), **worker))
        write_rows(records, args.out + '/' + key + '.parquet')
        write_rows(timing, args.out + '/' + key + '.timings.parquet')
        write_csv(timing, args.out + '/' + key + '.timings.csv')
        summary = dict(proteins=len(batch), rollouts=len(records), finished=sum(r['finished'] for r in records),
            malformed=sum(r['malformed'] for r in records), seconds=batch_seconds,
            zero_contact=sum(not r['contact_ends'] for r in records),code_sha256=code['sha256'])
        write_json(summary, args.out + '/' + key + '.done.json')
        print(f'[exp356] {key} {json.dumps(summary)}', flush=True)
    write_json(code,args.out+f'/shard-{args.shard:03d}.code.json')
    write_json(dict(shard=args.shard, shards=args.shards, proteins=len(targets), rollouts=args.rollouts),
               args.out + f'/shard-{args.shard:03d}.complete.json')


if __name__ == '__main__':
    main()
