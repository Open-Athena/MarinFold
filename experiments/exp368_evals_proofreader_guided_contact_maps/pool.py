"""Generate nested ordinary rollout pools with measured cumulative GPU cost."""

import argparse
import json
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from runtime import GENERATOR, HERE, ROOT, TARGETS, load_release, pin_run


def main() -> None:
    """Generate 8, then 24, then 96 samples per protein, committing each wave."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='pilot.json')
    parser.add_argument('--out', default=ROOT + '/pool-v1')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=4)
    args = parser.parse_args()
    api = load_release(('records', 'storage'))
    plan = json.loads((HERE / args.plan).read_text())
    pin = pin_run(api, plan, args.out, args.shard, kind='ordinary-generation')
    targets = {row['entry_id']: row for row in api.storage.read_rows(TARGETS)}
    panel = sorted([targets[entry] for entry in plan['entries']], key=lambda row: (row['L'], row['entry_id']))
    if any(row['split'] != plan['split'] for row in panel):
        raise ValueError('Wrong split in panel')
    panel = panel[args.shard::args.shards]
    fs, root = api.storage.filesystem(args.out)
    model = Path('/tmp/exp368-generator')
    started = time.perf_counter()
    api.storage.stage_directory(GENERATOR, model, include_training_state=False)
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=str(model), dtype='bfloat16', max_model_len=8192,
        gpu_memory_utilization=.88, enable_prefix_caching=False,
        generation_config='vllm', max_num_seqs=128, seed=368)
    load_seconds = time.perf_counter() - started
    gpu = torch.cuda.get_device_properties(0)
    worker = dict(model_nickname='contacts-v1-exp277-step-266344', runner_tag='iris',
        gpu_name=gpu.name, gpu_total_memory_gb=gpu.total_memory / 1e9,
        gpu_compute_capability=f'{gpu.major}.{gpu.minor}', hostname=socket.gethostname(),
        platform=platform.platform(), torch_version=torch.__version__)
    end = tokenizer.convert_tokens_to_ids('<end>')
    for target in panel:
        previous = 0
        for size in plan.get('pool_sizes', [8, 32, 128]):
            key = f"{target['entry_id']}/pool-{size}"
            if fs.exists(root + '/' + key + '.done.json'):
                previous = size
                continue
            prompts, settings, work = [], [], []
            for sample in range(previous, size):
                identity = f"exp368:{target['entry_id']}:r{sample}"
                prompt, offset = api.records.make_prompt(target['sequence'], identity)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                prompts.append(prompt)
                settings.append(SamplingParams(temperature=1., top_p=.95, top_k=-1,
                    max_tokens=min(6 * target['L'] + 128, 8192 - len(ids) - 1),
                    stop_token_ids=[end], skip_special_tokens=False,
                    seed=api.records.stable_seed('sample:' + identity)))
                work.append((sample, ids, offset))
            start = time.perf_counter()
            outputs = llm.generate(prompts, settings, use_tqdm=False)
            elapsed = time.perf_counter() - start
            rows = []
            for (sample, ids, offset), result in zip(work, outputs, strict=True):
                output = result.outputs[0]
                completion = list(output.token_ids)
                parsed = api.records.parse_contacts(tokenizer.convert_ids_to_tokens(completion),
                    offset, target['L'], target['contacts'])
                rows.append(dict(entry_id=target['entry_id'], group_id=target['group_id'],
                    split=target['split'], L=target['L'], sample=sample, policy='ordinary',
                    prompt_ids=ids, completion_ids=completion, **parsed,
                    gt_count=len(target['contacts']), finished=bool(completion) and completion[-1] == end))
            api.storage.write_rows(rows, args.out + '/' + key + '.parquet')
            timing = dict(stem=target['entry_id'], n_residues=target['L'],
                n_pairs=sum(len(row['pairs']) for row in rows), mode='vllm-ordinary',
                sample_start=previous, sample_end=size, elapsed_seconds=elapsed,
                model_load_seconds=load_seconds, total_seconds=time.perf_counter() - start + load_seconds,
                timing_scope='one protein, this wave only; sum waves for cumulative pool cost',
                timestamp_utc=datetime.now(UTC).isoformat(), **worker)
            api.storage.write_csv([timing], args.out + '/' + key + '.timings.csv')
            api.storage.write_json(dict(rollouts=len(rows), **pin),
                                   args.out + '/' + key + '.done.json')
            print(f'[exp368] {key} added={size-previous} seconds={elapsed:.2f}', flush=True)
            previous = size
    api.storage.write_json(dict(proteins=len(panel), **pin),
                           args.out + f'/shard-{args.shard:03d}.complete.json')


if __name__ == '__main__':
    main()
