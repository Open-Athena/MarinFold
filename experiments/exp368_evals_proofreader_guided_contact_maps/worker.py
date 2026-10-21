"""Run resumable validation/test inference policies with per-input timing ledgers."""

import argparse
import gc
import json
import platform
import socket
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from engine import Engine, Policy, select
from runtime import GENERATOR, HERE, ROOT, TARGETS, load_release, pin_run, stage_reader


def load_engine(api) -> Engine:
    """Verify the generator/reader backbone identity before sharing its cache."""
    reader_dir, generator_dir = Path('/tmp/exp368-reader'), Path('/tmp/exp368-generator')
    stage_reader(api, reader_dir)
    api.storage.stage_directory(GENERATOR, generator_dir, include_training_state=False)
    reader, tokenizer = api.model.load_model(reader_dir)
    state = reader.backbone.state_dict()
    seen = set()
    head = None
    for path in sorted(generator_dir.glob('*.safetensors')):
        with safe_open(path, framework='pt', device='cpu') as tensors:
            for name in tensors.keys():
                if name == 'lm_head.weight':
                    head = tensors.get_tensor(name)
                    continue
                if not name.startswith('model.'):
                    raise ValueError(f'Unexpected generator weight: {name}')
                key = name.removeprefix('model.')
                expected, actual = tensors.get_tensor(name), state[key]
                if key == 'embed_tokens.weight':
                    actual = actual[:expected.shape[0]]
                if not torch.equal(expected, actual):
                    raise ValueError(f'Generator and proofreader features differ: {key}')
                seen.add(key)
    if seen != set(state) or head is None:
        raise ValueError('Incomplete generator/backbone verification')
    lm_head = nn.Linear(head.shape[1], head.shape[0], bias=False)
    lm_head.load_state_dict({'weight': head})
    del state, head
    gc.collect()
    print('[exp368] all backbone tensors match the frozen generator', flush=True)
    return Engine(reader.to('cuda'), lm_head.to('cuda'), tokenizer)


def verify_cache(engine: Engine, prompt: list[int], length: int, offset: int) -> dict:
    """Compare batched cached predictions against the published full forward."""
    rng = torch.Generator(device='cuda').manual_seed(368)
    state = select(engine.start(prompt), torch.tensor([0, 0], device='cuda'))
    for _ in range(8):
        state = engine.extend(state, length, offset, rng)
    observed, recall = engine.score(state)
    assess = engine.tokenizer.convert_tokens_to_ids('<assess>')
    ids = torch.cat([state.tokens, state.tokens.new_full((2, 1), assess)], dim=1)
    positions = torch.arange(len(prompt) + 2, state.tokens.shape[1], 3, device='cuda')[None].expand(2, -1)
    with torch.inference_mode(), engine.autocast():
        logits, expected_recall = engine.reader(ids, torch.ones_like(ids, dtype=torch.bool), positions)
    error = (observed - logits.sigmoid()).abs()
    recall_error = (recall - expected_recall).abs()
    result = dict(max_probability_error=error.max().item(), mean_probability_error=error.mean().item(),
                  max_recall_error=recall_error.max().item())
    if result['max_probability_error'] > 1e-5 or result['max_recall_error'] > 1e-5:
        raise ValueError(f'Cached/full release predictions disagree: {result}')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        released_logits, _ = engine.reader(ids, torch.ones_like(ids, dtype=torch.bool), positions)
    result['fp32_vs_released_bf16_max_difference'] = (observed - released_logits.sigmoid()).abs().max().item()
    return result


def released_score(engine: Engine, state) -> tuple[torch.Tensor, torch.Tensor]:
    """Rescore the final map with the release's BF16 full-forward execution."""
    if not state.count:
        return state.hidden.new_empty((1, 0)), state.hidden.new_zeros(1)
    assess = engine.tokenizer.convert_tokens_to_ids('<assess>')
    ids = torch.cat([state.tokens, state.tokens.new_full((1, 1), assess)], dim=1)
    positions = torch.arange(state.prompt_length + 2, state.prompt_length + 3 * state.count,
                             3, device=engine.device)[None]
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        logits, recall = engine.reader(ids, torch.ones_like(ids, dtype=torch.bool), positions)
    return logits.sigmoid(), recall


def main() -> None:
    """Execute a frozen campaign shard; commit each input independently."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', default='pilot.json')
    parser.add_argument('--out', default=ROOT + '/pilot-v1')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    api = load_release()
    plan = json.loads((HERE / args.plan).read_text())
    pin = pin_run(api, plan, args.out, args.shard, kind='guided-search')
    targets = {row['entry_id']: row for row in api.storage.read_rows(TARGETS)}
    panel = sorted([targets[entry] for entry in plan['entries']], key=lambda row: (row['L'], row['entry_id']))
    if any(row['split'] != plan['split'] for row in panel):
        raise ValueError('Panel includes the wrong data split')
    panel = panel[args.shard::args.shards]
    policies = [Policy(**p) for p in plan['policies']]
    fs, root = api.storage.filesystem(args.out)
    started = time.perf_counter()
    engine = load_engine(api)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    gpu = torch.cuda.get_device_properties(0)
    worker = dict(model_nickname='exp277-step-266344+exp356-step-18000', runner_tag='iris',
        gpu_name=gpu.name, gpu_total_memory_gb=gpu.total_memory / 1e9,
        gpu_compute_capability=f'{gpu.major}.{gpu.minor}', hostname=socket.gethostname(),
        platform=platform.platform(), torch_version=torch.__version__)
    print(f'[exp368] loaded in {load_seconds:.1f}s; proteins={len(panel)} policies={len(policies)}', flush=True)
    for target in panel:
        for policy in policies:
            key = target['entry_id'] + '/' + policy.name
            if fs.exists(root + '/' + key + '.done.json'):
                continue
            rows = []
            for sample in range(plan['rollouts']):
                sample_key = key + f'-r{sample}'
                if fs.exists(root + '/' + sample_key + '.done.json'):
                    rows.extend(api.storage.read_rows(args.out + '/' + sample_key + '.parquet'))
                    continue
                start = time.perf_counter()
                identity = f"exp368:{target['entry_id']}:r{sample}"
                prompt_text, offset = api.records.make_prompt(target['sequence'], identity)
                prompt = engine.tokenizer.encode(prompt_text, add_special_tokens=False)
                if args.smoke and sample == 0:
                    verification = verify_cache(engine, prompt, target['L'], offset)
                    print(f'[exp368] cache verification {verification}', flush=True)
                    api.storage.write_json(verification, args.out + '/' + key + '.verification.json')
                engine.generated_tokens = engine.scored_contacts = 0
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                inference_start = time.perf_counter()
                state = engine.rollout(prompt, target['L'], offset, policy,
                    api.records.stable_seed('sample:' + identity), plan.get('contact_limit'))
                q, recall = released_score(engine, state)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - inference_start
                completion = state.tokens[0, len(prompt):].tolist()
                # Reference labels enter only here, after all policy decisions.
                parsed = api.records.parse_contacts(engine.tokenizer.convert_ids_to_tokens(completion),
                    offset, target['L'], target['contacts'])
                if parsed['pairs'] != state.pairs[0].tolist():
                    raise ValueError('Cached pairs differ from the released parser')
                rows.append(dict(entry_id=target['entry_id'], group_id=target['group_id'],
                    split=target['split'], L=target['L'], policy=policy.name, sample=sample,
                    prompt_ids=prompt, completion_ids=completion, finished=state.finished,
                    probabilities=q[0].tolist(), predicted_recall=recall[0].item(),
                    gt_count=len(target['contacts']), **parsed))
                api.storage.write_rows([rows[-1]], args.out + '/' + sample_key + '.parquet')
                timing = dict(stem=target['entry_id'], n_residues=target['L'], n_pairs=state.count,
                    mode=policy.name, sample=sample, elapsed_seconds=elapsed, model_load_seconds=load_seconds,
                    total_seconds=time.perf_counter() - start + load_seconds,
                    timing_scope='sequential per-rollout generation and final proofreading',
                    generated_tokens=engine.generated_tokens, scored_contacts=engine.scored_contacts,
                    peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                    timestamp_utc=datetime.now(UTC).isoformat(), **worker)
                api.storage.write_csv([timing], args.out + '/' + sample_key + '.timings.csv')
                api.storage.write_json(pin, args.out + '/' + sample_key + '.done.json')
                print(f"[exp368] {key} r{sample} contacts={state.count} seconds={elapsed:.2f}", flush=True)
                del state, q
            api.storage.write_rows(rows, args.out + '/' + key + '.parquet')
            api.storage.write_json(dict(proteins=1, rollouts=len(rows), policy=asdict(policy), **pin),
                                   args.out + '/' + key + '.done.json')
    api.storage.write_json(dict(proteins=len(panel), policies=len(policies), **pin),
                           args.out + f'/shard-{args.shard:03d}.complete.json')


if __name__ == '__main__':
    main()
