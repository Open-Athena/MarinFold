"""Save paired individual rollouts from one regional checkpoint on one H100."""

import argparse
import gzip
import hashlib
import json
import os
import platform
import socket
import time
from datetime import UTC, datetime
from pathlib import Path

import fsspec
import torch
from marinfold.document_structures.contacts_v1 import GenerationConfig, build_document, residues_from_sequence
from marinfold.inference._tokenizer import model_source_path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from contacts import parse_contacts, sample_seed

HERE = Path(__file__).resolve().parent


def publish(payload: dict, uri: str, *, compressed: bool = False) -> None:
    """Write a complete artifact; the marker is published after the raw data."""
    data = json.dumps(payload, separators=(',', ':')).encode()
    if compressed:
        data = gzip.compress(data, compresslevel=6, mtime=0)
    with fsspec.open(uri, 'wb') as stream:
        stream.write(data)


def stage(spec: dict, destination: Path) -> list[dict]:
    """Read only regional S3 objects and verify their pinned identity."""
    destination.mkdir(parents=True, exist_ok=True)
    fs, root = fsspec.core.url_to_fs(spec['uri'])
    verified = []
    for entry in spec['files']:
        remote = root + '/' + entry['name']
        info = fs.info(remote)
        if info['size'] != entry['size']:
            raise ValueError(f'Checkpoint size mismatch: {remote}')
        if 'etag' in entry and info['ETag'].strip('"') != entry['etag']:
            raise ValueError(f'Checkpoint ETag mismatch: {remote}')
        local = destination / entry['name']
        fs.get_file(remote, str(local))
        with local.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if 'sha256' in entry and digest != entry['sha256']:
            raise ValueError(f'Checkpoint SHA256 mismatch: {remote}')
        verified.append({**entry, 'sha256': digest})
    return verified


def generate(model: LLM, tokenizer: AutoTokenizer, prompts: list[str], params: list[SamplingParams]) -> list[dict]:
    """Continue length-capped samples while preserving every original token."""
    outputs = model.generate(prompts, params, use_tqdm=False)
    records = []
    for output, parameter in zip(outputs, params, strict=True):
        completion = output.outputs[0]
        records.append({'prompt_token_ids': list(output.prompt_token_ids),
                        'token_ids': list(completion.token_ids), 'text': completion.text,
                        'finish_reason': completion.finish_reason, 'seed': parameter.seed,
                        'initial_tokens': len(completion.token_ids), 'initial_finish_reason': completion.finish_reason,
                        'initial_budget': parameter.max_tokens, 'continuation_seed': None})
    extensions, extension_params, indices = [], [], []
    for index, (record, parameter) in enumerate(zip(records, params, strict=True)):
        if record['finish_reason'] == 'stop':
            continue
        if record['finish_reason'] != 'length':
            raise ValueError(f"Unexpected completion status: {record['finish_reason']}")
        prefix = record['prompt_token_ids'] + record['token_ids']
        remaining = 8192-len(prefix)
        if remaining <= 0:
            continue
        prompt_digest = hashlib.sha256(json.dumps(record['prompt_token_ids'], separators=(',', ':')).encode()).hexdigest()
        prefix_digest = hashlib.sha256(json.dumps(record['token_ids'], separators=(',', ':')).encode()).hexdigest()
        identity = f'exp364-continuation-v1:{parameter.seed}:{prompt_digest}:{prefix_digest}'
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big') % (2**63-1)
        continuation = parameter.clone()
        continuation.max_tokens, continuation.seed = remaining, seed
        record['continuation_seed'] = seed
        extensions.append({'prompt_token_ids': prefix})
        extension_params.append(continuation)
        indices.append(index)
    if indices:
        suffixes = model.generate(extensions, extension_params, use_tqdm=False)
        for index, suffix, prompt in zip(indices, suffixes, extensions, strict=True):
            assert list(suffix.prompt_token_ids) == prompt['prompt_token_ids']
            record = records[index]
            record['token_ids'] += list(suffix.outputs[0].token_ids)
            record['text'] = tokenizer.decode(record['token_ids'], skip_special_tokens=False)
            record['finish_reason'] = suffix.outputs[0].finish_reason
    return records


def main() -> None:
    """Generate resumable per-protein files, retaining all malformed/capped samples."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['mpnn', 'default'], required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=6)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    plan = json.loads((HERE / 'data/plan.json').read_text())
    spec = plan['models'][args.model]
    targets = json.loads((HERE/'inputs/targets.json').read_text())
    targets.sort(key=lambda r: (r['L'], r['stem']))
    if args.smoke:
        targets = [min((r for r in targets if r['eval_set']=='eval-val'), key=lambda r: abs(r['L']-200))]
    else:
        targets = [r for k,r in enumerate(targets) if k % args.shards == args.shard]
    root = plan['artifact_root'] + ('/smoke' if args.smoke else '/raw') + '/' + args.model
    remote_fs = fsspec.filesystem('hf')
    worker_start = time.monotonic()
    model_dir = Path('/tmp/exp373-model')
    verified = stage(spec, model_dir)
    stage_seconds = time.monotonic()-worker_start
    effective = model_source_path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(effective)
    config = json.loads((Path(effective)/'config.json').read_text())
    assert config['rope_theta']==500000 and config['rope_scaling']['rope_type']=='llama3'
    assert len(tokenizer)==config['vocab_size']==2845
    end_id = tokenizer.convert_tokens_to_ids('<end>')
    start = time.monotonic()
    model = LLM(model=str(effective), dtype='bfloat16', max_model_len=8192,
                gpu_memory_utilization=0.85, enable_prefix_caching=False,
                generation_config='vllm', max_num_seqs=128, seed=373)
    model_load = time.monotonic()-start
    device = torch.cuda.get_device_properties(0)
    hardware = dict(gpu_name=device.name, gpu_total_memory_gb=device.total_memory/2**30,
                    gpu_compute_capability=f'{device.major}.{device.minor}', hostname=socket.gethostname(),
                    platform=platform.platform(), torch_version=torch.__version__, runner_tag='iris',
                    model_nickname=spec['nickname'], checkpoint_step=spec['step'], model_load_seconds=model_load,
                    model_stage_seconds=stage_seconds, job_id=os.environ.get('IRIS_JOB_ID',''))
    publish({'model': spec, 'verified_files': verified, 'hardware': hardware, 'effective_config': config},
            f'{root}/worker-{args.shard:02d}.json')
    for target in targets:
        residues = residues_from_sequence(target['input_seq'])
        for temperature in ([1.0] if args.smoke else plan['temperatures']):
            for mode in (['randomized'] if args.smoke else ['fixed','randomized']):
                key = f"{target['stem']}-t{temperature:.1f}-{mode}"
                marker_uri = f'{root}/{key}.complete.json'
                if remote_fs.exists(marker_uri):
                    print('RESUME', key, flush=True)
                    continue
                total_start = time.monotonic()
                count = 300 if mode=='fixed' else 100
                prompt_ids = [f"{target['stem']}:r{0 if mode=='fixed' else k}" for k in range(count)]
                documents = {identifier: build_document(identifier, residues, [], config=GenerationConfig())
                             for identifier in set(prompt_ids)}
                prompts = [documents[identifier].document.split('<begin_statements>')[0]+'<begin_statements>' for identifier in prompt_ids]
                token_lengths = [len(tokenizer(p, add_special_tokens=False).input_ids) for p in prompts]
                assert len(set(token_lengths))==1
                params = [SamplingParams(temperature=temperature, top_p=0.95, top_k=-1,
                         max_tokens=min(8192-n,6*target['L']+128), stop_token_ids=[end_id],
                         skip_special_tokens=False, seed=sample_seed(target['stem'],mode,k//100,k%100))
                         for k,n in enumerate(token_lengths)]
                inference_start = time.monotonic()
                records = generate(model, tokenizer, prompts, params)
                elapsed = time.monotonic()-inference_start
                for k, record in enumerate(records):
                    document = documents[prompt_ids[k]]
                    record.update(rollout=k, block=k//100, prompt_id=prompt_ids[k],
                                  offset=document.n_term_index,
                                  **parse_contacts(record['text'], document.n_term_index, target['L']))
                timing = {**hardware, 'stem':target['stem'], 'n_residues':target['L'],
                          'n_pairs':max(0,(target['L']-6)*(target['L']-5)//2), 'mode':mode,
                          'temperature':temperature, 'elapsed_seconds':elapsed, 'n_rollouts':count,
                          'timestamp_utc':datetime.now(UTC).isoformat()}
                payload = {'target':target, 'model':args.model, 'temperature':temperature, 'mode':mode,
                           'prompts':{identifier: documents[identifier].document.split('<begin_statements>')[0]+'<begin_statements>' for identifier in documents},
                           'timing':timing, 'samples':records}
                raw_uri = f'{root}/{key}.json.gz'
                publish(payload, raw_uri, compressed=True)
                timing['total_seconds'] = time.monotonic()-total_start+stage_seconds+model_load
                unfinished = sum(r['finish_reason']!='stop' for r in records)
                marker = {'raw_uri':raw_uri, 'count':count, 'unfinished':unfinished,
                          'extended':sum(r['continuation_seed'] is not None for r in records), 'timing':timing}
                publish(marker, marker_uri)
                print('COMPLETE', args.model, key, count, 'seconds', round(elapsed,2), 'unfinished', unfinished, flush=True)
                if args.smoke and (unfinished or not any(r['contacts'] for r in records)):
                    raise ValueError('Smoke must terminate and yield valid contacts')


if __name__ == '__main__':
    main()
