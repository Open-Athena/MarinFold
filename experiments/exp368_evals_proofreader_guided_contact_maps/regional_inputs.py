"""Stage verified model inputs from the durable RNO2A mirror."""

import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from runtime import HERE, digest_file

REGIONAL = 's3://marinfold-exp368-rno2a/MarinFold/exp368/inputs-v1'


def stage_regional(api) -> dict[str, str]:
    """Rebuild the exact original generator using only regional weight reads."""
    fs, root = api.storage.filesystem(REGIONAL)
    location = fs.call_s3('head_bucket', Bucket='marinfold-exp368-rno2a')
    if location['ResponseMetadata']['HTTPHeaders']['x-amz-bucket-region'] != 'RNO2A':
        raise ValueError('Regional inputs are not in the GPU region')
    complete = json.loads(fs.cat(root + '/complete.json'))
    if len(complete['completed']) != len(complete['inputs']) or complete['reserved_bytes'] > 7_000_000_000:
        raise ValueError('Input mirror is incomplete or exceeds its transfer reservation')
    reader, generator, head = [Path('/tmp/exp368-' + name) for name in ['reader', 'generator', 'head']]
    api.storage.stage_directory(REGIONAL + '/reader', reader)
    api.storage.stage_directory(REGIONAL + '/head', head)
    release = json.loads((HERE / 'release_manifest.json').read_text())
    for name in ['model.safetensors', 'heads.pt', 'config.json', 'proofreader.json',
                 'tokenizer.json', 'tokenizer_config.json']:
        if digest_file(reader / name) != release['files'][name]['sha256']:
            raise ValueError('Regional reader hash mismatch: ' + name)
    head_manifest = json.loads((HERE / 'shared_generator_manifest.json').read_text())
    for name, expected in head_manifest.items():
        if digest_file(head / name) != expected['sha256']:
            raise ValueError('Regional generator-head hash mismatch: ' + name)
    generator.mkdir(parents=True, exist_ok=True)
    weights = load_file(head / 'lm_head.safetensors')
    with safe_open(reader / 'model.safetensors', framework='pt', device='cpu') as tensors:
        for name in tensors.keys():
            value = tensors.get_tensor(name)
            if name == 'embed_tokens.weight':
                value = value[:2845].contiguous()
            weights['model.' + name] = value
        save_file(weights, generator / 'model.safetensors', metadata={'format': 'pt'})
    del weights
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        shutil.copyfile(head / name, generator / name)
    targets = Path('/tmp/exp368-targets.parquet')
    fs.get_file(root + '/targets.parquet', str(targets))
    expected = json.loads((HERE / 'regional_inputs_manifest.json').read_text())
    if digest_file(targets) != expected['targets_sha256']:
        raise ValueError('Regional target hash mismatch')
    if digest_file(generator / 'model.safetensors') != expected['generator_weights_sha256']:
        raise ValueError('Reconstructed regional generator differs from the verified pilot generator')
    print('[exp368-regional] all input hashes verified against the original pilot', flush=True)
    return dict(os.environ, EXP368_SHARED_MODELS='1', EXP368_TARGETS=str(targets))
