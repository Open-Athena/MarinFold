"""Stage verified model inputs from the selected durable regional mirror."""

import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from runtime import HERE, digest_file

def stage_regional(api, region: str = 'RNO2A') -> dict[str, str]:
    """Rebuild the exact original generator using only regional weight reads."""
    input_manifest = json.loads((HERE / 'regional_inputs_manifest.json').read_text())
    regional = input_manifest['mirrors'][region]
    fs, root = api.storage.filesystem(regional)
    location = fs.call_s3('head_bucket', Bucket=root.split('/')[0])
    if location['ResponseMetadata']['HTTPHeaders']['x-amz-bucket-region'] != region:
        raise ValueError('Regional inputs are not in the GPU region')
    complete = json.loads(fs.cat(root + '/complete.json'))
    if len(complete['completed']) != len(complete['inputs']) or complete['reserved_bytes'] > 7_000_000_000:
        raise ValueError('Input mirror is incomplete or exceeds its transfer reservation')
    reader, generator, head = [Path('/tmp/exp368-' + name) for name in ['reader', 'generator', 'head']]
    api.storage.stage_directory(regional + '/reader', reader)
    api.storage.stage_directory(regional + '/head', head)
    release = json.loads((HERE / 'release_manifest.json').read_text())
    for name in ['model.safetensors', 'heads.pt', 'config.json', 'proofreader.json',
                 'tokenizer.json', 'tokenizer_config.json']:
        if digest_file(reader / name) != release['files'][name]['sha256']:
            raise ValueError('Regional reader hash mismatch: ' + name)
    head_manifest = json.loads((HERE / 'shared_generator_manifest.json').read_text())
    for name, file_manifest in head_manifest.items():
        if digest_file(head / name) != file_manifest['sha256']:
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
    if digest_file(targets) != input_manifest['targets_sha256']:
        raise ValueError('Regional target hash mismatch')
    if digest_file(generator / 'model.safetensors') != input_manifest['generator_weights_sha256']:
        raise ValueError('Reconstructed regional generator differs from the verified pilot generator')
    print('[exp368-regional] all input hashes verified against the original pilot', flush=True)
    return dict(os.environ, EXP368_SHARED_MODELS='1', EXP368_TARGETS=str(targets))
