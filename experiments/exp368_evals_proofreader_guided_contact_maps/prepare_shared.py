"""Extract only the generator's LM head for a bounded shared-node transfer.

The completed H100 smoke verified every other generator tensor equals the
published reader backbone (excluding its extra assessment embedding). This small
bundle permits exact local reconstruction without transferring the backbone twice.
"""

import hashlib
import json

import numpy as np
import torch
from safetensors.torch import save_file

from runtime import GENERATOR, HERE, ROOT, load_release


def main() -> None:
    """Range-read the LM head, pin its bytes and publish the small shared bundle."""
    api = load_release(('storage',))
    fs, root = api.storage.filesystem(GENERATOR)
    index = json.loads(fs.cat(root + '/model.safetensors.index.json'))
    filename = index['weight_map']['lm_head.weight']
    size = int.from_bytes(fs.cat_file(root + '/' + filename, start=0, end=8), 'little')
    header = json.loads(fs.cat_file(root + '/' + filename, start=8, end=8 + size))
    spec = header['lm_head.weight']
    if spec['dtype'] != 'F32' or spec['shape'] != [2845, 2048]:
        raise ValueError('Unexpected frozen generator head layout')
    start, end = spec['data_offsets']
    raw = fs.cat_file(root + '/' + filename, start=8 + size + start, end=8 + size + end)
    if len(raw) != end - start:
        raise ValueError('Incomplete ranged LM-head read')
    local = HERE / '_cache/shared-generator'
    local.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(np.frombuffer(raw, dtype='<f4').copy().reshape(spec['shape']))
    save_file({'lm_head.weight': tensor}, local / 'lm_head.safetensors')
    for name in ['config.json', 'tokenizer.json', 'tokenizer_config.json']:
        (local / name).write_bytes(fs.cat(root + '/' + name))
    provenance = dict(generator=GENERATOR, source_file=filename, tensor=spec,
        tensor_sha256=hashlib.sha256(raw).hexdigest(),
        backbone_equivalence_job='/bizon/exp368-smoke-fp32-v2',
        backbone_equivalence='All generator backbone tensors compared exactly with the released reader before GPU inference')
    (local / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    manifest = api.storage.upload_directory(local, ROOT + '/shared-generator-v1')
    (HERE / 'shared_generator_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(files=len(manifest), bytes=sum(row['size'] for row in manifest.values()))))


if __name__ == '__main__':
    main()
