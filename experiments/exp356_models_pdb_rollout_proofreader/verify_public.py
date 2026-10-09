"""Download a released checkpoint anonymously and exercise its shipped inference CLI."""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from storage import ROOT, upload_directory, write_json


def main() -> None:
    """Verify public bytes and single-contact/full-rollout inference on one GPU."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', default='hf://buckets/open-athena/MarinFold/data/exp356/rollouts-v1')
    parser.add_argument('--out', default=ROOT+'/reports/public-verification')
    args = parser.parse_args()
    fs = HfFileSystem(token=False)
    source = args.checkpoint.removeprefix('hf://')
    manifest = json.loads(fs.cat(source+'/manifest.json'))
    if sum(item['size'] for item in manifest['files'].values()) > 10_000_000_000:
        raise ValueError('Anonymous verification download exceeds the transfer limit')
    local = Path('/tmp/exp356-public-verification')
    model, outputs = local/'model', local/'outputs'
    model.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    for name, item in manifest['files'].items():
        if Path(name).name != name:
            raise ValueError(f'Expected flat checkpoint path: {name}')
        target = model/name
        fs.get_file(source+'/'+name, str(target))
        with target.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if target.stat().st_size != item['size'] or digest != item['sha256']:
            raise ValueError(f'Published checkpoint checksum mismatch: {name}')
    data = args.data.removeprefix('hf://')
    corpus = json.loads(fs.cat(data+'/_SUCCESS.json'))
    example = None
    # Any valid training trajectory is sufficient to verify the public interface;
    # this check does not tune on held-out data or assert quality from one example.
    for item in corpus['files']:
        with fs.open(data+'/'+item['name'], 'rb') as handle:
            rows = pq.read_table(handle).to_pylist()
        example = next((row for row in rows if row['split']=='train' and not row['malformed']
            and len(row['contact_ends'])>=2), None)
        if example is not None:
            break
    if example is None:
        raise ValueError('No suitable public training rollout for interface verification')
    env = dict(os.environ, HF_HUB_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
    env.pop('HF_TOKEN', None)
    predictions = {}
    for label, completion in [('one', example['completion_ids'][:example['contact_ends'][0]+1]),
                              ('full', example['completion_ids'])]:
        supplied = dict(prompt_ids=example['prompt_ids'], completion_ids=completion)
        input_path, output_path = outputs/f'{label}_input.json', outputs/f'{label}_prediction.json'
        input_path.write_text(json.dumps(supplied))
        subprocess.run([sys.executable, str(model/'infer.py'), '--checkpoint', str(model),
            '--rollout', str(input_path), '--device', 'cuda', '--out', str(output_path)],
            cwd=model, env=env, check=True)
        result = json.loads(output_path.read_text())
        expected = 1 if label=='one' else len(example['contact_ends'])
        values = [result['estimated_precision'], result['estimated_recall']]
        values.extend(contact['probability_correct'] for contact in result['contacts'])
        if len(result['contacts']) != expected or any(not math.isfinite(v) or not 0<=v<=1 for v in values):
            raise ValueError(f'Invalid published inference output for {label}')
        predictions[label] = {key:value for key,value in result.items() if key!='contacts'}
    report = dict(checkpoint=args.checkpoint, anonymous_download=True, all_file_hashes_verified=True,
        inference_code='published infer.py', offline_inference=True, example_identity=example['identity'],
        job_id=os.environ.get('EXP356_JOB_ID'), predictions=predictions,
        verified_at=datetime.now(UTC).isoformat())
    (outputs/'verification.json').write_text(json.dumps(report, indent=2))
    files = upload_directory(outputs, args.out)
    write_json(dict(report, files=files), args.out+'/_SUCCESS.json')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
