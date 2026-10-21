"""Verify a same-region input mirror before relocating saved scoring pins.

Completed predictions keep their original provenance. Only the input URI in the
new recovery prefix's shard pins changes, after all input object sizes and
content ETags match both the original URI and its regional mirror. Original
pins are archived byte-for-byte and a separate relocation ledger is published.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

from runtime import HERE, load_release


def listing(fs, root: str) -> dict:
    """Read bounded single-PUT object identities for one input snapshot."""
    result = {name.removeprefix(root + '/'): dict(size=value['size'], etag=value.get('ETag', '').strip('"'))
              for name, value in fs.find(root, detail=True).items() if value['type'] == 'file'}
    if not result or sum(item['size'] for item in result.values()) > 1_000_000_000:
        raise ValueError('Expected a nonempty validation input snapshot below 1 GB')
    if any(len(item['etag']) != 32 for item in result.values()):
        raise ValueError('Input identity requires single-PUT content ETags')
    return result


def main() -> None:
    """Copy within RNO storage, verify identity, then update only input locations."""
    migration = json.loads((HERE / 'data/native_validation_scored_migration.json').read_text())
    base = 's3://marinfold-exp368-rno2a/MarinFold/exp368/'
    source = base + 'results/validation-pool'
    destination = base + 'native-results-v1/validation-pool'
    scores = base + 'native-results-v1/validation-pool-scored'
    if migration['destination'] != scores:
        raise ValueError('Scored recovery snapshot is not the expected fresh prefix')
    cache = HERE / '_cache/validation-pool-scored-east-v2'
    paths = sorted(cache.glob('shard-*.pin.json'))
    if len(paths) != 4:
        raise ValueError('Expected four original scoring pins')
    originals = [(path, path.read_bytes(), json.loads(path.read_text())) for path in paths]
    old_inputs = {pin['input'] for _, _, pin in originals}
    if len(old_inputs) != 1:
        raise ValueError('Original scoring shards used different input locations')
    old_input = old_inputs.pop()
    api = load_release(('storage',))
    fs, source_root = api.storage.filesystem(source)
    _, destination_root = api.storage.filesystem(destination)
    _, score_root = api.storage.filesystem(scores)
    location = fs.call_s3('head_bucket', Bucket=source_root.split('/')[0])
    if location['ResponseMetadata']['HTTPHeaders']['x-amz-bucket-region'] != 'RNO2A':
        raise ValueError('Input mirror must remain in the GPU region')
    expected = listing(fs, source_root)
    old_fs, old_root = api.storage.filesystem(old_input)
    if listing(old_fs, old_root) != expected:
        raise ValueError('Regional inputs differ from the original scoring input snapshot')

    def copy(relative: str) -> None:
        target = destination_root + '/' + relative
        if not fs.exists(target):
            fs.cp_file(source_root + '/' + relative, target)

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(copy, sorted(expected)))
    if listing(fs, destination_root) != expected or listing(fs, source_root) != expected:
        raise ValueError('Same-region copy failed identity verification')
    archived = []
    for path, payload, pin in originals:
        digest = hashlib.sha256(payload).hexdigest()
        if digest != migration['files'][path.name]['sha256']:
            raise ValueError('Original scoring pin differs from the verified migration')
        revised = dict(pin, input=destination)
        current = json.loads(fs.cat(score_root + '/' + path.name))
        if current not in [pin, revised]:
            raise ValueError('Scoring pin changed beyond the verified input relocation')
        archive = score_root + '/provenance/input-relocation/' + path.name
        if fs.exists(archive) and fs.cat(archive) != payload:
            raise ValueError('Archived scoring pin conflicts')
        fs.pipe_file(archive, payload, timeout=60)
        if fs.cat(archive) != payload:
            raise ValueError('Original scoring pin archive failed readback')
        api.storage.write_json(revised, scores + '/' + path.name)
        if json.loads(fs.cat(score_root + '/' + path.name)) != revised:
            raise ValueError('Relocated scoring pin failed readback')
        archived.append(dict(name=path.name, original_sha256=digest, original=pin, relocated=revised))
    audit = dict(created_at=datetime.now(UTC).isoformat(), original_input=old_input,
        regional_source=source, input=destination, files=expected, archived_pins=archived,
        changed_fields=['input'], scientific_change='None; every input object has identical content identity.')
    (HERE / 'data/native_validation_input_relocation.json').write_text(json.dumps(audit, indent=2) + '\n')
    api.storage.write_json(audit, scores + '/provenance/input-relocation/audit.json')
    print(json.dumps(dict(files=len(expected), bytes=sum(item['size'] for item in expected.values()),
                         input=destination, pins=len(archived))))


if __name__ == '__main__':
    main()
