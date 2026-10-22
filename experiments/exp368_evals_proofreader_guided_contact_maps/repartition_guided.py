"""Repartition a stopped guided campaign without changing trajectory identities.

Only execution partition metadata changes. The exact original source bundle,
plan and per-trajectory seeds remain authoritative. Original shard pins are
archived in the new recovery prefix before fresh workers create their own pins;
all saved trajectory artifacts retain their original provenance unchanged.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime

from runtime import HERE, load_release


def main() -> None:
    """Verify the migrated snapshot, archive old pins and publish new shards."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--shards', type=int, default=48)
    args = parser.parse_args()
    if not 12 <= args.shards <= 64:
        raise ValueError('Recovery is bounded to 12–64 independent GPU shards')
    records = [json.loads(s) for s in (HERE / 'data/server_tasks.jsonl').read_text().splitlines()]
    original = next(row for row in records if row['task_id'] == '20-pilot-guided-000')
    task = original['task']
    if task['kind'] != 'guided' or task['plan'] != 'pilot.json' or task['shards'] != 12:
        raise ValueError('Original frozen guided task differs from the expected campaign')
    migration = json.loads((HERE / 'data/native_guided_migration.json').read_text())
    destination = migration['destination']
    if destination != 's3://marinfold-exp368-rno2a/MarinFold/exp368/native-results-v1/pilot-guided':
        raise ValueError('Repartition is restricted to the new, stopped native recovery prefix')
    cache = HERE / '_cache/pilot-guided-east-v3'
    plan = json.loads((HERE / task['plan']).read_text())
    plan_sha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    done = list(cache.rglob('*.done.json'))
    for path in done:
        marker = json.loads(path.read_text())
        if marker['code_sha256'] != original['source']['sha256'] or marker['plan_sha256'] != plan_sha:
            raise ValueError('Completed trajectory has a different code or plan: ' + str(path))
    api = load_release(('storage',))
    fs, root = api.storage.filesystem(destination)
    pins = []
    for path in sorted(cache.glob('shard-*.pin.json')):
        payload = path.read_bytes()
        pin = json.loads(payload)
        if pin['shards'] != 12 or pin['code_sha256'] != original['source']['sha256'] or pin['plan_sha256'] != plan_sha:
            raise ValueError('Original shard pin does not match the saved trajectory source')
        if hashlib.sha256(payload).hexdigest() != migration['files'][path.name]['sha256']:
            raise ValueError('Original pin is absent from the verified migration ledger')
        key = root + '/' + path.name
        archived = root + f'/provenance/repartition-{args.shards}/' + path.name
        if fs.exists(key) and fs.cat(key) != payload:
            raise ValueError('A worker changed the new recovery pin; stop before repartitioning')
        if fs.exists(archived):
            if fs.cat(archived) != payload:
                raise ValueError('Archived original pin conflicts')
        else:
            fs.pipe_file(archived, payload, timeout=60)
        if fs.cat(archived) != payload:
            raise ValueError('Archived pin failed readback verification')
        pins.append(dict(original=key, archived=archived, sha256=hashlib.sha256(payload).hexdigest()))
    if len(pins) != 12:
        raise ValueError('Expected all twelve original shard pins')
    control = 's3://marinfold-exp368-rno2a/MarinFold/exp368/native-control-v1'
    for pin in pins:
        if fs.exists(pin['original']):
            fs.rm(pin['original'])
    recorded = {row['task_id'] for row in records if row['control'] == control}
    new = []
    for shard in range(args.shards):
        identity = f'70-native-guided-{shard:03}'
        replacement = dict(task, shard=shard, shards=args.shards)
        uri = control + '/queue/' + identity + '.json'
        queue_fs, key = api.storage.filesystem(uri)
        if queue_fs.exists(key):
            if json.loads(queue_fs.cat(key)) != replacement:
                raise ValueError('Existing recovery descriptor conflicts')
        else:
            api.storage.write_json(replacement, uri)
        if identity not in recorded:
            new.append(dict(task_id=identity, control=control, source=original['source'], task=replacement))
    with (HERE / 'data/server_tasks.jsonl').open('a') as stream:
        for row in new:
            stream.write(json.dumps(row) + '\n')
    audit = dict(created_at=datetime.now(UTC).isoformat(), original_shards=12,
        recovery_shards=args.shards, source=original['source']['sha256'], plan_sha256=plan_sha,
        completion_markers_verified=len(done), migration_manifest='data/native_guided_migration.json',
        archived_pins=pins, control=control, result_root=destination.rsplit('/', 1)[0],
        scientific_change='None: source, cohort, policies, seeds and saved trajectories are unchanged.')
    (HERE / 'data/native_guided_repartition.json').write_text(json.dumps(audit, indent=2) + '\n')
    api.storage.write_json(audit, destination + f'/provenance/repartition-{args.shards}/audit.json')
    print(json.dumps({key: audit[key] for key in ['recovery_shards', 'completion_markers_verified', 'control']}))


if __name__ == '__main__':
    main()
