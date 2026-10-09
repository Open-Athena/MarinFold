"""Audit rollout completeness and publish the immutable training commit marker."""

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from storage import ROOT, filesystem, read_rows, upload_directory, write_json

HERE = Path(__file__).resolve().parent


def inspect_file(task: tuple[str, str, dict]) -> tuple[dict, list[dict], list[dict]]:
    """Inspect one committed batch and its contemporaneous per-input timings."""
    source, name, targets = task
    fs, root = filesystem(source)
    stem = name.removesuffix('.parquet')
    if not fs.exists(root + '/' + stem + '.done.json'):
        raise ValueError(f'Uncommitted rollout object: {name}')
    blob = fs.cat(root+'/'+name)
    rows = pq.read_table(pa.BufferReader(blob)).to_pylist()
    timings = read_rows(source + '/' + stem + '.timings.parquet')
    audit = []
    for row in rows:
        target = targets[row['entry_id']]
        truth = {tuple(pair) for pair in target['contacts']}
        seen = set()
        for pair,label,unique in zip(row['pairs'],row['labels'],row['unique'],strict=True):
            pair = tuple(pair)
            if label != float(pair in truth) or unique != (pair not in seen):
                raise ValueError(f"Incorrect label or duplicate accounting: {row['identity']}")
            seen.add(pair)
        k = len(row['contact_ends'])
        if not all(len(row[field]) == k for field in ('labels', 'pairs', 'unique')):
            raise ValueError(f"Contact/label alignment failure: {row['identity']}")
        if row['gt_count'] <= 0:
            raise ValueError('Missing ground truth')
        if row['contact_ends'] != sorted(set(row['contact_ends'])):
            raise ValueError('Contact endpoints must strictly increase')
        if k and max(row['contact_ends']) >= len(row['completion_ids']):
            raise ValueError('Contact readout outside completion')
        audit.append({key:row[key] for key in ('identity','entry_id','split','group_id','gt_count','finished','malformed','invalid_contacts')})
        audit[-1]['n_contacts'] = k
    return dict(name=name,size=len(blob),sha256=hashlib.sha256(blob).hexdigest()), audit, timings


def main() -> None:
    """Require every intended protein/rollout before allowing production training."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=ROOT + '/data/v1/rollouts')
    parser.add_argument('--targets', default=ROOT + '/data/v1/targets.parquet')
    parser.add_argument('--shards', type=int, default=32)
    parser.add_argument('--rollouts', type=int, default=8)
    parser.add_argument('--label', default='production')
    parser.add_argument('--report-out', help='Publish the small audit and timing ledgers from a cluster worker')
    args = parser.parse_args()
    fs, root = filesystem(args.source)
    missing = [i for i in range(args.shards) if not fs.exists(f'{root}/shard-{i:03d}.complete.json')]
    if missing:
        raise ValueError(f'Incomplete shards: {missing}')
    plans = [json.loads(fs.cat(f'{root}/shard-{i:03d}.plan.json')) for i in range(args.shards)]
    for i,plan in enumerate(plans):
        if (plan['shard'],plan['shards'],plan['rollouts'],plan['targets']) != (i,args.shards,args.rollouts,args.targets):
            raise ValueError('Shard generation configuration differs from the requested audit')
    targets = read_rows(args.targets)
    by_id = {r['entry_id']:r for r in targets}
    names = sorted(Path(p).name for p in fs.glob(root + '/shard-*-batch-*.parquet') if not p.endswith('.timings.parquet'))
    files, audit, timings = [], [], []
    with ThreadPoolExecutor(12) as pool:
        for item, records, measured in pool.map(inspect_file, [(args.source,n,by_id) for n in names]):
            files.append(item)
            audit.extend(records)
            timings.extend(measured)
    expected = {f"{r['entry_id']}:r{i}" for r in targets for i in range(args.rollouts)}
    actual = Counter(row['identity'] for row in audit)
    if set(actual) != expected or any(n != 1 for n in actual.values()):
        raise ValueError(f'Rollout coverage mismatch: expected {len(expected)}, actual {len(actual)}')
    for row in audit:
        target = by_id[row['entry_id']]
        if row['split'] != target['split'] or row['group_id'] != target['group_id'] or row['gt_count'] != len(target['contacts']):
            raise ValueError(f"Target provenance mismatch: {row['identity']}")
    counts = dict(structures=len(targets), rollouts=len(audit), files=len(files),
        malformed_rollouts=sum(r['malformed'] > 0 for r in audit),
        empty_rollouts=sum(r['n_contacts'] == 0 for r in audit),
        unfinished_rollouts=sum(not r['finished'] for r in audit),
        invalid_contacts=sum(r['invalid_contacts'] for r in audit))
    retained = [r for r in audit if r['malformed'] == 0 and r['n_contacts'] > 0]
    retained_ids = {r['entry_id'] for r in retained}
    if retained_ids != set(by_id):
        raise ValueError(f'{len(set(by_id)-retained_ids)} proteins have no valid training trajectory')
    counts['retained_rollouts'] = len(retained)
    counts['structures_by_split'] = dict(Counter(r['split'] for r in targets))
    manifest = dict(source=args.source, target_source=args.targets,
        target_fingerprint=hashlib.sha256(json.dumps(targets,sort_keys=True).encode()).hexdigest(),
        counts=counts, files=files,generation_plans=plans, sampling=dict(rollouts=args.rollouts,temperature=1.0,top_p=.95,top_k=-1,
            budget='min(6L+128,8192-prompt_tokens-1)', prefix_resampling=True))
    write_json(manifest, args.source + '/_SUCCESS.json')
    data = HERE / 'data'
    data.mkdir(exist_ok=True)
    (data / f'{args.label}_rollout_manifest.json').write_text(json.dumps(manifest,indent=2))
    pd.DataFrame(timings).to_csv(data / f'{args.label}_timings.csv', index=False)
    pd.DataFrame(audit).to_csv(data / f'{args.label}_rollout_audit.csv.gz', index=False)
    if args.report_out:
        upload_directory(data, args.report_out)
    print(json.dumps(counts,indent=2))


if __name__ == '__main__':
    main()
