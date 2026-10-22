"""Evaluate threshold mixtures using complete, previously generated validation rollouts."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from analyze import evaluate
from metrics import bootstrap


def main() -> None:
    """Average balanced assignment rotations within proteins before uncertainty estimates."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--guided', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if plan['split'] != 'validation':
        raise ValueError('This exploratory analysis is restricted to validation')
    entries = plan['entries']
    targets = {r['entry_id']: r for r in pq.read_table(args.targets,
        filters=[('entry_id', 'in', entries)]).to_pylist()}
    components = sorted({name for mixture in plan['mixtures'] for name in mixture['components']})
    records, committed = [], []
    for entry in entries:
        target = targets[entry]
        if target['split'] != 'validation':
            raise ValueError('Mixture contains a non-validation target')
        samples, timings = {}, {}
        for component in components:
            key = args.guided / entry / component
            if not key.with_suffix('.done.json').exists():
                raise ValueError('Missing committed component: ' + str(key))
            rows = pq.read_table(key.with_suffix('.parquet')).to_pylist()
            if len(rows) != 4 or {r['sample'] for r in rows} != set(range(4)):
                raise ValueError('Expected exactly four complete component trajectories')
            for row in rows:
                identity = (component, row['sample'])
                samples[identity] = row
                timings[identity] = pd.read_csv(str(key) + f'-r{row["sample"]}.timings.csv').iloc[0]
        for mixture in plan['mixtures']:
            names = mixture['components']
            rotations = [[(names[(sample + offset) % len(names)], sample) for sample in range(4)]
                         for offset in range(len(names))]
            full = [[(name, sample) for name in names for sample in range(4)]]
            for assignments in [rotations, full]:
                for rotation, identities in enumerate(assignments):
                    rows = [samples[identity] for identity in identities]
                    cost = sum(timings[identity]['elapsed_seconds'] for identity in identities)
                    hardware = ';'.join(sorted({timings[identity]['gpu_name'] for identity in identities}))
                    for method in plan['methods']:
                        record = evaluate(rows, target, method, cost, mixture['name'], hardware)
                        records.append(dict(record, rotation=rotation,
                            assignment=json.dumps(identities), assignment_count=len(assignments)))
            committed.append(dict(entry_id=entry, policy=mixture['name'], complete=True))
    args.out.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(records)
    raw.to_csv(args.out / 'per_assignment.csv.gz', index=False)
    keys = ['entry_id', 'group_id', 'L', 'policy', 'rollouts', 'method', 'hardware']
    values = [c for c in raw.select_dtypes('number').columns if c not in keys + ['rotation']]
    # Rotations reuse trajectories. Their average is one observation per protein,
    # estimating a uniform assignment policy rather than selecting its best seed.
    frame = raw.groupby(keys, as_index=False)[values].mean()
    frame.to_csv(args.out / 'per_protein.csv.gz', index=False)
    pd.DataFrame(committed).to_csv(args.out / 'completeness.csv', index=False)
    summaries = []
    for (policy, count, method), rows in frame.groupby(['policy', 'rollouts', 'method']):
        record = dict(policy=policy, rollouts=count, method=method, structures=len(rows),
            groups=rows.group_id.nunique(), mean_seconds=rows.elapsed_seconds.mean())
        for metric in ['r_precision', 'long_r_precision', 'average_precision', 'oracle_coverage',
                       'long_oracle_coverage', 'precision_L', 'precision_L2', 'precision_L5']:
            record[metric] = rows[metric].mean()
            record[metric + '_low'], record[metric + '_high'] = bootstrap(
                rows[metric].tolist(), rows.group_id.tolist())
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out / 'summary.csv', index=False)
    (args.out / 'reuse_provenance.json').write_text(json.dumps(dict(
        source=str(args.guided), plan=str(args.plan), new_gpu_inference=False,
        rotation_handling='Average metrics and included-trajectory costs within each protein; '
                          'never treat rotations as independent proteins or select the best rotation.',
        timing_scope='Reused measured inference cost of each included trajectory; CPU aggregation excluded.'
    ), indent=2) + '\n')
    print(summary[['policy', 'rollouts', 'method', 'r_precision', 'long_r_precision',
                   'oracle_coverage', 'mean_seconds']].to_string(index=False))


if __name__ == '__main__':
    main()
