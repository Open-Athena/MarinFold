"""Compute paired homology-group intervals on an explicitly complete cohort."""

import argparse
import json
from pathlib import Path

import pandas as pd

from metrics import bootstrap

METRICS = ['r_precision', 'long_r_precision', 'average_precision', 'long_average_precision',
           'oracle_coverage', 'long_oracle_coverage', 'precision_L', 'precision_L2', 'precision_L5']


def main() -> None:
    """Reject missing or duplicate observations before reporting paired effects."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--baseline', required=True, help='policy:rollouts:method')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    entries = set(json.loads(args.plan.read_text())['entries'])
    frame = pd.concat([pd.read_csv(path) for path in args.input], ignore_index=True)
    frame = frame[frame.entry_id.isin(entries)]
    keys = ['policy', 'rollouts', 'method']
    if frame.duplicated(keys + ['entry_id']).any():
        raise ValueError('Duplicate metrics across input tables')
    policy, n, method = args.baseline.split(':')
    baseline_key = (policy, int(n), method)
    baseline = frame[(frame.policy == policy) & (frame.rollouts == int(n)) & (frame.method == method)]
    if set(baseline.entry_id) != entries:
        raise ValueError('Baseline does not cover the required cohort')
    baseline = baseline.set_index('entry_id').sort_index()
    results, skipped = [], []
    for (policy, n, method), rows in frame.groupby(keys):
        if set(rows.entry_id) != entries:
            skipped.append(dict(policy=policy, rollouts=int(n), method=method,
                                missing=sorted(entries - set(rows.entry_id))))
            continue
        rows = rows.set_index('entry_id').sort_index()
        if not rows.group_id.equals(baseline.group_id):
            raise ValueError('Homology group assignment differs between methods')
        if (policy, n, method) == baseline_key:
            continue
        for metric in METRICS:
            delta = rows[metric] - baseline[metric]
            low, high = bootstrap(delta.tolist(), rows.group_id.tolist())
            results.append(dict(policy=policy, rollouts=n, method=method, baseline=args.baseline,
                metric=metric, structures=int(delta.notna().sum()), groups=rows.loc[delta.notna()].group_id.nunique(),
                baseline_mean=baseline[metric].mean(), candidate_mean=rows[metric].mean(),
                difference=delta.mean(), low=low, high=high, mean_seconds=rows.elapsed_seconds.mean(),
                baseline_seconds=baseline.elapsed_seconds.mean(),
                cost_ratio=rows.elapsed_seconds.sum() / baseline.elapsed_seconds.sum()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.out, index=False)
    args.out.with_suffix('.skipped.json').write_text(json.dumps(skipped, indent=2))
    print(f'Paired comparisons: {len(results)}; incomplete configurations skipped: {len(skipped)}')


if __name__ == '__main__':
    main()
